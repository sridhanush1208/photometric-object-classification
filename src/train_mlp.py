"""PyTorch MLP on quantile-normalised features, with an auxiliary redshift head (redshift is only a
training target here; at test time the inputs are photometry-derived features only).

  python src/train_mlp.py --features all --seeds 0,1,2 --name mlp_all"""
import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import QuantileTransformer

from common import (N_FOLDS, N_JOBS, SMOKE, WORK, Progress, load_work, log, mlogloss, save_preds,
                    select, split_folds)


class PeriodicEmb(nn.Module):
    """Periodic embeddings for numbers (Gorishniy et al. 2022, 'PLR'): each feature x becomes
    ReLU(W_f [sin(2πc_f x), cos(2πc_f x)]) with learned frequencies c_f, then all are concatenated."""

    def __init__(self, n_feat, n_freq, d_out, sigma):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n_feat, n_freq) * sigma)
        self.w = nn.Parameter(torch.randn(n_feat, 2 * n_freq, d_out) / (2 * n_freq) ** 0.5)
        self.b = nn.Parameter(torch.zeros(n_feat, d_out))

    def forward(self, x):
        v = 2 * np.pi * x.unsqueeze(-1) * self.c                      # (B, F, K)
        z = torch.cat([torch.sin(v), torch.cos(v)], -1)                # (B, F, 2K)
        return torch.relu(torch.einsum('bfk,fkd->bfd', z, self.w) + self.b).flatten(1)


class MLP(nn.Module):
    def __init__(self, d_in, hidden, layers, dropout, emb=None):
        super().__init__()
        self.emb = None
        if emb and emb[0] > 0:                                         # emb = (n_freq, d_out, sigma)
            self.emb = PeriodicEmb(d_in, *emb)
            d_in = d_in * emb[1]
        blocks, d = [], d_in
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(dropout)]
            d = hidden
        self.body = nn.Sequential(*blocks)
        self.cls = nn.Linear(d, 3)
        self.z = nn.Linear(d, 1)

    def forward(self, x):
        if self.emb is not None:
            x = self.emb(x)
        h = self.body(x)
        return self.cls(h), self.z(h).squeeze(1)


def predict(model, X, dev, bs=8192):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.softmax(model(X[i:i + bs].to(dev))[0], 1).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def train_fold(Xa, ya, za, Xb, yb, Xt, a, seed, dev):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(Xa.shape[1], a.hidden, a.layers, a.dropout,
                (a.emb_freq, a.emb_dim, a.emb_sigma)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = len(Xa) // a.bs                    # drop the last partial batch (BatchNorm needs > 1 row)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps,
                                                pct_start=0.1)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    Xa_d, ya_d, za_d = Xa.to(dev), ya.to(dev), za.to(dev)
    g = torch.Generator(device='cpu').manual_seed(seed)
    ema = None
    if a.ema > 0:                       # evaluate an exponential moving average of the weights
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(a.ema), use_buffers=True)
    best, best_state, bad = 1e9, None, 0
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Xa), generator=g).to(dev)
        for i in range(steps):
            idx = perm[i * a.bs:(i + 1) * a.bs]
            xb = Xa_d[idx]
            if a.noise > 0:                 # small Gaussian jitter on the (quantile-normal) inputs
                xb = xb + a.noise * torch.randn_like(xb)
            logit, zp = model(xb)
            loss = ce(logit, ya_d[idx]) + a.aux * mse(zp, za_d[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if ema is not None:
                ema.update_parameters(model)
        evm = ema.module if ema is not None else model
        vl = mlogloss(yb.numpy(), predict(evm, Xb, dev))
        if vl < best - 1e-5:
            best, best_state, bad = vl, copy.deepcopy(evm.state_dict()), 0
        else:
            bad += 1
        if bad >= a.patience:
            break
    model.load_state_dict(best_state)
    return predict(model, Xb, dev), predict(model, Xt, dev), best, ep + 1


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='all')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--dropout', type=float, default=0.15)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--bs', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--aux', type=float, default=0.2, help='weight of the redshift loss (0 = off)')
    ap.add_argument('--emb-freq', type=int, default=0, help='periodic embeddings: frequencies (0 = off)')
    ap.add_argument('--emb-dim', type=int, default=16)
    ap.add_argument('--emb-sigma', type=float, default=0.3)
    ap.add_argument('--scaler', default='quantile', choices=['quantile', 'robust'])
    ap.add_argument('--nfolds', type=int, default=N_FOLDS, help='5, 10 or 20 (more folds = more training rows per model)')
    ap.add_argument('--nested', action='store_true',
                    help='add the per-fold clean photo-z features from src/nested.py (5 folds only)')
    ap.add_argument('--noise', type=float, default=0.0, help='Gaussian input noise during training (0 = off)')
    ap.add_argument('--ema', type=float, default=0.0, help='EMA decay for the weights, e.g. 0.999 (0 = off)')
    ap.add_argument('--only-fold', type=int, default=None, help='quick experiment: train one fold, save nothing')
    ap.add_argument('--name', default=None)
    return ap


def load_inputs(spec, scaler='quantile'):
    """Normalised features (fit on train+test, label-free), labels, redshift target, folds.
    scaler='quantile': rank -> normal (default). scaler='robust': clip to the 0.5-99.5 percentiles,
    then (x - median) / IQR; keeps distances between values, so the net learns different patterns."""
    tr, te, y, fold, groups = load_work()
    cols = select(groups, spec)
    allX = np.vstack([tr[cols].values, te[cols].values]).astype(np.float64)
    allX = np.nan_to_num(allX, nan=0.0, posinf=1e6, neginf=-1e6)
    if scaler == 'robust':
        lo, hi = np.percentile(allX, [0.5, 99.5], axis=0)
        allX = np.clip(allX, lo, hi)
        q25, q50, q75 = np.percentile(allX, [25, 50, 75], axis=0)
        allX = ((allX - q50) / np.maximum(q75 - q25, 1e-9)).astype(np.float32)
    else:
        qt = QuantileTransformer(n_quantiles=2000, output_distribution='normal', subsample=10 ** 9,
                                 random_state=0).fit(allX)
        allX = qt.transform(allX).astype(np.float32)
    z = np.log1p(np.clip(np.load(WORK / 'redshift.npy'), 0, None))
    z = (z - z.mean()) / z.std()
    return (torch.from_numpy(allX[:len(tr)]), torch.from_numpy(allX[len(tr):]), y, torch.from_numpy(y),
            torch.from_numpy(z.astype(np.float32)), fold, cols)


def main(argv=None):
    a = parser().parse_args(argv)
    if SMOKE:
        a.epochs = 3
    torch.set_num_threads(N_JOBS)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X, Xt, y, yt, zt, fold, cols = load_inputs(a.features, a.scaler)
    assert not (a.nested and a.nfolds != N_FOLDS), '--nested needs the 5 original folds'
    fold = split_folds(fold, y, a.nfolds)
    nf = a.nfolds

    def fold_inputs(f, A, B):
        """Inputs for fold f; with --nested, append that fold's clean photo-z features, quantile-
        normalised with a transform fitted on the fold's training rows only."""
        if not a.nested:
            return X[A], X[B], Xt
        from nested import load_fold
        La, Lb, Lt, _ = load_fold(f)
        q = QuantileTransformer(n_quantiles=2000, output_distribution='normal', subsample=10 ** 9,
                                random_state=0).fit(La)
        t = lambda M: torch.from_numpy(q.transform(M).astype(np.float32))  # noqa: E731
        return torch.cat([X[A], t(La)], 1), torch.cat([X[B], t(Lb)], 1), torch.cat([Xt, t(Lt)], 1)
    name = a.name or f'mlp_{a.features.replace(",", "_")}'
    seeds = [int(s) for s in a.seeds.split(',')]
    log(f'{name}: {len(cols)} features on {dev}, seeds {seeds}, {vars(a)}')

    if a.only_fold is not None:
        f = a.only_fold
        A, B = torch.from_numpy(fold != f), torch.from_numpy(fold == f)
        Xa, Xb, Xtf = fold_inputs(f, A, B)
        vl = np.mean([train_fold(Xa, yt[A], zt[A], Xb, yt[B], Xtf[:2], a, 1000 * s + f, dev)[2]
                      for s in seeds])
        log(f'RESULT {name} fold {f}: {vl:.5f}')
        return vl

    oof = np.zeros((len(X), 3))
    tst = np.zeros((len(Xt), 3))
    prog = Progress(name, len(seeds) * nf)
    for s in seeds:
        o = np.zeros((len(X), 3))
        for f in range(nf):
            A = torch.from_numpy(fold != f)
            B = torch.from_numpy(fold == f)
            Xa, Xb, Xtf = fold_inputs(f, A, B)
            pb, pt, vl, ep = train_fold(Xa, yt[A], zt[A], Xb, yt[B], Xtf, a, 1000 * s + f, dev)
            o[B.numpy()] = pb
            tst += pt / (nf * len(seeds))
            prog.step(f'seed {s} fold {f}: {vl:.5f} ({ep} epochs)')
        log(f'  seed {s}: CV {mlogloss(y, o):.5f}')
        oof += o / len(seeds)
    return save_preds(name, oof, tst, y, dict(model='mlp', features=a.features, n_features=len(cols),
                                              nested=a.nested, seeds=a.seeds, nfolds=nf, args=vars(a)))


if __name__ == '__main__':
    main()
