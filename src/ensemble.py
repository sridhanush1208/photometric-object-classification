"""Blend saved model predictions, calibrate, and write a Kaggle submission.

  python src/ensemble.py                       # every model in work/preds
  python src/ensemble.py --models lgb_all,mlp_all --tag v2

Three blenders are scored with nested CV on the fixed folds (weights fit on 4 folds, scored on the
5th), and the best one is used:
  linear  - weighted average of probabilities (+ temperature scaling)
  geo     - weighted average of log-probabilities + per-class bias (calibration built in)
  stack   - multinomial logistic regression on all models' log-probabilities"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from sklearn.linear_model import LogisticRegression

from common import CLASSES, N_FOLDS, PREDS, SMOKE, SUBS, WORK, load_work, log, mlogloss


def fit_linear(P, y, E=None):
    """P: (models, n, 3). Softmax-parametrised weights, then one temperature."""
    M = len(P)

    def blend(w, Q):
        return np.tensordot(softmax(w), Q, 1)

    w = minimize(lambda w: mlogloss(y, blend(w, P)), np.zeros(M), method='L-BFGS-B').x
    lb = np.log(np.clip(blend(w, P), 1e-15, 1))
    T = minimize(lambda t: mlogloss(y, softmax(lb / t[0], 1)), [1.0], bounds=[(0.5, 3)],
                 method='L-BFGS-B').x[0]

    def apply(Q, Eq=None):
        return softmax(np.log(np.clip(blend(w, Q), 1e-15, 1)) / T, 1)
    return apply, dict(weights=softmax(w).round(4).tolist(), T=round(float(T), 4))


def fit_geo(P, y, E=None):
    L = np.log(np.clip(P, 1e-15, 1))
    M = len(P)

    def blend(th, Q):
        return log_softmax(np.tensordot(th[:M], Q, 1) + th[M:], 1)

    Y = np.eye(3)[y]
    n = len(y)

    def nll(th):                       # loss and its exact gradient (much faster than numerical)
        lp = blend(th, L)
        G = (np.exp(lp) - Y) / n                     # d loss / d logits
        return -(lp * Y).sum() / n, np.r_[np.einsum('mnc,nc->m', L, G), G.sum(0)]

    th = minimize(nll, np.r_[np.ones(M) / M, np.zeros(3)], jac=True, method='L-BFGS-B').x

    def apply(Q, Eq=None):
        return np.exp(blend(th, np.log(np.clip(Q, 1e-15, 1))))
    return apply, dict(weights=th[:M].round(4).tolist(), bias=th[M:].round(4).tolist())


def fit_geo_bounded(P, y, E=None, l2=1e-4):
    """V8 (from the 1 Nov audit): like 'geo' but every model weight is kept in [0, 1.2] and gently
    pulled toward equal weights, so correlated models can't get offsetting negative weights.
    Offered as an extra candidate; nested CV decides whether it is used."""
    L = np.log(np.clip(P, 1e-15, 1))
    M = len(P)
    Y = np.eye(3)[y]
    n = len(y)
    u = np.ones(M) / M

    def blend(th, Q):
        return log_softmax(np.tensordot(th[:M], Q, 1) + th[M:], 1)

    def nll(th):
        lp = blend(th, L)
        G = (np.exp(lp) - Y) / n
        return (-(lp * Y).sum() / n + 0.5 * l2 * ((th[:M] - u) ** 2).sum(),
                np.r_[np.einsum('mnc,nc->m', L, G) + l2 * (th[:M] - u), G.sum(0)])

    th = minimize(nll, np.r_[u, np.zeros(3)], jac=True, method='L-BFGS-B',
                  bounds=[(0.0, 1.2)] * M + [(-0.5, 0.5)] * 3).x

    def apply(Q, Eq=None):
        return np.exp(blend(th, np.log(np.clip(Q, 1e-15, 1))))
    return apply, dict(weights=th[:M].round(4).tolist(), bias=th[M:].round(4).tolist())


def fit_stack(P, y, E=None, C=1.0):
    def feats(Q):
        L = np.log(np.clip(Q, 1e-15, 1))
        return np.concatenate(list(L), 1)
    lr = LogisticRegression(C=C, max_iter=2000).fit(feats(P), y)

    def apply(Q, Eq=None):
        return lr.predict_proba(feats(Q))
    return apply, dict(C=C)


META_COLS = ['r', 'size_res', 'abs_b', 'u-g', 'g-r', 'r-i', 'eres_mean', 'loc_size_q20']


def fit_meta(P, y, E, residual=True):
    """Small LightGBM on every model's log-probabilities + a few key features, so the blend can
    trust different models for faint/bright, small/large, blue/red objects. With residual=True it
    starts from the 'geo' blend (init_score) and only learns corrections on top of it. Early
    stopping uses an inner 15% holdout of the rows it is given (never the rows it is scored on)."""
    import lightgbm as lgb
    from common import N_JOBS

    def feats(Q, Eq):
        return np.concatenate(list(np.log(np.clip(Q, 1e-15, 1))) + [Eq], 1)
    base_apply, _ = fit_geo(P, y) if residual else (None, None)

    def base(Q):
        return np.log(np.clip(base_apply(Q), 1e-15, 1)) if residual else None
    X, B = feats(P, E), base(P)
    prm = dict(objective='multiclass', num_class=3, learning_rate=0.02, num_leaves=15,
               min_child_samples=300, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
               lambda_l2=10.0, verbose=-1, num_threads=N_JOBS, seed=0)
    hold = np.random.default_rng(0).random(len(y)) < 0.15

    def ds(rows):
        return lgb.Dataset(X[rows], y[rows], init_score=None if B is None else B[rows])
    m = lgb.train(prm, ds(~hold), 3000, valid_sets=[ds(hold)], callbacks=[lgb.early_stopping(100, verbose=False)])
    it = max(1, int(m.best_iteration * 1.1))
    m = lgb.train(prm, ds(np.ones(len(y), bool)), it)

    def apply(Q, Eq=None):
        raw = m.predict(feats(Q, Eq), raw_score=True)
        return softmax(raw + base(Q), 1) if residual else softmax(raw, 1)
    return apply, dict(rounds=it, residual=residual)


def nested(fitter, P, y, fold, E):
    oof = np.zeros((P.shape[1], 3))
    for f in range(N_FOLDS):
        a, b = fold != f, fold == f
        apply, _ = fitter(P[:, a], y[a], E[a])
        oof[b] = apply(P[:, b], E[b])
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='all')
    ap.add_argument('--exclude', default='')
    ap.add_argument('--tag', default=time.strftime('%m%d_%H%M'))
    ap.add_argument('--out', default=None, help='also write the submission here (Kaggle: submission.csv)')
    ap.add_argument('--no-meta', action='store_true', help='skip the LightGBM meta-blender')
    a = ap.parse_args()

    tr, te, y, fold, _ = load_work()
    names = sorted(p.stem for p in PREDS.glob('*.npz')) if a.models == 'all' else a.models.split(',')
    names = [n for n in names if n not in a.exclude.split(',')]
    P, Pt = [], []
    for n in names:
        d = np.load(PREDS / f'{n}.npz')
        P.append(d['oof'].astype(np.float64))
        Pt.append(d['test'].astype(np.float64))
    P, Pt = np.stack(P), np.stack(Pt)
    P /= P.sum(2, keepdims=True)
    Pt /= Pt.sum(2, keepdims=True)
    log('single models (CV log loss):')
    for n, p in sorted(zip(names, P), key=lambda t: mlogloss(y, t[1])):
        log(f'  {mlogloss(y, p):.5f}  {n}')

    mc = [c for c in META_COLS if c in tr.columns]
    E, Et = tr[mc].values.astype(np.float64), te[mc].values.astype(np.float64)
    fitters = {'linear': fit_linear, 'geo': fit_geo, 'geo_bounded': fit_geo_bounded}
    for C in (0.1, 1.0):
        fitters[f'stack_C{C}'] = lambda P_, y_, E_=None, C=C: fit_stack(P_, y_, E_, C)
    if not a.no_meta:
        fitters['meta_lgb'] = fit_meta
    scores, oofs = {}, {}
    for k, fn in fitters.items():
        oofs[k] = nested(fn, P, y, fold, E)
        scores[k] = mlogloss(y, oofs[k])
        log(f'blend {k:10s} nested-CV {scores[k]:.5f}')
    best = min(scores, key=scores.get)
    apply, info = fitters[best](P, y, E)
    oof, test = oofs[best], apply(Pt, Et)

    # clipping: pick the floor on nested OOF predictions
    def clip(p, e):
        p = np.clip(p, e, 1)
        return p / p.sum(1, keepdims=True)
    eps = min([0, 1e-6, 1e-5, 1e-4], key=lambda e: mlogloss(y, clip(oof, e) if e else oof))
    if eps:
        oof, test = clip(oof, eps), clip(test, eps)
    final = mlogloss(y, oof)
    log(f'chosen: {best} {info} | clip {eps} | nested-CV {final:.5f}')

    # where is the loss now?
    loss = -np.log(np.clip(oof[np.arange(len(y)), y], 1e-15, 1))
    df = pd.DataFrame(dict(cls=np.array(CLASSES)[y], r=tr['r'].values, loss=loss))
    df['r_bin'] = pd.cut(df.r, [0, 18, 19, 20, 21, 22, 40])
    rep = df.pivot_table(index='r_bin', columns='cls', values='loss', aggfunc='mean', observed=True)
    log('mean loss by r magnitude x true class:\n' + rep.round(3).to_string())
    log('share of total loss by class: ' +
        ', '.join(f'{c} {df.loss[df.cls == c].sum() / df.loss.sum():.3f}' for c in CLASSES))

    ids = np.load(WORK / 'test_ids.npy')
    sub = pd.DataFrame(test, columns=CLASSES)
    sub.insert(0, 'object_id', ids)
    assert not sub.isna().any().any() and np.allclose(sub[CLASSES].sum(1), 1, atol=1e-6)
    assert SMOKE or len(sub) == 31000, len(sub)
    path = SUBS / f'sub_{a.tag}_cv{final:.5f}.csv'
    sub.to_csv(path, index=False)
    if a.out:
        sub.to_csv(a.out, index=False)
        log(f'WROTE {a.out}')
    (WORK / 'ensembles').mkdir(exist_ok=True)
    np.savez_compressed(WORK / 'ensembles' / f'{a.tag}.npz', oof=oof, test=test)
    with open(WORK / 'ensembles.jsonl', 'a') as f:
        f.write(json.dumps(dict(tag=a.tag, models=names, method=best, info=info, clip=eps,
                                nested_cv=round(final, 6), all=scores, file=path.name)) + '\n')
    log(f'WROTE {path}')


if __name__ == '__main__':
    main()
