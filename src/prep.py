"""Step 1: fixed folds, label-free features, and out-of-fold 'learned' features:
  photoz    - redshift predicted from photometry (allowed: confirmed by the coordinator; the real
              redshift is only ever a training target, never an input)
  knn_sky   - class mix of the nearest labelled objects on the sky (SDSS plates target one kind of
              object, so neighbours are informative)
  knn_color - class mix of the nearest labelled objects in colour/shape space
Train rows get out-of-fold values; test rows get the average over the 5 fold models, so both see
the same amount of labelled data."""
import json
import os

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

import features
from common import (CLASSES, FOLD_SEED, N_FOLDS, N_JOBS, SMOKE, WORK, Progress, labels, load_raw, log)

PROG = None   # set in main(): one progress counter across all 20 fold-steps

Z_BINS = [-1, 0.005, 0.3, 0.7, 1.5, 2.3, 3.0, 99]       # star | galaxy-ish | ... | high-z quasar


def oof_lgb(X, Xt, target, fold, params, rounds):
    """Out-of-fold predictions for train, fold-averaged predictions for test."""
    multi = params['objective'] == 'multiclass'
    k = params.get('num_class', 1)
    oof = np.zeros((len(X), k)) if multi else np.zeros(len(X))
    tst = np.zeros((len(Xt), k)) if multi else np.zeros(len(Xt))
    for f in range(N_FOLDS):
        a, b = fold != f, fold == f
        m = lgb.train(params, lgb.Dataset(X[a], target[a]), rounds,
                      valid_sets=[lgb.Dataset(X[b], target[b])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[b] = m.predict(X[b], num_iteration=m.best_iteration)
        tst += m.predict(Xt, num_iteration=m.best_iteration) / N_FOLDS
        PROG.step(f'{params["objective"]} fold {f} (best_iter {m.best_iteration})')
    return oof, tst


def oof_knn(Z, Zt, y, fold, ks, prefix):
    """Class fractions among the k nearest labelled neighbours, plus distance to the k-th."""
    kmax = max(ks)
    out = {f'{prefix}_k{k}_{c}': np.zeros(len(Z)) for k in ks for c in CLASSES}
    out.update({f'{prefix}_d{k}': np.zeros(len(Z)) for k in ks})
    outt = {n: np.zeros(len(Zt)) for n in out}
    for f in range(N_FOLDS):
        a, b = fold != f, fold == f
        nn = NearestNeighbors(n_neighbors=kmax, n_jobs=N_JOBS).fit(Z[a])
        ya = y[a]
        for rows, dst, w in ((b, out, None), (slice(None), outt, 1.0 / N_FOLDS)):
            q = Z[rows] if dst is out else Zt
            d, idx = nn.kneighbors(q)
            lab = ya[idx]
            for k in ks:
                for ci, c in enumerate(CLASSES):
                    v = (lab[:, :k] == ci).mean(1)
                    if w is None:
                        dst[f'{prefix}_k{k}_{c}'][rows] = v
                    else:
                        dst[f'{prefix}_k{k}_{c}'] += w * v
                v = np.log(d[:, k - 1] + 1e-9)
                if w is None:
                    dst[f'{prefix}_d{k}'][rows] = v
                else:
                    dst[f'{prefix}_d{k}'] += w * v
        PROG.step(f'{prefix} neighbours fold {f}')
    return out, outt


def main():
    global PROG
    PROG = Progress('prep', 4 * N_FOLDS)
    tr, te = load_raw()
    y = labels(tr)
    fold = np.zeros(len(tr), dtype=np.int64)
    for f, (_, b) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(tr, y)):
        fold[b] = f
    np.save(WORK / 'y.npy', y)
    np.save(WORK / 'fold.npy', fold)
    np.save(WORK / 'test_ids.npy', te['object_id'].values)
    np.save(WORK / 'redshift.npy', tr['redshift'].values)   # training target only (MLP aux head)

    log('label-free features')
    Xtr, Xte, groups = features.build(tr, te)
    base = [c for g in ('raw', 'color', 'flux', 'morph', 'errres', 'sky') for c in groups[g]]
    A, At = Xtr[base].values, Xte[base].values
    rounds = 200 if SMOKE else 5000

    # V8: the old out-of-fold learned features are unused by the current models (they use the clean
    # nested build instead); CS_PREP_LEARNED=0 skips them and saves ~3-4 min on Kaggle.
    if os.environ.get('CS_PREP_LEARNED', '1') == '1':
        log('photo-z regression (target: log1p redshift)')
        zt = np.log1p(np.clip(tr['redshift'].values, 0, None))
        # lr 0.1 / 63 leaves: ~2x faster than lr 0.05 / 127 leaves, and photo-z barely changes the score
        p = dict(objective='regression', learning_rate=0.1, num_leaves=63, min_child_samples=20,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                 verbose=-1, num_threads=N_JOBS, seed=0)
        pz, pzt = oof_lgb(A, At, zt, fold, p, rounds)
        log(f'  photo-z RMSE (log1p z): {np.sqrt(np.mean((pz - zt) ** 2)):.4f}')
        Xtr['pz'], Xte['pz'] = pz, pzt
        groups['photoz'] = ['pz']

        log('photo-z bins (multiclass over redshift ranges)')
        zb = np.digitize(tr['redshift'].values, Z_BINS[1:-1])
        p = dict(p, objective='multiclass', num_class=len(Z_BINS) - 1)
        pb, pbt = oof_lgb(A, At, zb, fold, p, rounds)
        for k in range(pb.shape[1]):
            Xtr[f'pzb_{k}'], Xte[f'pzb_{k}'] = pb[:, k], pbt[:, k]
            groups['photoz'].append(f'pzb_{k}')
        Xtr['pzb_mean'] = pb @ np.arange(pb.shape[1])
        Xte['pzb_mean'] = pbt @ np.arange(pb.shape[1])
        groups['photoz'].append('pzb_mean')

        log('sky neighbours')
        V, Vt = features.sky_vectors(tr), features.sky_vectors(te)
        o, ot = oof_knn(V, Vt, y, fold, [10, 30, 100], 'sky')
        for n in o:
            Xtr[n], Xte[n] = o[n], ot[n]
        groups['knn_sky'] = list(o)

        log('colour-space neighbours')
        cc = ['u-g', 'g-r', 'r-i', 'i-z', 'size_res', 'r']
        mu, sd = Xtr[cc].mean(), Xtr[cc].std()
        Z, Zt = ((Xtr[cc] - mu) / sd).values, ((Xte[cc] - mu) / sd).values
        Z[:, -1] *= 0.5
        Zt[:, -1] *= 0.5                                         # brightness counts half as much as colour
        o, ot = oof_knn(Z, Zt, y, fold, [25, 100], 'cknn')
        for n in o:
            Xtr[n], Xte[n] = o[n], ot[n]
        groups['knn_color'] = list(o)

    Xtr = Xtr.astype(np.float32)
    Xte = Xte.astype(np.float32)
    Xtr.to_pickle(WORK / 'feat_train.pkl')
    Xte.to_pickle(WORK / 'feat_test.pkl')
    (WORK / 'feature_groups.json').write_text(json.dumps(groups, indent=1))
    log(f'done: train {Xtr.shape}, test {Xte.shape}, groups ' +
        ', '.join(f'{g}={len(c)}' for g, c in groups.items()))


if __name__ == '__main__':
    main()
