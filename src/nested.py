"""Nested ('clean') learned features: photo-z + neighbour class mix, rebuilt separately for every outer
fold f using ONLY the labels of that fold's training rows.

Why (31 Oct leak test, holdout = fold 0, same LightGBM): learned features built by prep.py are
out-of-fold over the 5 folds, so a training row's feature was computed with the validation rows'
labels. Rebuilt cleanly, the all-features model scored 0.1556 vs 0.1583 (standard) and 0.1582
(no learned features). The contamination was hiding a real signal.

For outer fold f (train rows R = folds != f):
  R rows    -> out-of-fold over R's own 4 folds (inner CV)
  f rows    -> average of the 4 inner models (like test)
  test rows -> average of the 4 inner models
Saved to work/nested/f{f}.npz (train / val / test arrays + column names). Used by
train_gbdt.py --nested and train_mlp.py --nested.

  python src/nested.py"""
import os

import numpy as np

import features
import prep
from common import N_FOLDS, N_JOBS, SMOKE, WORK, Progress, load_raw, load_work, log

OUT = WORK / os.environ.get('CS_NESTED_DIR', 'nested')
# 31 Oct: all the gain comes from the photo-z features (pz + 7 redshift-range probabilities);
# the neighbour class mix added nothing, so it is off by default (CS_NESTED_KNN=1 turns it on).
WITH_KNN = os.environ.get('CS_NESTED_KNN', '0') == '1'
FAST = os.environ.get('CS_NESTED_FAST', '1') == '1'     # lr 0.2 / 31 leaves / 127 bins: ~3x faster


def build_fold(f, tr, te, y, fold, raw_tr, raw_te, groups):
    R, V = fold != f, fold == f
    inner = fold[R] - (fold[R] > f)                          # R's folds renumbered 0..3
    prep.N_FOLDS = N_FOLDS - 1
    lab_free = [c for g in ('raw', 'color', 'flux', 'morph', 'errres', 'sky') for c in groups[g]]
    A = tr.loc[R, lab_free].values
    O = np.vstack([tr.loc[V, lab_free].values, te[lab_free].values])   # val rows + test rows
    nV = V.sum()
    cols, trainF, otherF = [], [], []

    def put(name, a, b):
        cols.append(name)
        trainF.append(a)
        otherF.append(b)

    rounds = 200 if SMOKE else 5000
    zt = np.log1p(np.clip(raw_tr['redshift'].values, 0, None))[R]
    p = dict(objective='regression', learning_rate=0.1, num_leaves=63, min_child_samples=20, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=N_JOBS, seed=0)
    if FAST:
        p.update(learning_rate=0.2, num_leaves=31, max_bin=127)
    a, b = prep.oof_lgb(A, O, zt, inner, p, rounds)
    put('pz', a, b)
    zb = np.digitize(raw_tr['redshift'].values, prep.Z_BINS[1:-1])[R]
    a, b = prep.oof_lgb(A, O, zb, inner, dict(p, objective='multiclass', num_class=len(prep.Z_BINS) - 1), rounds)
    for k in range(a.shape[1]):
        put(f'pzb_{k}', a[:, k], b[:, k])
    put('pzb_mean', a @ np.arange(a.shape[1]), b @ np.arange(a.shape[1]))

    if not WITH_KNN:
        return save(f, cols, trainF, otherF, nV)
    Vall = features.sky_vectors(raw_tr)
    Vo = np.vstack([Vall[V], features.sky_vectors(raw_te)])
    o, oo = prep.oof_knn(Vall[R], Vo, y[R], inner, [10, 30, 100], 'sky')
    for n in o:
        put(n, o[n], oo[n])
    cc = ['u-g', 'g-r', 'r-i', 'i-z', 'size_res', 'r']
    mu, sd = tr.loc[R, cc].mean(), tr.loc[R, cc].std()
    Z = ((tr[cc] - mu) / sd).values
    Zt = ((te[cc] - mu) / sd).values
    Z[:, -1] *= 0.5
    Zt[:, -1] *= 0.5
    o, oo = prep.oof_knn(Z[R], np.vstack([Z[V], Zt]), y[R], inner, [25, 100], 'cknn')
    for n in o:
        put(n, o[n], oo[n])

    return save(f, cols, trainF, otherF, nV)


def save(f, cols, trainF, otherF, nV):
    T = np.stack(trainF, 1).astype(np.float32)
    O = np.stack(otherF, 1).astype(np.float32)
    np.savez_compressed(OUT / f'f{f}.npz', train=T, val=O[:nV], test=O[nV:], cols=np.array(cols))
    prep.N_FOLDS = N_FOLDS
    return cols


def load_fold(f):
    d = np.load(OUT / f'f{f}.npz')
    return d['train'], d['val'], d['test'], list(d['cols'])


def main():
    tr, te, y, fold, groups = load_work()
    raw_tr, raw_te = load_raw()
    OUT.mkdir(exist_ok=True)
    prep.PROG = Progress('nested learned features', N_FOLDS * (4 if WITH_KNN else 2) * (N_FOLDS - 1))
    for f in range(N_FOLDS):
        cols = build_fold(f, tr, te, y, fold, raw_tr, raw_te, groups)
        log(f'outer fold {f}: {len(cols)} learned features saved')


if __name__ == '__main__':
    main()
