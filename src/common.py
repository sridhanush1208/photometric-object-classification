"""Shared paths, constants and helpers for the Cosmic Sort pipeline."""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('CS_ROOT', Path(__file__).resolve().parents[1]))
DATA = Path(os.environ.get('CS_DATA', ROOT / 'data'))
WORK = Path(os.environ.get('CS_WORK', ROOT / 'work'))
PREDS = Path(os.environ.get('CS_PREDS', WORK / 'preds'))
PARAMS = WORK / 'params'
SUBS = Path(os.environ.get('CS_SUBS', ROOT / 'subs'))
for _d in (WORK, PREDS, PARAMS, SUBS):
    _d.mkdir(parents=True, exist_ok=True)

CLASSES = ['STAR', 'GALAXY', 'QSO']          # column order of every probability array
BANDS = ['u', 'g', 'r', 'i', 'z']
N_FOLDS = 5
FOLD_SEED = 42
N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 4))
SMOKE = os.environ.get('CS_SMOKE', '0') == '1'   # tiny run to test the whole pipeline end to end
EPS = 1e-15


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def fmt_secs(s):
    s = int(s)
    return f'{s // 3600}h{s % 3600 // 60:02d}m' if s >= 3600 else f'{s // 60}m{s % 60:02d}s'


class Progress:
    """Prints '[name] 3/15 <msg> | elapsed 2m10s | ~4m20s left' after each unit of work."""

    def __init__(self, name, total):
        self.name, self.total, self.done, self.t0 = name, total, 0, time.time()

    def step(self, msg=''):
        self.done += 1
        el = time.time() - self.t0
        left = el / self.done * (self.total - self.done)
        log(f'[{self.name}] {self.done}/{self.total} {msg} | elapsed {fmt_secs(el)} | ~{fmt_secs(left)} left')


def load_raw():
    tr = pd.read_csv(DATA / 'train.csv')
    te = pd.read_csv(DATA / 'test.csv')
    if SMOKE:
        tr = tr.sample(6000, random_state=0).reset_index(drop=True)
        te = te.sample(2000, random_state=0).reset_index(drop=True)
    return tr, te


def labels(tr):
    return tr['class'].map({c: i for i, c in enumerate(CLASSES)}).values.astype(np.int64)


def mlogloss(y, p):
    p = np.clip(p / p.sum(1, keepdims=True), EPS, 1)
    return float(-np.log(p[np.arange(len(y)), y]).mean())


def load_work():
    """Features built by prep.py: train/test frames, labels, fold ids, feature groups."""
    tr = pd.read_pickle(WORK / 'feat_train.pkl')
    te = pd.read_pickle(WORK / 'feat_test.pkl')
    y = np.load(WORK / 'y.npy')
    fold = np.load(WORK / 'fold.npy')
    groups = json.loads((WORK / 'feature_groups.json').read_text())
    return tr, te, y, fold, groups


def split_folds(fold, y, n):
    """n=5: the fixed folds. n=10 or 20: each fixed fold split into n/5 parts (stratified, seed 42), so
    models train on 90% / 95% of rows while every prediction is still out-of-fold and the 5-fold blend
    still works. (n=10 gives exactly the same split as before.)"""
    if n == N_FOLDS:
        return fold
    k = n // N_FOLDS
    assert n % N_FOLDS == 0 and k >= 2, 'nfolds must be 5, 10, 15, 20, ...'
    from sklearn.model_selection import StratifiedKFold
    out = np.zeros_like(fold)
    for f in range(N_FOLDS):
        idx = np.where(fold == f)[0]
        for j, (_, b) in enumerate(StratifiedKFold(k, shuffle=True, random_state=FOLD_SEED).split(idx, y[idx])):
            out[idx[b]] = k * f + j
    return out


def select(groups, spec):
    """spec: 'all', 'best' (chosen by ablate.py), or a comma list of group names; '-name' removes a
    group (e.g. 'all,-knn_color')."""
    if spec.startswith('best'):
        f = WORK / 'best_features.txt'
        spec = (f.read_text().strip() if f.exists() else 'all') + spec[4:]
    names = list(groups) if spec.startswith('all') else []
    for tok in spec.split(','):
        tok = tok.strip()
        if tok in ('', 'all'):
            continue
        if tok.startswith('-'):
            names.remove(tok[1:])
        else:
            names.append(tok)
    cols = []
    for n in names:
        cols += [c for c in groups[n] if c not in cols]
    return cols


def save_preds(name, oof, test, y, info=None):
    score = mlogloss(y, oof)
    np.savez_compressed(PREDS / f'{name}.npz', oof=oof.astype(np.float32), test=test.astype(np.float32))
    row = dict(name=name, cv=round(score, 6), time=time.strftime('%Y-%m-%d %H:%M'), **(info or {}))
    with open(WORK / 'results.jsonl', 'a') as f:
        f.write(json.dumps(row) + '\n')
    log(f'SAVED {name}: CV logloss {score:.5f}')
    return score
