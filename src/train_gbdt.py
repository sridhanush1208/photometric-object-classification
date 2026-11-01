"""Train LightGBM / XGBoost / CatBoost on the fixed folds; save OOF + test predictions.

  python src/train_gbdt.py --model lgb --features all --seeds 0,1,2 --name lgb_all
Tuned params from work/params/<model>.json (written by tune.py) are used when present."""
import argparse
import json

import numpy as np

from common import (N_FOLDS, N_JOBS, PARAMS, SMOKE, Progress, load_work, log, mlogloss, save_preds,
                    select, split_folds)

DEFAULTS = {
    'lgb': dict(objective='multiclass', num_class=3, learning_rate=0.03, num_leaves=63,
                min_child_samples=30, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
                lambda_l1=0.0, lambda_l2=1.0, max_bin=255, verbose=-1),
    'xgb': dict(objective='multi:softprob', num_class=3, eval_metric='mlogloss', learning_rate=0.03,
                max_depth=7, min_child_weight=2, subsample=0.8, colsample_bytree=0.6,
                reg_lambda=1.0, reg_alpha=0.0, tree_method='hist', max_bin=256),
    'cat': dict(loss_function='MultiClass', learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
                border_count=254, random_strength=1.0, bootstrap_type='Bernoulli', subsample=0.8),
}


def get_params(model, lr=None, tuned=True):
    p = dict(DEFAULTS[model])
    f = PARAMS / f'{model}.json'
    if tuned and f.exists():
        p.update(json.loads(f.read_text())['params'])
        log(f'using tuned params from {f}')
    if lr:
        p['learning_rate'] = lr
    return p


def fit_predict(model, p, Xa, ya, Xb, yb, Xt, seed, device, rounds, es=200):
    """Train on (Xa, ya) with early stopping on (Xb, yb); return (pred_b, pred_test, best_iter)."""
    if model == 'lgb':
        import lightgbm as lgb
        q = dict(p, seed=seed, num_threads=N_JOBS)
        m = lgb.train(q, lgb.Dataset(Xa, ya), rounds, valid_sets=[lgb.Dataset(Xb, yb)],
                      callbacks=[lgb.early_stopping(es, verbose=False)])
        it = m.best_iteration
        return m.predict(Xb, num_iteration=it), m.predict(Xt, num_iteration=it), it
    if model == 'xgb':
        import xgboost as xgb
        q = dict(p, seed=seed, nthread=N_JOBS)
        if device == 'gpu':
            q['device'] = 'cuda'
        da, db, dt = xgb.DMatrix(Xa, ya), xgb.DMatrix(Xb, yb), xgb.DMatrix(Xt)
        m = xgb.train(q, da, rounds, evals=[(db, 'valid')], early_stopping_rounds=es, verbose_eval=False)
        r = (0, m.best_iteration + 1)
        return m.predict(db, iteration_range=r), m.predict(dt, iteration_range=r), m.best_iteration + 1
    if model == 'cat':
        from catboost import CatBoostClassifier
        q = dict(p, iterations=rounds, random_seed=seed, od_type='Iter', od_wait=es, verbose=0,
                 use_best_model=True, allow_writing_files=False)
        if device == 'gpu':
            q.update(task_type='GPU', devices='0')
        else:
            q['thread_count'] = N_JOBS
        m = CatBoostClassifier(**q)
        m.fit(Xa, ya, eval_set=(Xb, yb))
        return m.predict_proba(Xb), m.predict_proba(Xt), m.get_best_iteration()
    raise ValueError(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['lgb', 'xgb', 'cat'])
    ap.add_argument('--features', default='all')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--rounds', type=int, default=30000)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'gpu'])
    ap.add_argument('--no-tuned', action='store_true')
    ap.add_argument('--nfolds', type=int, default=N_FOLDS, help='5, 10 or 20 (more folds = more training rows per model)')
    ap.add_argument('--nested', action='store_true',
                    help='add the clean per-fold photo-z features from src/nested.py (5 or 10 folds)')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='override one model param, e.g. --set extra_trees=true (repeatable)')
    ap.add_argument('--name', default=None)
    a = ap.parse_args()
    if SMOKE:
        a.rounds = 150

    tr, te, y, fold, groups = load_work()
    fold5 = fold.copy()
    fold = split_folds(fold, y, a.nfolds)
    nf = a.nfolds
    cols = select(groups, a.features)
    X, Xt = tr[cols].values, te[cols].values
    p = get_params(a.model, a.lr, not a.no_tuned)
    for kv in a.set:
        k, v = kv.split('=', 1)
        p[k] = json.loads(v) if v[:1] in '0123456789-[{' or v in ('true', 'false') else v
    name = a.name or f'{a.model}_{a.features.replace(",", "_")}'
    seeds = [int(s) for s in a.seeds.split(',')]
    log(f'{name}: {len(cols)} features, seeds {seeds}, params {p}')

    oof = np.zeros((len(X), 3))
    tst = np.zeros((len(Xt), 3))
    iters = []
    prog = Progress(name, len(seeds) * nf)
    for s in seeds:
        o = np.zeros((len(X), 3))
        for f in range(nf):
            a_, b_ = fold != f, fold == f
            Xa, Xb, Xtf = X[a_], X[b_], Xt
            if a.nested:
                # clean photo-z features of the 5-fold group g that holds this validation fold: built
                # without any of group g's labels, so they are clean for both of g's 10-fold halves
                from nested import load_fold
                g = f // (nf // N_FOLDS)          # 5-fold group holding validation fold f
                T, V, Lt, _ = load_fold(g)
                L = np.zeros((len(X), T.shape[1]), dtype=np.float32)
                L[fold5 != g], L[fold5 == g] = T, V
                Xa, Xb, Xtf = np.hstack([Xa, L[a_]]), np.hstack([Xb, L[b_]]), np.hstack([Xt, Lt])
            pb, pt, it = fit_predict(a.model, p, Xa, y[a_], Xb, y[b_], Xtf, s, a.device, a.rounds)
            o[b_] = pb
            tst += pt / (nf * len(seeds))
            iters.append(int(it))
            prog.step(f'seed {s} fold {f}: {mlogloss(y[b_], pb):.5f} (iter {it})')
        log(f'  seed {s}: CV {mlogloss(y, o):.5f}')
        oof += o / len(seeds)
    save_preds(name, oof, tst, y, dict(model=a.model, features=a.features, n_features=len(cols),
                                       nested=a.nested, seeds=a.seeds, nfolds=nf, iters=iters, params=p))


if __name__ == '__main__':
    main()
