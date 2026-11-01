"""Run the whole pipeline locally (same steps and settings as notebook/pipeline.ipynb).

    python run_pipeline.py            # needs data/train.csv and data/test.csv
    python run_pipeline.py --smoke    # quick end-to-end check on a 6,000-row sample

Writes submission.csv in the repository root. With a GPU (e.g. a Kaggle T4) the full run took 1h11m;
on CPU only it takes several hours (the nets and XGBoost/CatBoost are much faster on a GPU)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / ('work_smoke' if '--smoke' in sys.argv else 'work')
BEST_FEATURES = 'raw,color,flux,morph,errres,sky,local,dens,nmix,spec'
TUNED_PARAMS = {
    "lgb": {
        "model": "lgb",
        "features": "best",
        "tune_lr": 0.08,
        "score": 0.15519564375267847,
        "trial": 72,
        "iters": [
            140,
            132
        ],
        "params": {
            "num_leaves": 129,
            "min_child_samples": 6,
            "feature_fraction": 0.33566272615195697,
            "bagging_fraction": 0.9867093951206547,
            "lambda_l1": 3.0848736137035792e-06,
            "lambda_l2": 5.972915764461234,
            "min_gain_to_split": 0.3893637536883358,
            "path_smooth": 26.155304601473887
        }
    },
    "xgb": {
        "model": "xgb",
        "features": "best",
        "tune_lr": 0.08,
        "score": 0.1561363786458969,
        "trial": 27,
        "iters": [
            225,
            247
        ],
        "params": {
            "max_depth": 11,
            "min_child_weight": 10.569026150555239,
            "subsample": 0.8061110747783549,
            "colsample_bytree": 0.7402104823925528,
            "reg_lambda": 26.132280211577147,
            "reg_alpha": 5.6892790299999745e-08,
            "gamma": 0.1736314975486576
        }
    }
}
STEPS = [
    "src/nested.py",
    "src/train_gbdt.py --model lgb --features best --nested --seeds 0 --lr 0.03 --nfolds 20 --name lgb_best",
    "src/train_gbdt.py --model xgb --features best --nested --seeds 0 --lr 0.03 --nfolds 10 --name xgb_best --device gpu",
    "src/train_gbdt.py --model cat --features best --nested --seeds 0 --lr 0.06 --nfolds 10 --name cat_best --device gpu",
    "src/train_mlp.py --features best,-dens,-nmix,-spec --seeds 0,1 --hidden 256 --layers 4 --aux 0 --nfolds 20 --name mlp_base",
    "src/train_mlp.py --features best,-dens,-nmix,-spec --seeds 0 --aux 0.2 --nfolds 10 --name mlp_aux"
]


def main():
    env = dict(os.environ, CS_ROOT=str(ROOT), CS_WORK=str(WORK), CS_SUBS=str(ROOT / 'subs'),
               CS_PREP_LEARNED='0', PYTHONWARNINGS='ignore', PYTHONUNBUFFERED='1')
    if '--smoke' in sys.argv:
        env['CS_SMOKE'] = '1'
    gpu = False
    try:
        import torch
        gpu = torch.cuda.is_available()
    except ImportError:
        pass
    (WORK / 'params').mkdir(parents=True, exist_ok=True)
    (WORK / 'best_features.txt').write_text(BEST_FEATURES)
    for k, v in TUNED_PARAMS.items():
        (WORK / 'params' / f'{k}.json').write_text(json.dumps(v))
    out = ROOT / ('submission_smoke.csv' if '--smoke' in sys.argv else 'submission.csv')
    steps = ['src/prep.py'] + STEPS + [f'src/ensemble.py --tag pipeline --out {out}']
    t0 = time.time()
    for i, cmd in enumerate(steps, 1):
        if not gpu:
            cmd = cmd.replace('--device gpu', '--device cpu')
        print(f'\n=== step {i}/{len(steps)}: python {cmd}', flush=True)
        r = subprocess.run([sys.executable, '-u'] + cmd.split(), cwd=ROOT, env=env)
        if r.returncode != 0:
            sys.exit(f'step failed: {cmd}')
    print(f'\ndone in {(time.time() - t0) / 60:.1f} min -> {out.name}')


if __name__ == '__main__':
    main()
