# Photometric object classification: STAR / GALAXY / QSO

Classifies Sloan Digital Sky Survey (SDSS) objects as **stars, galaxies or quasars** from photometry
alone: brightness in five filters (`u g r i z`), their measurement errors, an object size and the
position on the sky. Built for a 4-day Kaggle classification hackathon (2026), scored with
multi-class log loss.

| | Score (multi-class log loss, lower is better) |
|---|---|
| Cross-validation of the final blend (nested, 181,221 training rows) | **0.14624** |
| Public leaderboard (30% of the 31,000 test rows) | **0.13356** |
| Runtime, end to end | **1h11m** on a free Kaggle notebook (GPU T4) |

## How it works

```
train.csv + test.csv
   │
   ├─ 1. Label-free features (train + test together, no labels)                 src/features.py
   │     colours · flux shape · morphology · error residuals · sky position
   │     · sky-neighbour comparison · colour-space density · noise-aware colour mixture · spectral shape
   │
   ├─ 2. Clean photo-z features, rebuilt per fold from that fold's training labels only   src/nested.py
   │
   ├─ 3. Five models, 10- or 20-fold, every prediction out-of-fold     src/train_gbdt.py, src/train_mlp.py
   │     LightGBM · XGBoost · CatBoost (trees, with photo-z) · two PyTorch MLPs
   │
   └─ 4. Blend fitted on the out-of-fold predictions, chosen by nested CV, calibrated   src/ensemble.py
         → submission.csv
```

### Features (103 for the tree models, 78 for the nets)

| Group | # | What it captures |
|---|---|---|
| raw | 13 | the given columns |
| color | 21 | all colour differences, colour errors and significance (colour separates object types far better than brightness) |
| flux | 12 | spectral shape independent of brightness, signal-to-noise per band, a u-band dropout index |
| morph | 4 | size relative to point sources of the same brightness (stars and quasars are point sources) |
| errres | 7 | measurement error relative to the typical error at that brightness (extended objects have larger errors) |
| sky | 6 | Galactic latitude, sky vector, local object density (near the Milky Way plane most objects are stars) |
| local | 15 | each object compared with its 300 nearest objects **on the sky**: removes seeing, sky-depth and dust variation |
| dens | 6 | how crowded the object's neighbourhood is in colour space (stars crowd onto the stellar locus) — trees only |
| nmix | 9 | **noise-aware colour mixture**: a Gaussian mixture of the colour distribution, with every component widened by the object's own colour-noise covariance — trees only |
| spec | 10 | spectral curvature, the SDSS perpendicular colour, Lyman-break dropout, cross-band error ratios, a local seeing floor — trees only |

All of these use no labels; statistics are computed on train and test together.

### Clean (nested) photo-z features

A LightGBM regression predicts `log(1 + redshift)` and a 7-class LightGBM predicts the redshift
range (star ≈ 0 … quasar > 3). These 9 predictions feed the tree models. `redshift` exists only in
the training data and is **never a model input**; it is used only as a *training target*, which the
organisers confirmed is allowed.

How they are built matters. Built the usual out-of-fold way, a training row's feature is computed
by a model that saw the *validation* rows' labels, and on a true holdout that contamination made
them useless (0.1583 vs 0.1582 without them). Rebuilt per fold group using only that group's
training labels (`src/nested.py`), the same features score 0.1556. On full CV: LightGBM 0.15339 →
0.15207.

### Models

| Model | Inputs | Folds × seeds | CV | Blend weight |
|---|---|---|---|---|
| LightGBM (tuned with Optuna) | 103 + photo-z | 20 × 1 | 0.14942 | 0.36 |
| XGBoost (GPU, tuned) | 103 + photo-z | 10 × 1 | 0.15003 | 0.09 |
| CatBoost (GPU) | 103 + photo-z | 10 × 1 | 0.15149 | 0.06 |
| MLP 256×4 | 78 | 20 × 2 | 0.14994 | 0.40 |
| MLP 512×3 + auxiliary redshift head | 78 | 10 × 1 | 0.15457 | 0.05 |

Training with 10 or 20 folds (90–95% of the rows per model, instead of 80% with 5 folds) improved
every model by about 0.001; every fold split nests inside one fixed 5-fold split, so all predictions
stay out-of-fold and the photo-z features stay clean. The MLPs use rank-normalised inputs,
Linear → BatchNorm → SiLU → Dropout blocks, AdamW and a one-cycle schedule.

### Blending

Several blenders are fitted on the out-of-fold predictions and each is scored with nested CV
(fitted on 4 fold groups, scored on the 5th): a weighted average with temperature, a weighted sum
of log-probabilities with per-class bias (plain and with non-negative bounded weights), logistic
stacking, and a small LightGBM correction model. The best is used (here the bounded log-linear
blend), then probabilities are floored at 1e-4.

### What did not help

Features built from neighbours' **labels**, pseudo-labelling the test set, periodic number
embeddings, weight averaging (EMA) and input noise for the nets, wider nets, extra models of the
same kind (the blend was saturated on variety), and giving the nets the photo-z / density /
mixture features (all made the nets worse).

## Running it

**On Kaggle (as submitted):** create a notebook with the competition data attached, import
`notebook/pipeline.ipynb`, set the accelerator to **GPU T4 x2**, then *Run All* (or *Save & Run
All*). It writes the source files, runs every step with a live progress table, and writes
`submission.csv`.

**Locally:**

```
pip install -r requirements.txt
# put train.csv and test.csv in data/
python run_pipeline.py --smoke      # ~5-10 min end-to-end check on a 6,000-row sample
python run_pipeline.py              # full run -> submission.csv (fast with a GPU, hours on CPU only)
```

## Repository layout

| Path | What it is |
|---|---|
| `notebook/pipeline.ipynb` | The exact notebook that produced the scores above (self-contained) |
| `run_pipeline.py` | Runs the same steps and settings locally |
| `src/common.py` | Paths, the fixed folds and the 10/20-fold split, log loss, progress printing |
| `src/features.py` | All label-free features |
| `src/prep.py` | Folds + feature build |
| `src/nested.py` | Clean per-fold photo-z features |
| `src/train_gbdt.py` | LightGBM / XGBoost / CatBoost training |
| `src/train_mlp.py` | MLP training |
| `src/ensemble.py` | Blenders, nested CV, calibration, submission file |

The data is not included; see `data/README.md`. GPU XGBoost, CatBoost and the nets are not
bit-for-bit deterministic, so re-runs differ very slightly.
