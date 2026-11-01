# Photometric object classification: STAR / GALAXY / QSO

Classifies Sloan Digital Sky Survey objects as stars, galaxies or quasars from photometry alone
(brightness in the `u g r i z` filters, their errors, object size and sky position), scored with
multi-class log loss.

## Results

| | Log loss (lower is better) |
|---|---|
| Cross-validation (nested, 181,221 training rows) | **0.14624** |
| Public leaderboard | **0.13356** |

## Approach

- **Label-free features:** colours, spectral shape, size and error residuals relative to objects of
  the same brightness, comparison with the 300 nearest objects on the sky, colour-space density, and
  a noise-aware Gaussian mixture of the colours.
- **Clean photo-z features:** redshift (available only in training) is predicted from photometry and
  used as a feature, rebuilt per fold from that fold's training labels only, so no labels leak.
  Redshift itself is never a model input.
- **Models:** LightGBM, XGBoost and CatBoost plus two PyTorch MLPs, trained with 10–20 folds.
- **Blend:** weights fitted on out-of-fold predictions and chosen by nested cross-validation.

## Run

Open `pipeline.ipynb` on Kaggle with the competition data attached and a GPU enabled, then run all
cells. It writes `submission.csv`.
