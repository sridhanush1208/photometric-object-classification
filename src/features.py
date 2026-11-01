"""Label-free features. Statistics (error-vs-magnitude curves, sky density) use train+test together,
which is allowed: it is competition data and no labels are involved."""
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree, NearestNeighbors

from common import BANDS, N_JOBS

LOCAL_K = 300

RAW = BANDS + [f'err_{b}' for b in BANDS] + ['size_r', 'gal_l', 'gal_b']
PAIRS = [(a, b) for i, a in enumerate(BANDS) for b in BANDS[i + 1:]]          # all 10 colours
ADJ = list(zip(BANDS, BANDS[1:]))


def _binned_curve(x, v, nbins=60):
    """Median of v as a smooth function of x (quantile bins + linear interpolation)."""
    q = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    idx = np.clip(np.searchsorted(q, x, side='right') - 1, 0, len(q) - 2)
    cx = np.array([np.median(x[idx == k]) for k in range(len(q) - 1)])
    cv = np.array([np.median(v[idx == k]) for k in range(len(q) - 1)])
    return np.interp(x, cx, cv)


def _binned_quantile(x, v, qs, nbins=60):
    q = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    idx = np.clip(np.searchsorted(q, x, side='right') - 1, 0, len(q) - 2)
    cx = np.array([np.median(x[idx == k]) for k in range(len(q) - 1)])
    out = []
    for qq in qs:
        cv = np.array([np.quantile(v[idx == k], qq) for k in range(len(q) - 1)])
        out.append(np.interp(x, cx, cv))
    return out


def sky_vectors(df):
    l, b = np.radians(df['gal_l'].values), np.radians(df['gal_b'].values)
    return np.c_[np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)]


def chord(deg):
    return 2 * np.sin(np.radians(deg) / 2)


def build(tr, te):
    """Returns (train_features, test_features, groups) where groups maps group name -> columns."""
    df = pd.concat([tr[RAW], te[RAW]], ignore_index=True)
    n_tr = len(tr)
    F, groups = {}, {}

    def add(group, name, values):
        F[name] = np.asarray(values, dtype=np.float32)
        groups.setdefault(group, []).append(name)

    for c in RAW:
        add('raw', c, df[c])

    # colours, colour errors and significance
    for a, b in PAIRS:
        add('color', f'{a}-{b}', df[a] - df[b])
    for a, b in ADJ:
        e = np.hypot(df[f'err_{a}'], df[f'err_{b}'])
        add('color', f'e_{a}-{b}', e)
        add('color', f'sig_{a}-{b}', (df[a] - df[b]) / e)
    add('color', 'mean_mag', df[BANDS].mean(1))
    add('color', 'ug_x_gr', (df.u - df.g) * (df.g - df.r))
    add('color', 'gr_x_ri', (df.g - df.r) * (df.r - df.i))

    # fluxes: shape of the spectrum independent of brightness, and per-band SNR
    flux = np.stack([10 ** (-0.4 * (df[b].values - 22.5)) for b in BANDS], 1)
    tot = flux.sum(1, keepdims=True)
    for k, b in enumerate(BANDS):
        add('flux', f'sed_{b}', flux[:, k] / tot[:, 0])
        add('flux', f'snr_{b}', 1.0857 / df[f'err_{b}'])
    add('flux', 'log_tot_flux', np.log10(tot[:, 0]))
    add('flux', 'u_drop', (df.u - df.g) - (df.g - df.r))          # high-redshift quasar u dropout

    # morphology: is the object bigger than a point source of the same brightness?
    ls = np.log(df['size_r'].values)
    add('morph', 'log_size', ls)
    add('morph', 'sb_r', df['r'].values + 5 * np.log10(df['size_r'].values))
    p10, p25, p50 = _binned_quantile(df['r'].values, ls, [0.10, 0.25, 0.50])
    add('morph', 'size_res', (ls - p25) / np.maximum(p50 - p10, 1e-3))
    add('morph', 'size_res_raw', ls - p25)

    # error-vs-brightness residuals: extended objects have larger errors at fixed magnitude
    for b in BANDS:
        le = np.log(df[f'err_{b}'].values)
        add('errres', f'eres_{b}', le - _binned_curve(df[b].values, le))
    E = np.stack([F[f'eres_{b}'] for b in BANDS], 1)
    add('errres', 'eres_mean', E.mean(1))
    add('errres', 'eres_gri', E[:, 1:4].mean(1))

    # sky position and local object density (Milky Way plane = many stars)
    V = sky_vectors(df)
    add('sky', 'abs_b', df['gal_b'].abs())
    add('sky', 'sin_b', V[:, 2])
    add('sky', 'vx', V[:, 0])
    add('sky', 'vy', V[:, 1])
    tree = KDTree(V)
    for deg in (0.25, 1.0):
        add('sky', f'dens_{deg}', np.log1p(tree.query_radius(V, chord(deg), count_only=True)))

    # 'local': compare each object with its LOCAL_K nearest objects on the sky (train+test, no labels).
    # Removes field-to-field changes in seeing (sizes), sky depth (errors) and Milky Way dust (colours).
    # Experiment 31 Oct: LightGBM CV 0.15713 -> 0.15599 (K=300).
    _, idx = NearestNeighbors(n_neighbors=LOCAL_K + 1, n_jobs=N_JOBS).fit(V).kneighbors(V)
    idx = idx[:, 1:]
    for b in BANDS:
        e = F[f'eres_{b}']
        add('local', f'loc_eres_{b}', e - np.median(e[idx], 1))
    s = F['size_res_raw']
    sn = s[idx]
    add('local', 'loc_size_q20', s - np.quantile(sn, 0.2, axis=1))
    add('local', 'loc_size_q50', s - np.median(sn, 1))
    point_like = (s < np.quantile(s, 0.35))[idx]
    for a, b in ADJ:
        c = F[f'{a}-{b}']
        med = np.nanmedian(np.where(point_like, c[idx], np.nan), 1)
        med = np.where(np.isnan(med), np.nanmedian(c), med)
        add('local', f'nb_col_{a}-{b}', med)
        add('local', f'loc_col_{a}-{b}', c - med)

    # 'dens': how crowded each object's neighbourhood is in colour space (train+test, no labels).
    # Stars crowd onto the stellar locus, quasars sit in sparse regions.
    # Experiment 31 Oct: LightGBM -0.0004 on average over 3 seeds.
    spaces = {'col4': ['u-g', 'g-r', 'r-i', 'i-z'], 'col4_shape': ['u-g', 'g-r', 'r-i', 'i-z', 'size_res'],
              'col4_mag': ['u-g', 'g-r', 'r-i', 'i-z', 'r']}
    for name, cols in spaces.items():
        Z = np.stack([F[c] for c in cols], 1).astype(np.float64)
        Z = (Z - Z.mean(0)) / Z.std(0)
        d, _ = NearestNeighbors(n_neighbors=51, n_jobs=N_JOBS).fit(Z).kneighbors(Z)
        for k in (10, 50):
            add('dens', f'dens_{name}_k{k}', np.log(d[:, k] + 1e-6))

    # 'nmix': noise-aware colour mixture (from the 31 Oct review; label-free, train+test).
    # An 8-component Gaussian mixture is fitted on the 4 adjacent colours of low-noise objects; each
    # object is then scored against every component WIDENED by its own colour-noise covariance
    # (adjacent colours share a band, hence the off-diagonal -err^2 terms). Features: the 8
    # responsibilities + total log-likelihood. Our 5-fold CV (tuned LightGBM, 3 seeds): -0.0014.
    from scipy.special import logsumexp, softmax
    from sklearn.mixture import GaussianMixture
    C = np.stack([F[f'{a}-{b}'] for a, b in ADJ], 1).astype(np.float64)
    e2 = np.stack([df[f'err_{b}'].values.astype(np.float64) ** 2 for b in BANDS], 1)
    cov = np.zeros((len(C), 4, 4))
    for i in range(4):
        cov[:, i, i] = e2[:, i] + e2[:, i + 1]
    for i in range(3):
        cov[:, i, i + 1] = cov[:, i + 1, i] = -e2[:, i + 1]
    clean = np.sqrt(np.stack([cov[:, i, i] for i in range(4)], 1)).max(1) < 0.12
    gm = GaussianMixture(8, covariance_type='full', reg_covar=1e-4, random_state=0, max_iter=200).fit(C[clean])
    R, LL = np.zeros((len(C), 8)), np.zeros(len(C))
    for a0 in range(0, len(C), 40000):
        sl = slice(a0, a0 + 40000)
        L = []
        for w, mu, S in zip(gm.weights_, gm.means_, gm.covariances_):
            Lc = np.linalg.cholesky(S[None] + cov[sl])
            z = np.linalg.solve(Lc, (C[sl] - mu)[..., None])[..., 0]
            L.append(np.log(w) - 0.5 * (z ** 2).sum(1) - np.log(np.diagonal(Lc, axis1=1, axis2=2)).sum(1))
        L = np.stack(L, 1)
        R[sl], LL[sl] = softmax(L, 1), logsumexp(L, 1)
    for k in range(8):
        add('nmix', f'nmix_r{k}', R[:, k])
    add('nmix', 'nmix_ll', LL)

    # 'spec' (V8, from the 1 Nov audit; untested gamble, trees only): spectral curvature, the SDSS
    # perpendicular colour, Lyman-break dropout, cross-band error ratios, error-residual spread and
    # a local seeing floor (15th percentile of log size among the 50 nearest sky neighbours).
    ug, gr, ri, iz = (F[f'{a}-{b}'] for a, b in ADJ)
    add('spec', 'curv_ugr', ug - gr)
    add('spec', 'curv_gri', gr - ri)
    add('spec', 'curv_riz', ri - iz)
    add('spec', 'd_perp', ri - gr / 4 - 0.18)
    add('spec', 'lyman_drop', ug - 2 * gr)
    add('spec', 'snr_ug_ratio', np.log(df['err_u'].values / df['err_g'].values))
    add('spec', 'snr_iz_ratio', np.log(df['err_z'].values / df['err_i'].values))
    add('spec', 'eres_std', E.std(1))
    add('spec', 'eres_z_minus_g', E[:, 4] - E[:, 1])
    add('spec', 'loc_raw_psf_diff', ls - np.quantile(ls[idx[:, :50]], 0.15, axis=1))

    X = pd.DataFrame(F)
    return X.iloc[:n_tr].reset_index(drop=True), X.iloc[n_tr:].reset_index(drop=True), groups
