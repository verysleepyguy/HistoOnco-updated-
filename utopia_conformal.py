"""UTOPIA-inspired conformal calibration for HistoOnco pathway-activity maps.

Turns each per-pathway prediction map from impute_integrated.py (single-cell
mode) into a calibrated, FDR-controlled "is this pathway active here" map.

Pipeline
--------
1. 5-fold spatial cross-validation (fold logic ported from the pilot
   notebook's Part E: k-means regions dealt into folds, with a shrinking
   buffer) to get out-of-fold (OOF) predictions for every cell.
2. 8 tissue-appearance clusters (ported from the pilot notebook's F7c:
   PCA-20 + MiniBatchKMeans, computed once at pixel level).
3. Block aggregation at scales {1, 2, 4, 8} feature pixels (16/32/64/128 um).
4. A UTOPIA-style conformal p-value per (pathway, scale, cluster), BH-adjusted
   within each cluster, converted to a confidence score and FDR calls.

See DIFFERENCES.md for every place this intentionally departs from UTOPIA's
own code (as opposed to the UTOPIA paper).

This module is new code; the only thing it imports from the existing
pipeline is get_data_cells / get_pathway_labels / CellDataset /
SpatialSmoothModel / SPATIAL_* (from impute_integrated), train_model (from
train) and plot_matrix (from visual) - per the "import, don't copy" rule.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from impute_integrated import (
        CellDataset, SpatialSmoothModel, get_data_cells, get_pathway_labels,
        _resolve_label_file, LABEL_BORDER_MARGIN, SPATIAL_KNN_K,
        SPATIAL_LOSS_WEIGHT)
from train import train_model
from utils import load_image, load_pickle, save_pickle
from visual import plot_matrix

# =============================== DEFAULTS ===================================
# Spatial cross-validation (Part E of the pilot notebook; production values).
K_FOLDS = 5
N_REGIONS_TRY = [60, 30, 15, 10, 5]
MIN_TRAIN_FRAC = 0.6
BUFFER = 16
CV_EPOCHS = 10
SEED = 0

# Appearance clusters (F7c of the pilot notebook).
K_APP = 8
N_PCA = 20
APPEARANCE_FIT_SAMPLE = 100_000

# Block aggregation.
SCALES = (1, 2, 4, 8)
SCALE_MIN_CELLS = {1: 2, 2: 5, 4: 5, 8: 5}

# Conformal.
THRESHOLD_QUANTILE = 0.75
THRESHOLD_QUANTILE_SWEEP = (0.5, 0.75, 0.9)
TOLERANCE_MODES = ('none', 'weak', 'medium', 'strong')
FDR_LEVELS = (0.10, 0.20)
P_MIN, P_MAX = 0.01, 0.5
# ==============================================================================


# ============================ fold generation ================================
# Direct port of ucell/HistoOnco_EndToEnd Part E's `split` / `kept_frac`
# (k-means regions, dealt into folds by greedy load-balancing, then a
# shrinking buffer around the held-out fold's cells).

def _split_into_regions(locs_f, n_regions, k_folds, seed):
    region = MiniBatchKMeans(
            n_regions, batch_size=10000, n_init=3,
            random_state=seed).fit_predict(locs_f)
    sizes = np.bincount(region, minlength=n_regions)
    fold_of_region = np.zeros(n_regions, int)
    load = np.zeros(k_folds)
    for r in np.argsort(-sizes, kind='stable'):
        f = int(np.argmin(load))
        fold_of_region[r] = f
        load[f] += sizes[r]
    fold = fold_of_region[region]
    dmin = np.zeros((k_folds, len(locs_f)), np.float32)
    for f in range(k_folds):
        dmin[f] = cKDTree(locs_f[fold == f]).query(locs_f, k=1, workers=-1)[0]
    return fold, dmin


def _kept_frac(fold, dmin, buf, k_folds):
    fracs = []
    for f in range(k_folds):
        other = fold != f
        denom = other.sum()
        if denom == 0:
            continue
        fracs.append(((fold != f) & (dmin[f] > buf)).sum() / denom)
    return float(np.mean(fracs)) if fracs else 0.0


def generate_spatial_folds(
        locs_f, k_folds=K_FOLDS, n_regions_try=N_REGIONS_TRY,
        min_train_frac=MIN_TRAIN_FRAC, buffer=BUFFER, seed=SEED):
    """Port of the notebook's Part E fold search.

    Returns (fold, n_regions_used, buffer_used) where `fold` is an (N,) int
    array in [0, k_folds).
    """
    buffers_try = list(dict.fromkeys([buffer, buffer // 2, buffer // 4, 0]))
    cache = {}
    chosen = None
    for buf in buffers_try:
        for nr in n_regions_try:
            if nr not in cache:
                cache[nr] = _split_into_regions(locs_f, nr, k_folds, seed)
            kf = _kept_frac(*cache[nr], buf, k_folds)
            print(f'  {nr:3d} regions, {buf:2d}px buffer -> '
                  f'{kf:.0%} of the other folds kept for training')
            if kf >= min_train_frac:
                chosen = (nr, buf)
                break
        if chosen:
            break
    if chosen is None:
        # fall back to the smallest region count and zero buffer rather than
        # raising - callers (e.g. tiny synthetic data) still get a usable
        # (if imperfect) fold assignment.
        nr = n_regions_try[-1]
        if nr not in cache:
            cache[nr] = _split_into_regions(locs_f, nr, k_folds, seed)
        chosen = (nr, 0)
        print('  WARNING: no (regions, buffer) combination reached '
              f'min_train_frac={min_train_frac}; falling back to {chosen}')
    n_regions, buffer_used = chosen
    fold = cache[n_regions][0]
    print(f'using {n_regions} regions and a {buffer_used}px buffer')
    return fold, n_regions, buffer_used


def fold_train_indices(fold, locs_f, f, buffer_used):
    """Candidate (not-fold-f) cells farther than buffer_used from every
    fold-f cell."""
    te = np.flatnonzero(fold == f)
    cand = np.flatnonzero(fold != f)
    if len(te) == 0 or len(cand) == 0:
        return cand
    d = cKDTree(locs_f[te]).query(locs_f[cand], k=1, workers=-1)[0]
    return cand[d > buffer_used]


# ======================== out-of-fold cross-validation =======================

def _recompute_spatial_neighbors(locs_f_subset, k):
    k_eff = min(k, len(locs_f_subset) - 1)
    if k_eff < 1:
        return np.zeros((len(locs_f_subset), 1), dtype=np.int64)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(locs_f_subset)
    return nn.kneighbors(locs_f_subset, return_distance=False)[:, 1:]


def _predict_cells(model, embs, locs_subset, device):
    n_out = model.hparams.n_out if hasattr(model, 'hparams') else None
    x = embs[locs_subset[:, 0], locs_subset[:, 1]].astype(np.float32)
    finite = np.isfinite(x).all(-1)
    out = np.full((len(locs_subset),) + (() if n_out is None else (n_out,)),
                  np.nan, dtype=np.float32)
    if finite.any():
        model.eval()
        with torch.no_grad():
            xt = torch.tensor(x[finite], device=model.device)
            pred = model(xt).cpu().numpy()
        if out.ndim == 1:
            out[finite] = pred[:, 0]
        else:
            out[finite] = pred
    return out


def compute_oof_predictions(
        embs, Y, locs, fold, buffer_used, cv_epochs=CV_EPOCHS, cal_states=1,
        device='cuda', lr=1e-4, spatial_weight=SPATIAL_LOSS_WEIGHT,
        spatial_k=SPATIAL_KNN_K, seed=SEED, k_folds=K_FOLDS):
    """Train one SpatialSmoothModel per fold on the other folds (spatial
    neighbours recomputed inside the training subset) and predict the
    held-out cells. Nothing is written to disk by train_model (no model.pt).
    """
    N, n_out = Y.shape
    locs_f = locs.astype(float)
    oof = np.full((N, n_out), np.nan, dtype=np.float32)

    for f in range(k_folds):
        te = np.flatnonzero(fold == f)
        tr = fold_train_indices(fold, locs_f, f, buffer_used)
        print(f'fold {f + 1}/{k_folds}: {len(te)} held-out cells, '
              f'{len(tr)} training cells')
        if len(tr) < 2 or len(te) == 0:
            print(f'  skipping fold {f}: not enough data')
            continue

        nbr_idx_tr = _recompute_spatial_neighbors(locs_f[tr], spatial_k)
        bs = max(1, min(128, len(tr) // 16))

        states_pred = []
        for s in range(cal_states):
            pl.seed_everything(seed + f * 1000 + s, workers=True)
            dataset = CellDataset(
                    x_all=embs, y=Y[tr], locs=locs[tr], nbr_idx=nbr_idx_tr)
            model_kwargs = dict(
                    n_inp=embs.shape[-1], n_out=n_out, lr=lr,
                    spatial_weight=spatial_weight, spatial_k=spatial_k)
            model, _hist, _trainer = train_model(
                    dataset=dataset, batch_size=bs, epochs=cv_epochs,
                    model=None, model_class=SpatialSmoothModel,
                    model_kwargs=model_kwargs, device=device)
            pred = _predict_cells(model, embs, locs[te], device)
            states_pred.append(pred)
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()
        oof[te] = np.nanmedian(np.stack(states_pred, 0), axis=0)

    return oof


# ============================ appearance clusters =============================
# Port of the pilot notebook's F7c: PCA(20) on standardized histology
# features, MiniBatchKMeans(K_APP), fit on a subsample, computed once at
# pixel level.

def compute_appearance_clusters(
        embs, k_app=K_APP, n_pca=N_PCA, fit_sample=APPEARANCE_FIT_SAMPLE,
        seed=SEED):
    H, W, C = embs.shape
    valid = np.isfinite(embs).all(-1)
    X = embs[valid]
    rng = np.random.default_rng(seed)

    n_fit = min(fit_sample, len(X))
    fit_idx = rng.choice(len(X), n_fit, replace=False)
    mu = X[fit_idx].mean(0)
    sd = X[fit_idx].std(0) + 1e-6
    n_components = min(n_pca, n_fit, C)
    pca = PCA(n_components=n_components, svd_solver='randomized',
              random_state=seed).fit((X[fit_idx] - mu) / sd)
    Z = pca.transform((X - mu) / sd)

    n_km = min(fit_sample, len(Z))
    km_idx = rng.choice(len(Z), n_km, replace=False)
    km = MiniBatchKMeans(
            k_app, batch_size=min(20000, n_km), n_init=3,
            random_state=seed).fit(Z[km_idx])
    labels = km.predict(Z)

    clusters_image = np.full((H, W), -1, dtype=np.int16)
    clusters_image[valid] = labels
    return clusters_image


def block_majority_cluster(clusters_image, scale, k_app=K_APP):
    """Majority appearance cluster per (scale x scale) pixel block.
    Returns an (h_blocks, w_blocks) int16 array, -1 where the block has no
    valid-cluster pixel."""
    H, W = clusters_image.shape
    h_blocks = -(-H // scale)
    w_blocks = -(-W // scale)
    Hc, Wc = h_blocks * scale, w_blocks * scale
    padded = np.full((Hc, Wc), -1, dtype=np.int16)
    padded[:H, :W] = clusters_image
    blocks = padded.reshape(h_blocks, scale, w_blocks, scale)
    counts = np.stack(
            [(blocks == k).sum(axis=(1, 3)) for k in range(k_app)], axis=-1)
    total = counts.sum(-1)
    majority = counts.argmax(-1).astype(np.int16)
    majority[total == 0] = -1
    return majority


# ============================== block aggregation =============================

def block_shape(H, W, scale):
    return -(-H // scale), -(-W // scale)


def cell_block_stats(locs, values, scale, h_blocks, w_blocks):
    """Mean of a per-cell array within each (scale x scale) block.

    values: (N,) or (N, n_out). Returns (mean_grid, count_grid) with shape
    (h_blocks, w_blocks) or (h_blocks, w_blocks, n_out), NaN where a block has
    no cells.
    """
    br = locs[:, 0] // scale
    bc = locs[:, 1] // scale
    bid = br.astype(np.int64) * w_blocks + bc.astype(np.int64)
    n_blocks = h_blocks * w_blocks

    cnt = np.bincount(bid, minlength=n_blocks).astype(np.float64)
    count_grid = cnt.reshape(h_blocks, w_blocks)

    values = np.asarray(values)
    if values.ndim == 1:
        s = np.bincount(bid, weights=values, minlength=n_blocks)
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = s / cnt
        mean_grid = mean.reshape(h_blocks, w_blocks).astype(np.float32)
    else:
        n_out = values.shape[1]
        mean_grid = np.full((n_blocks, n_out), np.nan, dtype=np.float32)
        for j in range(n_out):
            s = np.bincount(bid, weights=values[:, j], minlength=n_blocks)
            with np.errstate(invalid='ignore', divide='ignore'):
                mean_grid[:, j] = s / cnt
        mean_grid = mean_grid.reshape(h_blocks, w_blocks, n_out)
    mean_grid[count_grid == 0] = np.nan
    return mean_grid, count_grid


def dense_block_mean(map2d, scale, h_blocks, w_blocks):
    """Block mean of a dense (H, W) prediction map, ignoring NaN. Used for
    the production test side (design B: test blocks use the block mean of
    the full prediction map, not the sparse per-cell OOF data - see
    DIFFERENCES.md)."""
    H, W = map2d.shape
    Hc, Wc = h_blocks * scale, w_blocks * scale
    padded = np.full((Hc, Wc), np.nan, dtype=np.float32)
    padded[:H, :W] = map2d
    blocks = padded.reshape(h_blocks, scale, w_blocks, scale)
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(blocks, axis=(1, 3))
    return mean


# ================================ conformal core ================================

def nonconformity_score_clip(y, yhat, c):
    """UTOPIA's one-sided nonconformity score, reimplemented: finite (= -yhat)
    for points consistent with the null (y <= c), +inf otherwise (never
    counted as "as extreme or more" than a finite test score)."""
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    scores = np.full(len(y), np.inf, dtype=np.float64)
    is_null = y <= c
    scores[is_null] = -yhat[is_null]
    return scores


def compute_pvals(test_scores, calib_scores, tolerance=0.0):
    """UTOPIA's conformal p-value, reimplemented: (1 + #calib points with
    score < test score) / (1 + n_calib)."""
    sorted_calib = np.sort(calib_scores + tolerance)
    n = len(calib_scores)
    num_smaller = np.searchsorted(sorted_calib, test_scores, side='left')
    return (num_smaller + 1) / (n + 1)


def bh_adjust(pvals):
    """Benjamini-Hochberg adjustment, reimplemented."""
    pvals = np.asarray(pvals, dtype=np.float64)
    m = len(pvals)
    if m == 0:
        return np.array([])
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    adjusted = m * sorted_p / np.arange(1, m + 1)
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def compute_tolerance(calib_yhat, calib_y, test_yhat, null_number, magnitude):
    """UTOPIA's batch-correction tolerance step, reimplemented. Off by
    default (--tolerance none) - see DIFFERENCES.md."""
    quantile = float(np.mean(calib_y <= null_number))
    if quantile <= 0.1:
        return 0.0
    calib_sorted = np.sort(calib_yhat)
    test_sorted = np.sort(test_yhat)
    calib_q = np.quantile(calib_sorted, quantile)
    test_q = np.quantile(test_sorted, quantile)
    calib_first_q = calib_sorted[calib_sorted <= calib_q]
    test_first_q = test_sorted[test_sorted <= test_q]
    quantile_grid = {
            'weak': [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
            'medium': [0.001, 0.005, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.995, 0.999],
            'strong': [0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                       0.8, 0.9, 0.99, 0.995, 0.999],
    }[magnitude]
    if len(calib_first_q) == 0 or len(test_first_q) == 0:
        return 0.0
    calib_quant = np.quantile(calib_first_q, quantile_grid)
    test_quant = np.quantile(test_first_q, quantile_grid)
    shift = np.clip(test_quant - calib_quant, 0, None)
    return float(-np.max(shift))


def pvals_to_confidence(pvals, p_min=P_MIN, p_max=P_MAX):
    pvals = np.clip(np.asarray(pvals, dtype=np.float64), 1e-300, 1.0)
    scores = ((-np.log10(pvals) + np.log10(p_max))
              / (-np.log10(p_min) + np.log10(p_max)))
    return np.clip(scores, 0, 1)


def default_min_calib_pts(n_test_in_cluster):
    """UTOPIA's own default when min_calib_pts is not given explicitly."""
    return max(200, n_test_in_cluster * 0.01)


def conformal_pathway_scale(
        calib_y, calib_yhat, calib_cluster, test_yhat, test_cluster,
        threshold_quantile=THRESHOLD_QUANTILE, min_calib_pts=None,
        tolerance_mode='none', k_app=K_APP):
    """Core UTOPIA-style conformal step for one (pathway, scale).

    Returns (threshold, p_adj, diagnostics) where p_adj has the same length
    as test_yhat (BH-adjusted within each cluster; p=1 for test points whose
    cluster was skipped for lack of calibration data).
    """
    t = float(np.quantile(calib_y, threshold_quantile))
    p_adj = np.ones(len(test_yhat), dtype=np.float64)
    diagnostics = []

    for k in range(k_app):
        calib_in_k = calib_cluster == k
        test_in_k = test_cluster == k
        n_calib_k = int(calib_in_k.sum())
        n_test_k = int(test_in_k.sum())

        min_pts = (min_calib_pts if min_calib_pts is not None
                   else default_min_calib_pts(n_test_k))
        skipped = (n_calib_k < min_pts) or (n_test_k == 0)
        diagnostics.append(dict(
                cluster=k, n_calib=n_calib_k, n_test=n_test_k,
                min_calib_pts=min_pts, skipped=bool(skipped)))
        if skipped:
            continue

        calib_y_k = calib_y[calib_in_k]
        calib_yhat_k = calib_yhat[calib_in_k]
        test_yhat_k = test_yhat[test_in_k]

        if tolerance_mode == 'none':
            tolerance = 0.0
        else:
            tolerance = compute_tolerance(
                    calib_yhat_k, calib_y_k, test_yhat_k, t, tolerance_mode)

        calib_scores = nonconformity_score_clip(calib_y_k, calib_yhat_k, t)
        test_scores = nonconformity_score_clip(
                np.full(len(test_yhat_k), t), test_yhat_k, t)
        pvals = compute_pvals(test_scores, calib_scores, tolerance)
        p_adj[test_in_k] = bh_adjust(pvals)

    return t, p_adj, diagnostics


def get_fdp_power(p_adj, label, threshold, fdr_level):
    """Realized false-discovery proportion and power at a nominal FDR level.
    `label` is the true block value, `threshold` the null cutoff."""
    discovered = p_adj <= fdr_level
    true_active = label > threshold
    n_disc = int(discovered.sum())
    fdp = 0.0 if n_disc == 0 else 1.0 - float((discovered & true_active).sum()) / n_disc
    n_active = int(true_active.sum())
    power = float((discovered & true_active).sum()) / n_active if n_active > 0 else np.nan
    return fdp, power, n_disc


# ============================= depth diagnostics (best effort) =================

def _try_load_depth_per_cell(prefix, embs_shape):
    """Best-effort: re-derive the (non-UCell_) log_total_counts column at the
    same per-cell row order get_data_cells() uses, for the FDR-validation
    depth-tercile stratification. Returns None if unavailable; never raises.
    """
    try:
        path = _resolve_label_file(prefix)
        lower = path.lower()
        if lower.endswith('.parquet'):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        if 'log_total_counts' not in df.columns:
            return None
        xy, _labels = get_pathway_labels(path)
        locs = np.stack([xy[:, 1], xy[:, 0]], -1)
        he_shape = None
        try:
            wsi = load_image(f'{prefix}he.tiff')
            he_shape = np.array(wsi.shape[:2])
        except Exception:
            return None
        target_shape = np.array(embs_shape[:2])
        rescale_factor = he_shape // target_shape
        locs = (locs.astype(float) / rescale_factor).round().astype(int)
        m = LABEL_BORDER_MARGIN
        inside = (
                (locs[:, 0] >= m) & (locs[:, 0] < embs_shape[0] - m) &
                (locs[:, 1] >= m) & (locs[:, 1] < embs_shape[1] - m))
        depth = df['log_total_counts'].to_numpy()[inside]
        return depth.astype(np.float32)
    except Exception as e:
        print(f'  (depth-tercile diagnostic unavailable: {e!r})')
        return None


# ================================== orchestration ================================

def run(
        prefix, epochs=None, n_states=3, device='cuda',
        k_folds=K_FOLDS, n_regions_try=N_REGIONS_TRY, min_train_frac=MIN_TRAIN_FRAC,
        buffer=BUFFER, seed=SEED, cv_epochs=CV_EPOCHS, cal_states=1,
        scales=SCALES, min_cells=None, threshold_quantile=THRESHOLD_QUANTILE,
        sweep_thresholds=False, tolerance='none', min_calib_pts=None,
        k_app=K_APP, n_pca=N_PCA, mode='both', force=False):
    """Main entry point, called from impute_integrated.py's --conformal hook
    (and usable standalone). `epochs`/`n_states`/`device` are accepted for a
    uniform call signature with impute() but only `device` is actually used
    here - cv_epochs/cal_states govern the conformal cross-validation.
    """
    if min_cells is None:
        min_cells = dict(SCALE_MIN_CELLS)
    assert tolerance in TOLERANCE_MODES, f'--tolerance must be one of {TOLERANCE_MODES}'

    print('utopia_conformal: loading data via get_data_cells(...)')
    embs, labels, locs, _nbr_idx = get_data_cells(prefix)
    names = list(labels.columns)
    Y = labels.to_numpy(np.float32)
    N, n_out = Y.shape
    locs_f = locs.astype(float)
    H, W = embs.shape[:2]

    out_dir = f'{prefix}iSCALE_output/conformal/'
    os.makedirs(out_dir, exist_ok=True)
    raw_dir = os.path.join(out_dir, 'raw_files')
    os.makedirs(raw_dir, exist_ok=True)

    # ---- 1. spatial folds + OOF predictions -------------------------------
    oof_path = os.path.join(out_dir, 'oof_predictions.npz')
    if os.path.exists(oof_path) and not force:
        z = np.load(oof_path, allow_pickle=True)
        fold, oof_pred = z['fold'], z['pred']
        n_regions_used, buffer_used = int(z['n_regions']), int(z['buffer'])
        print(f'loaded saved OOF predictions from {oof_path}')
    else:
        print('generating spatial folds ...')
        fold, n_regions_used, buffer_used = generate_spatial_folds(
                locs_f, k_folds=k_folds, n_regions_try=n_regions_try,
                min_train_frac=min_train_frac, buffer=buffer, seed=seed)
        print('computing out-of-fold predictions ...')
        oof_pred = compute_oof_predictions(
                embs, Y, locs, fold, buffer_used, cv_epochs=cv_epochs,
                cal_states=cal_states, device=device, seed=seed, k_folds=k_folds)
        np.savez(
                oof_path, cell_idx=np.arange(N), fold=fold, pred=oof_pred,
                names=np.array(names), n_regions=n_regions_used,
                buffer=buffer_used)
        print(f'saved {oof_path}')

    # ---- 2. appearance clusters --------------------------------------------
    clusters_path = os.path.join(out_dir, 'clusters.npy')
    if os.path.exists(clusters_path) and not force:
        clusters_image = np.load(clusters_path)
        print(f'loaded saved appearance clusters from {clusters_path}')
    else:
        print('computing tissue-appearance clusters ...')
        clusters_image = compute_appearance_clusters(
                embs, k_app=k_app, n_pca=n_pca, seed=seed)
        np.save(clusters_path, clusters_image)
        print(f'saved {clusters_path}')
    cell_cluster = clusters_image[locs[:, 0], locs[:, 1]]

    depth_per_cell = _try_load_depth_per_cell(prefix, embs.shape)
    has_depth = depth_per_cell is not None and len(depth_per_cell) == N

    quantiles_to_run = (list(THRESHOLD_QUANTILE_SWEEP) if sweep_thresholds
                        else [threshold_quantile])

    threshold_rows = []
    diag_rows = []
    fdr_rows = []
    fdr_depth_rows = []

    # A cell with a non-finite OOF prediction (rare: its own pixel's feature
    # vector was non-finite in whichever fold held it out) must not be
    # allowed into cell_block_stats' weighted sums, or it would NaN out the
    # whole block it falls in. Drop such cells once, up front, and use the
    # same filtered arrays for every scale.
    finite_cell = np.isfinite(Y).all(1) & np.isfinite(oof_pred).all(1)
    n_bad = int((~finite_cell).sum())
    if n_bad:
        print(f'{n_bad} / {N} cells have a non-finite label or OOF '
              f'prediction; excluded from block aggregation')
    locs_c = locs[finite_cell]
    Y_c = Y[finite_cell]
    oof_c = oof_pred[finite_cell]
    fold_onehot_c = np.eye(k_folds, dtype=np.float32)[fold[finite_cell]]
    depth_c = depth_per_cell[finite_cell] if has_depth else None

    for scale in scales:
        min_c = min_cells.get(scale, min_cells.get(str(scale), 2))
        h_blocks, w_blocks = block_shape(H, W, scale)
        cluster_grid = block_majority_cluster(clusters_image, scale, k_app)

        label_grid, cnt_grid = cell_block_stats(locs_c, Y_c, scale, h_blocks, w_blocks)
        oof_grid, _ = cell_block_stats(locs_c, oof_c, scale, h_blocks, w_blocks)
        fold_frac_grid, _ = cell_block_stats(locs_c, fold_onehot_c, scale, h_blocks, w_blocks)
        majority_fold_grid = np.where(
                cnt_grid[..., None] > 0, fold_frac_grid, -1).argmax(-1)
        purity_grid = np.nan_to_num(np.nanmax(fold_frac_grid, axis=-1), nan=0.0)

        depth_grid = None
        if has_depth:
            depth_grid, _ = cell_block_stats(locs_c, depth_c, scale, h_blocks, w_blocks)

        base_valid = (cnt_grid >= min_c) & (cluster_grid >= 0)

        # ---------------- production mode ----------------------------------
        if mode in ('production', 'both'):
            conf_arr = np.zeros((n_out, h_blocks, w_blocks), dtype=np.float16)
            calls10_arr = np.zeros((n_out, h_blocks, w_blocks), dtype=np.uint8)
            calls20_arr = np.zeros((n_out, h_blocks, w_blocks), dtype=np.uint8)

            for j, name in enumerate(names):
                calib_mask = base_valid & np.isfinite(label_grid[..., j]) & np.isfinite(oof_grid[..., j])
                if calib_mask.sum() == 0:
                    threshold_rows.append(dict(
                            pathway=name, scale=scale, mode='production',
                            quantile=threshold_quantile, threshold=np.nan,
                            n_calib_blocks=0))
                    continue

                map_path = (f'{prefix}iSCALE_output/super_res_gene_expression/'
                            f'cnts-super/{name}.pickle')
                try:
                    pred_map = load_pickle(map_path, verbose=False)
                except FileNotFoundError:
                    continue
                test_pred_grid = dense_block_mean(pred_map, scale, h_blocks, w_blocks)
                test_mask = base_valid & np.isfinite(test_pred_grid)
                if test_mask.sum() == 0:
                    continue

                t, p_adj_flat, diag = conformal_pathway_scale(
                        calib_y=label_grid[..., j][calib_mask],
                        calib_yhat=oof_grid[..., j][calib_mask],
                        calib_cluster=cluster_grid[calib_mask],
                        test_yhat=test_pred_grid[test_mask],
                        test_cluster=cluster_grid[test_mask],
                        threshold_quantile=threshold_quantile,
                        min_calib_pts=min_calib_pts, tolerance_mode=tolerance,
                        k_app=k_app)

                threshold_rows.append(dict(
                        pathway=name, scale=scale, mode='production',
                        quantile=threshold_quantile, threshold=t,
                        n_calib_blocks=int(calib_mask.sum())))
                for d in diag:
                    diag_rows.append(dict(
                            pathway=name, scale=scale, mode='production', **d))

                conf = pvals_to_confidence(p_adj_flat)
                conf_arr[j][test_mask] = conf.astype(np.float16)
                calls10_arr[j][test_mask] = (p_adj_flat <= FDR_LEVELS[0]).astype(np.uint8)
                calls20_arr[j][test_mask] = (p_adj_flat <= FDR_LEVELS[1]).astype(np.uint8)

            conf_path = os.path.join(out_dir, f'confidence_scale{scale}.npz')
            np.savez(
                    conf_path, confidence=conf_arr, calls_fdr10=calls10_arr,
                    calls_fdr20=calls20_arr, names=np.array(names))
            print(f'saved {conf_path}')

        # ---------------- validation mode (leave-one-fold-out) -------------
        if mode in ('validation', 'both'):
            for q in quantiles_to_run:
                for f_ in range(k_folds):
                    calib_mask_f = (base_valid & (majority_fold_grid != f_)
                                     & (purity_grid > 0.999))
                    test_mask_f = (base_valid & (majority_fold_grid == f_)
                                    & (purity_grid > 0.999))
                    if calib_mask_f.sum() == 0 or test_mask_f.sum() == 0:
                        continue
                    for j, name in enumerate(names):
                        cy = label_grid[..., j][calib_mask_f]
                        cp = oof_grid[..., j][calib_mask_f]
                        ty = label_grid[..., j][test_mask_f]
                        tp = oof_grid[..., j][test_mask_f]
                        finite_c = np.isfinite(cy) & np.isfinite(cp)
                        finite_t = np.isfinite(ty) & np.isfinite(tp)
                        if finite_c.sum() == 0 or finite_t.sum() == 0:
                            continue
                        t, p_adj_flat, _diag = conformal_pathway_scale(
                                calib_y=cy[finite_c], calib_yhat=cp[finite_c],
                                calib_cluster=cluster_grid[calib_mask_f][finite_c],
                                test_yhat=tp[finite_t],
                                test_cluster=cluster_grid[test_mask_f][finite_t],
                                threshold_quantile=q,
                                min_calib_pts=min_calib_pts, tolerance_mode=tolerance,
                                k_app=k_app)
                        label_test = ty[finite_t]
                        test_cluster_f = cluster_grid[test_mask_f][finite_t]
                        for fdr_level in FDR_LEVELS:
                            fdp, power, n_disc = get_fdp_power(
                                    p_adj_flat, label_test, t, fdr_level)
                            fdr_rows.append(dict(
                                    pathway=name, scale=scale, fold=f_,
                                    quantile=q, fdr_nominal=fdr_level,
                                    cluster=-1, fdp_realized=fdp, power=power,
                                    n_discoveries=n_disc,
                                    n_test_blocks=int(finite_t.sum())))
                            for k in range(k_app):
                                in_k = test_cluster_f == k
                                if in_k.sum() == 0:
                                    continue
                                fdp_k, power_k, n_disc_k = get_fdp_power(
                                        p_adj_flat[in_k], label_test[in_k], t,
                                        fdr_level)
                                fdr_rows.append(dict(
                                        pathway=name, scale=scale, fold=f_,
                                        quantile=q, fdr_nominal=fdr_level,
                                        cluster=k, fdp_realized=fdp_k,
                                        power=power_k, n_discoveries=n_disc_k,
                                        n_test_blocks=int(in_k.sum())))

                        if depth_grid is not None:
                            # depth_grid is 2D (one scalar mean per block):
                            # depth_per_cell is a 1D per-cell array.
                            depth_test = depth_grid[test_mask_f][finite_t]
                            finite_d = np.isfinite(depth_test)
                            if finite_d.sum() >= 3:
                                terc = pd.qcut(
                                        depth_test[finite_d], 3,
                                        labels=False, duplicates='drop')
                                for fdr_level in FDR_LEVELS:
                                    for tci in np.unique(terc):
                                        sel = np.zeros(len(depth_test), dtype=bool)
                                        sel[np.flatnonzero(finite_d)[terc == tci]] = True
                                        fdp_t, power_t, n_disc_t = get_fdp_power(
                                                p_adj_flat[sel], label_test[sel],
                                                t, fdr_level)
                                        fdr_depth_rows.append(dict(
                                                pathway=name, scale=scale, fold=f_,
                                                quantile=q, fdr_nominal=fdr_level,
                                                depth_tercile=int(tci),
                                                fdp_realized=fdp_t, power=power_t,
                                                n_discoveries=n_disc_t,
                                                n_test_blocks=int(sel.sum())))

    # ---- write diagnostics tables ------------------------------------------
    pd.DataFrame(threshold_rows).to_csv(
            os.path.join(out_dir, 'thresholds.tsv'), sep='\t', index=False)
    pd.DataFrame(diag_rows).to_csv(
            os.path.join(out_dir, 'calibration_diagnostics.tsv'), sep='\t', index=False)
    if fdr_rows:
        fdr_df = pd.DataFrame(fdr_rows)
        fdr_df.to_csv(os.path.join(out_dir, 'fdr_validation.tsv'), sep='\t', index=False)
        summary = (fdr_df[fdr_df.cluster == -1]
                   .groupby(['pathway', 'scale', 'quantile', 'fdr_nominal'])
                   .agg(mean_fdp=('fdp_realized', 'mean'),
                        worst_fdp=('fdp_realized', 'max'),
                        mean_power=('power', 'mean'))
                   .reset_index())
        summary.to_csv(
                os.path.join(out_dir, 'fdr_validation_summary.tsv'), sep='\t', index=False)
        _plot_fdr_curve(fdr_df, os.path.join(out_dir, 'fdr_curve.png'))
    if fdr_depth_rows:
        pd.DataFrame(fdr_depth_rows).to_csv(
                os.path.join(out_dir, 'fdr_validation_depth_stratified.tsv'),
                sep='\t', index=False)

    # ---- example confidence maps -------------------------------------------
    _save_example_maps(out_dir, names, scales)

    print(f'utopia_conformal.run done. Outputs in {out_dir}')


def _plot_fdr_curve(fdr_df, outfile):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pooled = fdr_df[fdr_df.cluster == -1]
    fig, ax = plt.subplots(figsize=(5, 5))
    for scale in sorted(pooled.scale.unique()):
        sub = pooled[pooled.scale == scale]
        agg = sub.groupby('fdr_nominal')['fdp_realized'].mean().reset_index()
        ax.plot(agg.fdr_nominal, agg.fdp_realized, 'o-', label=f'scale {scale}')
    lims = [0, max(0.25, pooled.fdr_nominal.max() if len(pooled) else 0.25)]
    ax.plot(lims, lims, 'k--', linewidth=1, label='nominal = realized')
    ax.set_xlabel('nominal FDR')
    ax.set_ylabel('realized FDP (mean over pathways/folds)')
    ax.set_title('Validation: nominal vs realized FDR')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f'saved {outfile}')


def _save_example_maps(out_dir, names, scales):
    scale = min(scales)
    conf_path = os.path.join(out_dir, f'confidence_scale{scale}.npz')
    if not os.path.exists(conf_path):
        return
    z = np.load(conf_path)
    conf = z['confidence']
    for j in range(min(3, conf.shape[0])):
        outfile = os.path.join(out_dir, f'example_confidence_{names[j]}.png')
        try:
            plot_matrix(conf[j].astype(np.float32), outfile)
        except Exception as e:
            print(f'  could not plot example map for {names[j]}: {e!r}')


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', type=str)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--n-states', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--k-folds', type=int, default=K_FOLDS)
    parser.add_argument('--n-regions-try', type=int, nargs='+', default=N_REGIONS_TRY)
    parser.add_argument('--min-train-frac', type=float, default=MIN_TRAIN_FRAC)
    parser.add_argument('--buffer', type=int, default=BUFFER)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--cv-epochs', type=int, default=CV_EPOCHS)
    parser.add_argument('--cal-states', type=int, default=1)
    parser.add_argument('--scales', type=int, nargs='+', default=list(SCALES))
    parser.add_argument('--threshold-quantile', type=float, default=THRESHOLD_QUANTILE)
    parser.add_argument('--sweep-thresholds', action='store_true')
    parser.add_argument('--tolerance', type=str, default='none', choices=TOLERANCE_MODES)
    parser.add_argument('--min-calib-pts', type=float, default=None)
    parser.add_argument('--k-app', type=int, default=K_APP)
    parser.add_argument('--n-pca', type=int, default=N_PCA)
    parser.add_argument('--mode', type=str, default='both',
                         choices=['production', 'validation', 'both'])
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    run(
            args.prefix, epochs=args.epochs, n_states=args.n_states,
            device=args.device, k_folds=args.k_folds,
            n_regions_try=args.n_regions_try, min_train_frac=args.min_train_frac,
            buffer=args.buffer, seed=args.seed, cv_epochs=args.cv_epochs,
            cal_states=args.cal_states, scales=args.scales,
            threshold_quantile=args.threshold_quantile,
            sweep_thresholds=args.sweep_thresholds, tolerance=args.tolerance,
            min_calib_pts=args.min_calib_pts, k_app=args.k_app, n_pca=args.n_pca,
            mode=args.mode, force=args.force)


if __name__ == '__main__':
    main()
