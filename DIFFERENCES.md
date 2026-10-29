# Differences from UTOPIA (paper vs. code) and from a literal UTOPIA port

`utopia_conformal.py` reimplements UTOPIA's conformal-inference idea for
HistoOnco's single-cell pathway-activity maps; it does not call into the
UTOPIA package (tests import UTOPIA's own functions only to cross-check our
reimplementation - see `tests/test_conformal_pvals.py`). This file lists
every place our code intentionally differs from UTOPIA's actual GitHub
source (`UTOPIA/UTOPIA/inference/conformal.py`, `confidence.py`, `fdr.py`,
`calibration/generate_folds.py`), as opposed to differences from the paper's
description, per the project instructions.

## 1. The batch-correction "tolerance" step defaults to off here, on there

UTOPIA's `get_adj_pvals` (`inference/conformal.py`) defaults to
`correct_batch_effect=True, correct_batch_effect_magnitude="medium"`, so
every call that doesn't explicitly pass `correct_batch_effect=False` applies
`compute_tolerance(...)` - a quantile-shift correction between the
calibration and test `yhat` distributions. This correction does not appear
in the UTOPIA paper's description of the conformal procedure; it is a
code-only addition.

We reimplemented `compute_tolerance` identically (for optional use) but
default `--tolerance none` (tolerance = 0 everywhere), matching the paper,
not UTOPIA's shipped default. `--tolerance weak|medium|strong` is available
if you want to reproduce UTOPIA's own default behavior.

## 2. `min_calib_pts` is applied unconditionally here, only under `verbose=True` there

In UTOPIA's `get_adj_pvals`:

```python
if verbose:
    print(f"Minimum calibration points for cluster {cluster} is {min_calib_pts}.")
    if len(calib_yhat) < min_calib_pts:
        print(f"Skip cluster {cluster} because the calibration data only contains {len(calib_yhat)} points.")
        continue
```

The `if len(calib_yhat) < min_calib_pts: continue` line is nested inside
`if verbose:`. Since `verbose` defaults to `False` and none of UTOPIA's own
calling code passes `verbose=True`, **the minimum-calibration-points check
is never actually applied** in a default UTOPIA run, no matter how few
calibration points a cluster has.

We apply the check unconditionally in `conformal_pathway_scale()`
(`utopia_conformal.py`): a cluster with fewer than `min_calib_pts`
calibration points (default: UTOPIA's own fallback formula,
`max(200, n_test_in_cluster * 0.01)`) gets `p = 1` for all its test blocks,
and every cluster's calibration count is logged in
`calibration_diagnostics.tsv` regardless of whether it was skipped.

## 3. Null definition: quantile of block labels, not UTOPIA's fixed `null_number`

UTOPIA takes a fixed `null_number` per gene/cell-type (e.g. `0.05` of the
99.99th percentile of raw counts) because its targets (gene counts, cell-type
fractions) have a natural zero / fixed scale. UCell scores don't - "0" is not
a meaningful floor. Per the project design, the null threshold `t` is instead
the `THRESHOLD_QUANTILE`-quantile (default 0.75) of the calibration block
labels, computed per pathway and scale (and, in validation mode, from the
calibration folds only, never the held-out fold). This is a deliberate
redesign of the null, not a bug fix - UTOPIA's own fixed-null convention is
simply inapplicable to a bounded, rank-based score.

## 4. Production test blocks use the dense prediction map; calibration blocks use sparse per-cell OOF data

UTOPIA calibrates and tests on the same kind of per-pixel array throughout
(dense WSI arrays sliced into ROI/non-ROI). Here, calibration necessarily
comes from the 5-fold OOF predictions at the finitely many cell locations
(no OOF prediction exists for a tissue pixel that never happens to be a
cell's grid position); but the final deliverable confidence maps are built
from the single production model's dense `cnts-super/*.pickle` maps, which
cover every tissue pixel. So a production test block's "prediction" is the
block-mean of the dense full-resolution map, while the matching calibration
block's "prediction" is the block-mean of the sparse per-cell OOF values.
This is a known, documented small mismatch: the production model that makes
the dense map is trained on *all* cells (more data, no held-out buffer),
while the calibration distribution comes from the weaker, buffer-restricted
per-fold models. If anything this makes the conformal bands conservative
(calibration residuals come from under-trained models relative to the
production model being calibrated), but it is not a proof of validity.

## 5. P-value denominator includes "non-null" calibration points (kept as UTOPIA does it)

`nonconformity_score_clip` assigns `+inf` to any calibration point whose true
label is above the null threshold (`y > t`), and `compute_pvals`'s
denominator is `n + 1` where `n` is the *total* number of calibration points
passed in (null and non-null), not just the null ones. The `+inf` scores
never count as "smaller" than a finite test score, so this only makes
p-values slightly more conservative than if non-null points were dropped
before the count. We kept this exactly as UTOPIA's code does it (this is
`conformal.py`'s actual behavior, not a bug) rather than "fixing" it.

## 6. Grouping: one clustering at scale 1, majority-voted to coarser scales - UTOPIA reclusters independently per scale

Checked per the project instructions: UTOPIA's `img_process/histology_clustering.py`
does **not** derive a coarse-scale cluster label from the scale-1 one. It
runs PCA+KMeans **again, independently, at every scale**, each time on that
scale's own separately-downsampled embeddings
(`embeddings_scale_{S}.npy`, produced by `scaling_image_features.py`) -
so UTOPIA's scale-4 clustering is a fresh clustering of scale-4-resolution
features, not a coarsened version of its scale-1 clustering, and a pixel's
cluster id at one scale has no defined relationship to its cluster id at
another.

Matching that is infeasible here without also re-deriving block-level
averaged embeddings per scale (the project design (D) explicitly asks for
one clustering, computed once at pixel level and saved). Our appearance
clusters are computed once at the 16-um feature-pixel level; a block's
cluster at coarser scales (32/64/128 um) is the majority cluster among its
pixels (not among the cells landing in it - a block typically has far more
pixels than cells at scale 1, and this keeps the cluster assignment
identical whether or not any cell happens to occupy a given pixel).

## 7. Finding from synthetic testing: spatial-fold validation runs hot on realized FDR

This isn't a difference from UTOPIA's code - it's an empirical finding from
`tests/test_synthetic_e2e.py` worth flagging explicitly, because it affects
whether `--tolerance none` (this project's default) is trustworthy.

On the ~20k-cell synthetic dataset, validation-mode realized FDP runs well
above nominal (examples seen during development: ~55% realized at nominal
10%, ~59% at nominal 20%; it moved somewhat with signal strength, fold
count, and training length, but was never close to the nominal+0.03 bound).
This was investigated in depth before accepting it as a property of the
design rather than a bug:

- `bh_adjust`, `compute_pvals`, `nonconformity_score_clip` and
  `pvals_to_confidence` all match UTOPIA's own reference implementation
  exactly (`tests/test_conformal_pvals.py`).
- An idealized i.i.d. calibration/test split (same correlation, same
  label-distribution shape, no spatial structure at all) is close to
  nominal.
- A plain *random* (non-spatial) split of the real pipeline's own OOF
  predictions is close to nominal.
- The real, *spatial* leave-one-fold-out split - the one the design calls
  for - is not, and the gap persists across every combination of fold
  count, region count, training length, and synthetic signal strength
  tried.

The common factor: a model trained with an entire geographic fold (plus
buffer) withheld extrapolates measurably worse onto that region than a
model that merely had scattered individual cells withheld. Calibrating on
the first kind of OOF prediction (other folds, interpolation-like) and
testing on the second kind (held-out fold, extrapolation-like) breaks the
exchangeability the conformal guarantee depends on. This is a genuine
spatial covariate-shift effect, not an implementation defect - in the same
family as the depth confound the project design already calls out ("report
it, do not try to fix it here").

It's expected to matter less on real data (far more cells, far more
effective regions averaging out any single region's idiosyncrasy, longer
training), but it is not something to assume away. Before trusting
`--tolerance none` in production, check the real `fdr_validation.tsv` /
`fdr_curve.png`; if realized FDP runs hot there too, `--tolerance
weak|medium|strong` (UTOPIA's own correction for exactly this kind of
calibration/test batch effect) is the documented way to compensate, at the
cost of matching UTOPIA's code rather than the paper (difference #1 above).

## 8. Spatial folds use the pilot notebook's k-means-region algorithm, not UTOPIA's rectangle tiling

UTOPIA's own `calibration/generate_folds.py` recursively tiles rectangular
ROI sub-regions (`generate_folds_final` / `split_into_folds`). Per project
instructions, `utopia_conformal.py` ports the project's own pilot-notebook
fold algorithm instead (k-means regions on cell locations, dealt into folds
by greedy load-balancing, with a shrinking buffer) - this is a deliberate
substitution the project asked for, not an oversight.
