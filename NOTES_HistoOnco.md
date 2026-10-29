# iSCALE_HistoOnco

Copy of the iSCALE pipeline (all 81 Python modules from `amesch441/iSCALE`,
cloned 2026-09-02) with **`impute_integrated.py` modified** for the HistoOnco
project: predicting cancer-pathway activity scores (UCell on MSigDB Hallmark +
curated C6 oncogenic sets) from H&E histology features, on 10x Atera
single-cell-resolution spatial data.

## Contents

| item | notes |
|---|---|
| `impute_integrated.py` | **MODIFIED** — see below |
| the other 80 `*.py` files | unmodified iSCALE copies (kept flat; scripts import each other by bare name) |
| `_run_iSCALE_sbatch.sh`, `_run_iSCALE_bsub.sh` | cluster drivers |
| `download_checkpoints.sh` | fetches frozen HIPT/ViT feature-extractor weights |
| `environment.yml`, `requirements.txt` | `conda env create -f environment.yml` |
| `ucell_spatial_pipeline.ipynb` | produces the pathway-score label file |
| `README.md` | upstream iSCALE readme |

## Changes in `impute_integrated.py` (only this file was touched)

PARAMETERS block at the top of the file:

```
SINGLE_CELL_MODE     True  -> HistoOnco mode.  False -> original iSCALE, unchanged.
LABEL_FILE           None  -> auto-find {prefix}ucell_pathway_scores.{parquet,csv.gz,csv}
PATHWAY_COL_PREFIX   'UCell_'
SPATIAL_KNN_K        6      k nearest spatial neighbors for the smoothness term
SPATIAL_LOSS_WEIGHT  0.1    lambda in  L_total = L_pred + lambda * L_spatial
LABEL_BORDER_MARGIN  1
```

1. **Target = pathway scores, not gene counts.** `get_pathway_labels()` reads the
   label table (`.csv` / `.csv.gz` / `.parquet`, auto-detected), selects every
   column starting `UCell_`, derives `n_out` from the count. Nothing hardcoded.
   Raises a clear error if no `UCell_` column exists.
2. **`get_data_cells()`** replaces `get_data_smooth()` in single-cell mode: no
   DBSCAN cross-capture merge, no target min-max normalization, precomputes the
   k-NN spatial index once, writes pathway names to `{prefix}gene-names.txt` for
   the downstream scripts.
3. **`CellDataset`** replaces disk-mask spot pooling: one feature vector per cell,
   supervised directly against that cell's own label row. `SpotDataset` untouched.
4. **`SpatialSmoothModel`** (subclass of `ForwardSumModel`, same network):
   `L_total = MSE(pred, own_label) + lambda * MSE(pred, mean pred over k spatial NN)`.
   Squared error (not cosine); neighbor term on predictions, forwarded in-step.
5. Multi-capture / batch-correction logic **bypassed, not deleted** — reachable
   via `SINGLE_CELL_MODE = False`, commented at each bypass site.

The `SINGLE_CELL_MODE = False` path is unchanged from upstream.

## Run

```
# label file at {prefix}ucell_pathway_scores.csv (or .csv.gz / .parquet)
python impute_integrated.py {prefix} --epochs=1000 --device=cuda --n-states=5
```

`--dist` / `radius.txt` are accepted but unused in single-cell mode.

## One-step pipeline: UCell -> training -> conformal calibration

Three new files turn the above into a one-step pipeline. None of them modify
any existing file except the one additive change to `impute_integrated.py`
described below.

| file | purpose |
|---|---|
| `ucell_scores.py` | headless port of `ucell_spatial_pipeline.ipynb`: raw 10x count matrix + `cells.csv` -> `{prefix}ucell_pathway_scores.*` (`UCell_*` columns + a `log_total_counts` depth column). Needs the optional deps listed at the bottom of `requirements.txt`. |
| `utopia_conformal.py` | 5-fold spatial cross-validation (fold logic ported from the pilot notebook's Part E) -> out-of-fold predictions -> 8 tissue-appearance clusters (ported from Part E/F7c) -> UTOPIA-style conformal p-values / BH-adjusted confidence scores / FDR calls, at scales {1,2,4,8} feature pixels. See `DIFFERENCES.md` for every place this intentionally departs from UTOPIA's own code. Outputs land in `{prefix}iSCALE_output/conformal/`. |
| `histoonco_pipeline.py` | thin driver: `python histoonco_pipeline.py PREFIX --counts-h5 ... --cells-csv ... --alignment-csv ... --epochs 10 --n-states 3 --device cuda`. Does the microns -> Xenium px -> (inverse alignment matrix) -> full-res H&E px -> he.tiff px coordinate conversion the UCell notebook deliberately skips, then runs the existing `impute_integrated.py` training, then `utopia_conformal.py`. Each stage is skipped if its output already exists (`--force-ucell` / `--force-train` / `--force-conformal` to redo); `--skip-ucell`, `--skip-train`, `--no-conformal` to omit a stage entirely; `--run-features` to also shell out to `preprocess.py` + `extract_features.py` first. |

### The one new flag on `impute_integrated.py`

`--conformal` (boolean, default off). With it absent, behavior and outputs
are byte-for-byte identical to before - the diff is purely additive:

```python
# in get_args():
    parser.add_argument('--conformal', action='store_true')

# at the end of main(), after the existing impute(...) call:
    if args.conformal:
        import utopia_conformal
        utopia_conformal.run(
                args.prefix, epochs=args.epochs, n_states=args.n_states,
                device=args.device)
```

`utopia_conformal.run()` re-derives everything else it needs from `prefix`
via the existing `get_data_cells()`, so no other function signature in
`impute_integrated.py` changed.

### Known limitation: check `fdr_validation.tsv` before trusting `--tolerance none`

Synthetic testing found that validation-mode realized FDR can run well
above nominal with the default `--tolerance none`, for reasons unrelated to
code correctness - see DIFFERENCES.md item 7 for the full investigation.
Short version: a model trained with a whole geographic fold held out
extrapolates worse onto that fold than onto scattered individually-held-out
cells, which weakens calibration/test exchangeability. Always look at the
real run's `fdr_validation.tsv` / `fdr_curve.png` before trusting the
confidence maps at face value; `--tolerance weak|medium|strong` is
available if realized FDR runs hot on real data too.
