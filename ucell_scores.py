"""Headless, faithful translation of ucell_spatial_pipeline.ipynb.

Reads a 10x-style cell x gene count matrix (.h5) plus a cells.csv with cell
centroids, runs UCell on MSigDB Hallmark + curated C6 oncogenic gene sets, and
writes a pathway-score label table consumable by impute_integrated.py's
get_pathway_labels() / _resolve_label_file():

    {prefix}ucell_pathway_scores.{parquet,csv.gz,csv}
    columns: cell_id, x, y, UCell_<pathway>..., log_total_counts

``log_total_counts`` is NOT UCell_-prefixed (get_pathway_labels only selects
PATHWAY_COL_PREFIX-prefixed columns), so it passes through as a depth
diagnostic for utopia_conformal.py without becoming a model target.

This script does NOT convert x/y into he.tiff pixel space - the coordinates
it writes are copied straight from cells.csv, exactly as the notebook does.
The micron -> Xenium px -> (inverse alignment matrix) -> full-res H&E px ->
he.tiff px conversion is the driver's job (histoonco_pipeline.py).

Every tunable below keeps the exact name it had in the notebook, and the
QC / coverage-filter / variance-filter logic is translated line-for-line
(not redesigned) so behaviour matches the pilot run. ``scanpy``, ``gseapy``,
``pyucell``, ``anndata`` and ``h5py`` are optional dependencies: imported
lazily inside run_ucell_pipeline(), never at module import time, so the rest
of the HistoOnco pipeline can import this file without them installed.
"""
import argparse
import gc
import os
import re
import time

import numpy as np
import pandas as pd

# =========================== TUNABLE PARAMETERS ============================
# (same names, same defaults as ucell_spatial_pipeline.ipynb Step 1)

# ---- Step 3: per-cell / per-gene QC ----
MIN_COUNTS_PER_CELL = 10     # drop cells with fewer total transcripts than this
MIN_GENES_PER_CELL = 3       # drop cells expressing fewer distinct genes than this
MIN_CELLS_PER_GENE = 5       # drop genes detected in fewer cells than this

# ---- Step 3: normalization ----
NORMALIZE_TARGET_SUM = 1e4   # counts-per-cell target for sc.pp.normalize_total

# ---- Step 5: gene-set coverage filter ----
MIN_GENE_SET_COVERAGE = 0.30  # keep a set only if >= this fraction of its genes are in the panel
MIN_GENES_RETAINED = 5        # ...AND at least this many genes remain after intersect

# ---- Step 6: UCell ----
UCELL_LAYER = 'counts'       # AnnData layer UCell ranks: "counts" (raw) or "norm" (lib-size norm)
UCELL_MAX_RANK = 1500        # pyucell default; genes ranked worse than this are ignored
UCELL_CHUNK_SIZE = 10_000    # cells scored per chunk; lower to ~2000 if UCell still OOMs
UCELL_N_JOBS = 1             # KEEP AT 1: n_jobs>1 pickles a full AnnData copy per worker
                              # process (loky backend) -> out-of-RAM on a small VM

# ---- Step 7: pathway-score variance filter ----
MIN_SCORE_VARIANCE = 1e-4    # drop UCell pathways whose across-cell score variance is below this

# ---- Step 4: MSigDB ----
# C6 oncogenic-signature name tokens relevant to cervical / HPV biology.
#   E6 -> TP53 degradation ; E7 -> RB1 inactivation -> E2F / cell cycle ; frequent PIK3CA activation
C6_NAME_TOKENS = [
        'E2F', 'RB_', 'RB1', 'P53', 'TP53',
        'CYCLIN_D1', 'CCND1', 'CYCLIN',
        'PI3K', 'PIK3', 'AKT', 'MTOR', 'PTEN',
]

# ---- cells.csv column autodetection (first match wins; add your file's names if needed) ----
CELL_ID_CANDIDATES = ['cell_id', 'cell_ID', 'Cell_ID', 'id', 'cell']
X_CANDIDATES = ['x_centroid', 'x', 'centroid_x', 'x_pixel', 'x_px', 'pxl_col_in_fullres']
Y_CANDIDATES = ['y_centroid', 'y', 'centroid_y', 'y_pixel', 'y_px', 'pxl_row_in_fullres']

# ---- output ----
PATHWAY_COL_PREFIX = 'UCell_'
LOG_DEPTH_COL = 'log_total_counts'
# =========================================================================


def _pick(columns, candidates, what):
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(
            f'No {what} column found in cells.csv. Looked for {candidates}, '
            f'file has {list(columns)}. Add the right name to the '
            f'*_CANDIDATES list at the top of ucell_scores.py.')


def subset_cells(adata, keep_bool, chunk=100_000):
    """Drop cells (rows) with minimal peak RAM. Faithful port of the notebook's
    subset_cells(): gathers kept rows in small chunks into pre-sized CSR
    arrays instead of taking a second whole-matrix copy."""
    import anndata as ad
    import scipy.sparse as sp

    keep_bool = np.asarray(keep_bool, dtype=bool)
    if keep_bool.all():
        return adata

    keep_idx = np.flatnonzero(keep_bool)
    X = adata.X
    if not sp.isspmatrix_csr(X):
        X = X.tocsr()
    nnz_per_row = np.diff(X.indptr)
    out_nnz = int(nnz_per_row[keep_idx].sum())

    nd = np.empty(out_nnz, X.data.dtype)
    ni = np.empty(out_nnz, X.indices.dtype)
    npr = np.empty(keep_idx.size + 1, dtype=np.int64)
    npr[0] = 0
    w = 0
    for s in range(0, keep_idx.size, chunk):
        rs = keep_idx[s:s + chunk]
        blk = X[rs]
        m = blk.data.size
        nd[w:w + m] = blk.data
        ni[w:w + m] = blk.indices
        npr[s + 1:s + 1 + rs.size] = w + blk.indptr[1:]
        w += m
        del blk
    new_X = sp.csr_matrix((nd, ni, npr), shape=(keep_idx.size, X.shape[1]))
    out = ad.AnnData(X=new_X, obs=adata.obs.iloc[keep_idx].copy(), var=adata.var)
    for k in list(adata.layers.keys()):
        out.layers[k] = new_X
    del X, nd, ni, npr
    gc.collect()
    return out


def load_10x_h5_lean(path, subsample_n=None, seed=0):
    """Low-memory 10x .h5 reader -> AnnData (cells x genes), CSR, all genes kept."""
    import anndata as ad
    import h5py
    import scipy.sparse as sp

    with h5py.File(path, 'r') as f:
        grp = 'matrix' if 'matrix' in f else list(f.keys())[0]
        g = f[grp]
        data = g['data'][:]
        indices = g['indices'][:]
        indptr = g['indptr'][:]
        n_genes, n_cells = (int(v) for v in g['shape'][:])
        barcodes = g['barcodes'][:].astype(str)
        if 'features' in g:
            feat = g['features']
            gene_names = feat['name'][:].astype(str)
            gene_ids = feat['id'][:].astype(str)
            ftype = feat['feature_type'][:].astype(str) if 'feature_type' in feat else None
        else:
            gene_names = g['gene_names'][:].astype(str)
            gene_ids = g['genes'][:].astype(str)
            ftype = None

    nnz = int(indptr[-1])
    print(f'  {n_cells:,} cells x {n_genes:,} genes, {nnz:,} non-zeros')
    data = data.astype(np.float32, copy=False)
    if nnz < np.iinfo(np.int32).max:
        indices = indices.astype(np.int32, copy=False)
        indptr = indptr.astype(np.int32, copy=False)

    X = sp.csr_matrix((data, indices, indptr), shape=(n_cells, n_genes))
    del data, indices, indptr
    gc.collect()

    var = pd.DataFrame({'gene_ids': gene_ids}, index=pd.Index(gene_names))
    if ftype is not None:
        var['feature_types'] = ftype
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=barcodes), var=var)
    adata.var_names_make_unique()

    if subsample_n is not None and subsample_n < adata.n_obs:
        rng = np.random.default_rng(seed)
        m = np.zeros(adata.n_obs, dtype=bool)
        m[rng.choice(adata.n_obs, size=subsample_n, replace=False)] = True
        adata = subset_cells(adata, m)
        gc.collect()
        print(f'  subsampled to {adata.n_obs:,} cells')
    return adata


def _pick_latest_human(vers):
    if hasattr(vers, 'columns'):
        col = 'db_ver' if 'db_ver' in vers.columns else vers.columns[0]
        cand = [str(v) for v in vers[col].tolist()]
    else:
        cand = [str(v) for v in vers]
    human = [v for v in cand if v.endswith('.Hs')]
    human = [v for v in human if 'msigdb' not in v.lower()] or human
    if not human:
        raise RuntimeError(f'No human (.Hs) MSigDB version in {cand}')

    def key(v):
        core = v.replace('msigdb_v', '').replace('.Hs', '')
        return [int(p) if p.isdigit() else p for p in core.split('.')]
    return sorted(human, key=key)[-1]


def run_ucell_pipeline(
        prefix, h5_path, cells_csv_path, subsample_n_cells=None,
        output_format='csv', seed=0):
    """End-to-end UCell scoring. Returns the path written.

    Mirrors ucell_spatial_pipeline.ipynb Steps 2 (cont.) - 8 exactly, plus one
    addition the notebook did not have: a non-UCell_-prefixed
    `log_total_counts` column (natural log of each retained cell's raw total
    transcript count) for downstream depth diagnostics.
    """
    import anndata as ad  # noqa: F401  (imported for clarity; used via adata)
    import scanpy as sc
    import scipy.sparse as sp
    from gseapy import Msigdb

    try:
        import pyucell as uc
    except ImportError as e:
        raise ImportError(
                'ucell_scores.py needs the optional dependencies scanpy, '
                'gseapy, pyucell (and h5py, anndata) to run. Install them or '
                'point --label-file at an existing ucell_pathway_scores '
                'table.') from e

    sc.settings.verbosity = 1

    # ---- Step 2 (cont.): load matrix + cells.csv, align ----
    print(f'RAM-lean load of {h5_path} ...')
    try:
        adata = load_10x_h5_lean(h5_path, subsample_n=subsample_n_cells, seed=seed)
    except Exception as e:
        print(f'lean reader failed ({e!r}); falling back to scanpy.read_10x_h5')
        adata = sc.read_10x_h5(h5_path)
        adata.var_names_make_unique()
    adata.obs_names = adata.obs_names.astype(str)
    gc.collect()
    print(f'Raw matrix (cells x genes): {adata.shape}')

    adata.var['n_cells_by_counts'] = np.bincount(adata.X.indices, minlength=adata.n_vars)

    cells_df = pd.read_csv(cells_csv_path)
    cid_col = _pick(cells_df.columns, CELL_ID_CANDIDATES, 'cell-id')
    x_col = _pick(cells_df.columns, X_CANDIDATES, 'x-centroid')
    y_col = _pick(cells_df.columns, Y_CANDIDATES, 'y-centroid')
    print(f'Using  cell_id={cid_col!r}  x={x_col!r}  y={y_col!r}')

    cells_df[cid_col] = cells_df[cid_col].astype(str)
    coords = (cells_df.set_index(cid_col)[[x_col, y_col]]
              .rename(columns={x_col: 'x', y_col: 'y'}))
    coords = coords[~coords.index.duplicated(keep='first')]

    csv_ids = set(coords.index)
    keep_mask = adata.obs_names.isin(csv_ids)
    n_dropped = int((~keep_mask).sum())
    print(f'cells in matrix: {adata.n_obs}  |  cells.csv rows: {len(csv_ids)}  |  '
          f'matrix cells w/o centroid (dropped): {n_dropped}')

    adata = subset_cells(adata, keep_mask)
    gc.collect()

    c = coords.reindex(adata.obs_names)
    adata.obs['cell_id'] = adata.obs_names.to_numpy()
    adata.obs['x'] = c['x'].to_numpy()
    adata.obs['y'] = c['y'].to_numpy()
    del c, coords, cells_df
    gc.collect()
    assert adata.obs[['x', 'y']].notna().all().all(), \
        'Some kept cells still have NaN coordinates'
    print(f'Aligned AnnData: {adata.shape}  ({n_dropped} cells dropped)')

    # ---- Step 3: per-cell / per-gene QC ----
    n0 = adata.n_obs
    counts_per_cell = np.asarray(adata.X.sum(axis=1)).ravel()
    genes_per_cell = adata.X.getnnz(axis=1)
    cell_ok = (counts_per_cell >= MIN_COUNTS_PER_CELL) & (genes_per_cell >= MIN_GENES_PER_CELL)
    print(f'cells failing QC: {int((~cell_ok).sum())} / {n0}')

    # log-depth diagnostic column, computed on the pre-QC counts for the cells
    # that survive QC (natural log of raw total transcript count per cell)
    log_total_counts_kept = np.log(counts_per_cell[cell_ok] + 1e-9)
    del counts_per_cell, genes_per_cell

    adata = subset_cells(adata, cell_ok)
    adata.obs[LOG_DEPTH_COL] = log_total_counts_kept.astype(np.float32)
    gc.collect()

    adata.var['n_cells_by_counts'] = np.bincount(adata.X.indices, minlength=adata.n_vars)
    n_genes_pass = int((adata.var['n_cells_by_counts'] >= MIN_CELLS_PER_GENE).sum())
    print(f'cells: {n0} -> {adata.n_obs}  |  genes detected in >= '
          f'{MIN_CELLS_PER_GENE} cells: {n_genes_pass}')

    # ---- layer UCell will read ----
    if UCELL_LAYER == 'counts':
        adata.layers['counts'] = adata.X
    elif UCELL_LAYER == 'norm':
        sc.pp.normalize_total(adata, target_sum=NORMALIZE_TARGET_SUM)
        adata.layers['norm'] = adata.X
    else:
        raise ValueError(f"UCELL_LAYER must be 'counts' or 'norm', got {UCELL_LAYER!r}")
    gc.collect()

    # ---- Step 4: MSigDB gene sets ----
    msig = Msigdb()
    vers = msig.list_dbver()
    dbver = _pick_latest_human(vers)
    print(f'using MSigDB version: {dbver}')

    hallmark = msig.get_gmt(category='h.all', dbver=dbver)
    if not hallmark:
        raise RuntimeError('gseapy returned an empty Hallmark collection')
    c6_all = msig.get_gmt(category='c6.all', dbver=dbver)
    if not c6_all:
        raise RuntimeError('gseapy returned an empty C6 collection')

    pat = re.compile('|'.join(C6_NAME_TOKENS))
    c6_selected = {k: v for k, v in c6_all.items() if pat.search(k.upper())}
    print(f'{len(c6_selected)} C6 sets selected by name match')

    gene_sets_raw = {**hallmark, **c6_selected}
    print(f'Total gene sets entering the coverage filter: {len(gene_sets_raw)} '
          f'({len(hallmark)} Hallmark + {len(c6_selected)} C6)')

    # ---- Step 5: filter gene sets to genes usable in this panel ----
    usable = pd.Series(True, index=adata.var_names)
    if 'feature_types' in adata.var.columns:
        usable &= (adata.var['feature_types'].to_numpy() == 'Gene Expression')
    usable &= (adata.var['n_cells_by_counts'].to_numpy() >= MIN_CELLS_PER_GENE)
    panel = set(adata.var_names[usable.to_numpy()])
    print(f'usable genes: {len(panel)} / {adata.n_vars}')

    gene_sets_filtered = {}
    for name, genes in gene_sets_raw.items():
        genes = sorted(set(map(str, genes)))
        present = [g for g in genes if g in panel]
        frac = len(present) / max(len(genes), 1)
        kept = (frac >= MIN_GENE_SET_COVERAGE) and (len(present) >= MIN_GENES_RETAINED)
        if kept:
            gene_sets_filtered[name] = present
    print(f'Retained {len(gene_sets_filtered)} / {len(gene_sets_raw)} gene sets')
    assert gene_sets_filtered, \
        'No gene sets survived Step 5 - loosen MIN_GENE_SET_COVERAGE / MIN_GENES_RETAINED'

    # ---- Step 6: run UCell ----
    assert UCELL_LAYER in adata.layers, f'UCELL_LAYER={UCELL_LAYER!r} not in {list(adata.layers.keys())}'
    import inspect
    want = dict(signatures=gene_sets_filtered, layer=UCELL_LAYER, max_rank=UCELL_MAX_RANK,
                chunk_size=UCELL_CHUNK_SIZE, n_jobs=UCELL_N_JOBS, suffix='_UCell')
    ok = set(inspect.signature(uc.compute_ucell_scores).parameters)
    kw = {k: v for k, v in want.items() if k in ok}
    print(f'{adata.n_obs:,} cells x {len(gene_sets_filtered)} sets  |  {kw}')
    t0 = time.time()
    uc.compute_ucell_scores(adata, **kw)
    gc.collect()

    score_cols = [f'{name}_UCell' for name in gene_sets_filtered]
    missing = [c for c in score_cols if c not in adata.obs.columns]
    assert not missing, f'UCell did not create expected columns, e.g. {missing[:3]}'
    adata.obs[score_cols] = adata.obs[score_cols].astype(np.float32)
    print(f'Computed {len(score_cols)} UCell pathway scores for {adata.n_obs:,} '
          f'cells in {time.time() - t0:.0f}s')

    # ---- Step 7: pathway-score variance filter ----
    score_df = adata.obs[score_cols]
    variances = score_df.var(axis=0).sort_values()
    low_var = variances[variances < MIN_SCORE_VARIANCE].index.tolist()
    retained_cols = [c for c in score_cols if c not in low_var]
    print(f'Near-zero-variance pathways dropped: {len(low_var)}  |  retained: {len(retained_cols)}')
    assert retained_cols, 'Every pathway was filtered as low-variance - lower MIN_SCORE_VARIANCE'

    # ---- Step 8: save the training-label table ----
    rename = {c: f'{PATHWAY_COL_PREFIX}{c[:-6] if c.endswith("_UCell") else c}' for c in retained_cols}
    out = (adata.obs[['cell_id', 'x', 'y', LOG_DEPTH_COL] + retained_cols]
           .rename(columns=rename)
           .reset_index(drop=True))
    out[list(rename.values())] = out[list(rename.values())].astype(np.float32)
    out[LOG_DEPTH_COL] = out[LOG_DEPTH_COL].astype(np.float32)

    if output_format == 'parquet':
        out_path = f'{prefix}ucell_pathway_scores.parquet'
        out.to_parquet(out_path, index=False)
    elif output_format == 'csv.gz':
        out_path = f'{prefix}ucell_pathway_scores.csv.gz'
        out.to_csv(out_path, index=False, float_format='%.5f')
    else:
        out_path = f'{prefix}ucell_pathway_scores.csv'
        out.to_csv(out_path, index=False, float_format='%.5f')
    print(f'wrote {out_path}  shape={out.shape}')
    return out_path


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', type=str)
    parser.add_argument('--h5', type=str, required=True, help='cell_feature_matrix.h5 path')
    parser.add_argument('--cells-csv', type=str, required=True, help='cells.csv (or .csv.gz) path')
    parser.add_argument('--subsample-n-cells', type=int, default=None)
    parser.add_argument('--output-format', type=str, default='csv',
                         choices=['csv', 'csv.gz', 'parquet'])
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main():
    args = get_args()
    run_ucell_pipeline(
            prefix=args.prefix, h5_path=args.h5, cells_csv_path=args.cells_csv,
            subsample_n_cells=args.subsample_n_cells,
            output_format=args.output_format, seed=args.seed)


if __name__ == '__main__':
    main()
