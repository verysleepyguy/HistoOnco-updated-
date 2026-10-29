"""Thin HistoOnco driver: UCell labels -> impute_integrated training -> UTOPIA
conformal calibration.

    python histoonco_pipeline.py PREFIX --counts-h5 ... --cells-csv ... \
        --alignment-csv ... [--pixel-size-um 0.2125 --he-downsample 3.6525] \
        --epochs 10 --n-states 3 --device cuda \
        [--skip-ucell] [--skip-train] [--no-conformal] [--run-features]

Each stage is skipped automatically if its output already exists (pass
--force-<stage> to redo it anyway). This file only wires together existing
entry points (ucell_scores.run_ucell_pipeline, impute_integrated.py as a
subprocess, utopia_conformal.run) plus the one conversion step the UCell
notebook deliberately does not do: turning cells.csv's x/y (microns, in the
Xenium coordinate frame) into he.tiff pixel coordinates.

Coordinate chain (see the project's alignment CSV):
    microns  --/ pixel_size_um -->  Xenium px
             --apply inverse of the 3x3 alignment matrix-->  full-res H&E px
             --/ he_downsample -->  he.tiff px
(impute_integrated.get_data_cells() takes it from he.tiff px down to the
16-um feature grid itself, via the he.tiff-shape / embeddings-shape ratio -
this driver's job stops at he.tiff px.)
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

import ucell_scores


def convert_coords_to_he_pixels(df, alignment_csv, pixel_size_um, he_downsample):
    """df has columns x, y in microns (the Xenium coordinate frame the
    alignment matrix was fit in). Returns a copy with x, y overwritten in
    he.tiff pixel units."""
    M = np.loadtxt(alignment_csv, delimiter=',')
    assert M.shape == (3, 3), f'alignment matrix has shape {M.shape}, expected (3, 3)'
    M_inv = np.linalg.inv(M)

    xen_px = df[['x', 'y']].to_numpy(dtype=float) / pixel_size_um
    homog = np.concatenate([xen_px, np.ones((len(xen_px), 1))], axis=1)
    he_full_px = (M_inv @ homog.T).T[:, :2]
    he_tiff_px = he_full_px / he_downsample

    out = df.copy()
    out['x'] = he_tiff_px[:, 0]
    out['y'] = he_tiff_px[:, 1]
    return out


def apply_coordinate_conversion(label_path, alignment_csv, pixel_size_um, he_downsample):
    lower = label_path.lower()
    if lower.endswith('.parquet'):
        df = pd.read_parquet(label_path)
    else:
        df = pd.read_csv(label_path)
    df = convert_coords_to_he_pixels(df, alignment_csv, pixel_size_um, he_downsample)
    if lower.endswith('.parquet'):
        df.to_parquet(label_path, index=False)
    elif lower.endswith('.csv.gz'):
        df.to_csv(label_path, index=False, float_format='%.5f')
    else:
        df.to_csv(label_path, index=False, float_format='%.5f')
    return df


def sanity_check_registration(df, prefix, grid_factor=16):
    """Fraction of cells landing inside the embedding grid, and mean cells
    per occupied grid pixel (expect roughly 2.9 for this dataset)."""
    from utils import load_image, load_pickle

    try:
        embs = load_pickle(f'{prefix}embeddings-hist.pickle', verbose=False)
        H, W = np.asarray(embs['cls'][0]).shape[-2:]
    except FileNotFoundError:
        print('  (skipping registration sanity check: embeddings-hist.pickle not found yet)')
        return

    wsi = load_image(f'{prefix}he.tiff', verbose=False)
    current_shape = np.array(wsi.shape[:2])
    target_shape = np.array([H, W])
    rescale_factor = current_shape // target_shape

    locs = np.stack([df['y'].to_numpy(float), df['x'].to_numpy(float)], -1)
    locs = (locs / rescale_factor).round().astype(int)
    inside = ((locs[:, 0] >= 0) & (locs[:, 0] < H) &
              (locs[:, 1] >= 0) & (locs[:, 1] < W))
    frac_inside = float(inside.mean())

    pix = locs[inside, 0].astype(np.int64) * W + locs[inside, 1]
    n_occ = len(np.unique(pix))
    cells_per_pixel = inside.sum() / max(n_occ, 1)
    print(f'  registration sanity check: {frac_inside:.1%} of cells land inside the '
          f'{H} x {W} feature grid, {cells_per_pixel:.1f} cells per occupied pixel '
          f'(expect roughly 2.9)')


def find_label_file(prefix):
    for ext in ('.parquet', '.csv.gz', '.csv'):
        path = f'{prefix}ucell_pathway_scores{ext}'
        if os.path.exists(path):
            return path
    return None


def run_ucell_stage(args):
    label_path = find_label_file(args.prefix)
    if label_path is not None and not args.force_ucell:
        print(f'[1/3] UCell labels already exist at {label_path}, skipping '
              f'(pass --force-ucell to redo)')
        return label_path

    print('[1/3] running UCell scoring ...')
    label_path = ucell_scores.run_ucell_pipeline(
            prefix=args.prefix, h5_path=args.counts_h5,
            cells_csv_path=args.cells_csv,
            subsample_n_cells=args.subsample_n_cells,
            output_format=args.ucell_output_format, seed=args.seed)

    print('[1/3] converting label coordinates to he.tiff pixel space ...')
    df = apply_coordinate_conversion(
            label_path, args.alignment_csv, args.pixel_size_um, args.he_downsample)
    sanity_check_registration(df, args.prefix)
    return label_path


def have_prediction_maps(prefix):
    map_dir = f'{prefix}iSCALE_output/super_res_gene_expression/cnts-super/'
    return os.path.isdir(map_dir) and any(f.endswith('.pickle') for f in os.listdir(map_dir))


def run_train_stage(args):
    if have_prediction_maps(args.prefix) and not args.force_train:
        print('[2/3] prediction maps already exist, skipping training '
              '(pass --force-train to redo)')
        return

    print('[2/3] running impute_integrated.py ...')
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
            sys.executable, 'impute_integrated.py', args.prefix,
            f'--n-states={args.n_states}', f'--device={args.device}']
    if args.epochs is not None:
        cmd.append(f'--epochs={args.epochs}')
    subprocess.run(cmd, cwd=repo_dir, check=True)


def have_conformal_outputs(prefix):
    out_dir = f'{prefix}iSCALE_output/conformal/'
    return os.path.exists(os.path.join(out_dir, 'oof_predictions.npz'))


def run_conformal_stage(args):
    if args.no_conformal:
        print('[3/3] --no-conformal set, skipping conformal calibration')
        return
    if have_conformal_outputs(args.prefix) and not args.force_conformal:
        print('[3/3] conformal outputs already exist, skipping '
              '(pass --force-conformal to redo)')
        return

    print('[3/3] running utopia_conformal ...')
    import utopia_conformal
    utopia_conformal.run(
            args.prefix, epochs=args.epochs, n_states=args.n_states,
            device=args.device)


def run_feature_stage(args):
    """Optional: shell out to the existing preprocess.py / extract_features.py
    if the caller asked for it. Assumes he.tiff / embeddings-hist.pickle are
    produced by those scripts in their usual way; this driver only checks
    that the files exist otherwise."""
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    he_tiff = f'{args.prefix}he.tiff'
    embs_pickle = f'{args.prefix}embeddings-hist.pickle'
    if os.path.exists(he_tiff) and os.path.exists(embs_pickle) and not args.force_features:
        print('[0/3] he.tiff and embeddings-hist.pickle already exist, skipping feature extraction')
        return
    if not args.run_features:
        if not (os.path.exists(he_tiff) and os.path.exists(embs_pickle)):
            raise FileNotFoundError(
                    f'{he_tiff} and/or {embs_pickle} do not exist, and '
                    f'--run-features was not passed. Run preprocess.py and '
                    f'extract_features.py yourself, or pass --run-features.')
        return
    print('[0/3] running preprocess.py ...')
    subprocess.run([sys.executable, 'preprocess.py', args.prefix], cwd=repo_dir, check=True)
    print('[0/3] running extract_features.py ...')
    subprocess.run([sys.executable, 'extract_features.py', args.prefix], cwd=repo_dir, check=True)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', type=str)
    parser.add_argument('--counts-h5', type=str)
    parser.add_argument('--cells-csv', type=str)
    parser.add_argument('--alignment-csv', type=str)
    parser.add_argument('--pixel-size-um', type=float, default=0.2125)
    parser.add_argument('--he-downsample', type=float, default=3.6525)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--n-states', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--subsample-n-cells', type=int, default=None)
    parser.add_argument('--ucell-output-format', type=str, default='csv',
                         choices=['csv', 'csv.gz', 'parquet'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip-ucell', action='store_true')
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--no-conformal', action='store_true')
    parser.add_argument('--run-features', action='store_true',
                         help='shell out to preprocess.py + extract_features.py first')
    parser.add_argument('--force-features', action='store_true')
    parser.add_argument('--force-ucell', action='store_true')
    parser.add_argument('--force-train', action='store_true')
    parser.add_argument('--force-conformal', action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    if not args.prefix.endswith('/') and not args.prefix.endswith('\\'):
        print(f'WARNING: prefix {args.prefix!r} does not end in a path '
              f'separator; iSCALE conventions expect e.g. "data/sample/"')

    run_feature_stage(args)

    if not args.skip_ucell:
        if args.counts_h5 is None or args.cells_csv is None or args.alignment_csv is None:
            raise ValueError(
                    '--counts-h5, --cells-csv and --alignment-csv are required '
                    'unless --skip-ucell is passed')
        run_ucell_stage(args)
    else:
        print('[1/3] --skip-ucell set, skipping UCell scoring')

    if not args.skip_train:
        run_train_stage(args)
    else:
        print('[2/3] --skip-train set, skipping training')

    run_conformal_stage(args)


if __name__ == '__main__':
    main()
