"""
Prepares the Photochem grid h5 for targeted reruns by marking specific cases
as not completed, so the existing Photochem_grid_121625.py picks them up.

Run on Hyak BEFORE re-launching Photochem_grid_121625.py:

    # Dry run — prints what would be changed, touches nothing:
    python MiniNepGrid_Scripts/prepare_photochem_rerun.py

    # Apply — backs up h5 then marks cases as not completed:
    python MiniNepGrid_Scripts/prepare_photochem_rerun.py --apply

Target cases (in Photochem_1D_lowmetal_restart.h5, grid shape (1,6,8,11,5,3)):
  1. All cases where completed==True AND status=='Photochem-not-converged'
  2. The 3 cases [2.0, 0.875, 50.0, 8.0, 0.2575, {5,7,9}] that previously
     had no PICASO PT profile (now fixed by patching the PICASO h5 via
     SlicePhotochemGrid_LowMetal.ipynb).

Prerequisites:
  - SlicePhotochemGrid_LowMetal.ipynb has been run to patch the PICASO h5.
  - The Photochem h5 exists at PHOTOCHEM_H5 below.
"""

import sys
import argparse
import shutil
import numpy as np
import h5py
from pathlib import Path

PHOTOCHEM_H5 = 'data/grid_results/Photochem_1D_lowmetal_restart.h5'

# ---------------------------------------------------------------------------
# Grid definition — must match get_gridvals_Photochem() in Photochem_grid_121625.py
# Low-metal slice keeps metallicities <= 2.375 (6 of the original 9 values).
# ---------------------------------------------------------------------------
_rad    = np.array([2.0])
_metal  = np.linspace(0.5, 2.375, 6)   # [0.5, 0.875, 1.25, 1.625, 2.0, 2.375]
_tint   = np.linspace(50, 400, 8)
_semi   = np.array([0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
_ctoO   = np.linspace(0.01, 1.0, 5)
_logKzz = np.array([5.0, 7.0, 9.0])
GRID_SHAPE = (len(_rad), len(_metal), len(_tint), len(_semi), len(_ctoO), len(_logKzz))
N_TOTAL    = int(np.prod(GRID_SHAPE))


def picaso_patched_flat_indices() -> list[int]:
    """Return the 3 Photochem flat indices for the PICASO-patched case."""
    i_rad   = int(np.where(np.isclose(_rad,   2.0))[0][0])
    i_metal = int(np.where(np.isclose(_metal, 0.875))[0][0])
    i_tint  = int(np.where(np.isclose(_tint,  50.0))[0][0])
    i_semi  = int(np.where(np.isclose(_semi,  8.0))[0][0])
    i_ctoO  = int(np.where(np.isclose(_ctoO,  0.2575))[0][0])
    return [
        int(np.ravel_multi_index(
            (i_rad, i_metal, i_tint, i_semi, i_ctoO, i_kzz),
            GRID_SHAPE
        ))
        for i_kzz in range(len(_logKzz))
    ]


def find_not_converged_indices(h5_path: Path) -> np.ndarray:
    """Return flat indices of completed cases with status 'Photochem-not-converged'."""
    with h5py.File(h5_path, 'r') as f:
        completed  = f['completed'][:]          # (N_total,) bool
        status_raw = f['results/status'][...]   # GRID_SHAPE + (1,) bytes

    status_flat = status_raw.reshape(N_TOTAL, -1)[:, 0]
    not_converged = np.array([s == b'Photochem-not-converged' for s in status_flat])
    return np.where(not_converged & completed)[0]


def main():
    parser = argparse.ArgumentParser(
        description='Mark photochem cases for rerun by clearing their completed flag'
    )
    parser.add_argument(
        '--apply', action='store_true',
        help='Actually modify the h5 (default is dry run — prints only)'
    )
    args = parser.parse_args()

    h5_path = Path(PHOTOCHEM_H5)
    if not h5_path.exists():
        print(f'ERROR: h5 not found: {h5_path}', file=sys.stderr)
        sys.exit(1)

    with h5py.File(h5_path, 'r') as f:
        n_complete = int(f['completed'][:].sum())

    not_conv_idx   = find_not_converged_indices(h5_path)
    picaso_idx     = picaso_patched_flat_indices()
    overlap        = set(not_conv_idx.tolist()) & set(picaso_idx)
    all_targets    = sorted(set(not_conv_idx.tolist()) | set(picaso_idx))

    print('=== Photochem rerun preparation ===')
    print(f'  h5 file       : {h5_path}')
    print(f'  grid shape    : {GRID_SHAPE}  ({N_TOTAL} total cases)')
    print(f'  completed     : {n_complete} / {N_TOTAL}')
    print()
    print(f'  Non-converged completed cases : {len(not_conv_idx)}')
    print(f'  PICASO-patched flat indices   : {picaso_idx}')
    print(f'    (Kzz=5 → {picaso_idx[0]}, Kzz=7 → {picaso_idx[1]}, Kzz=9 → {picaso_idx[2]})')
    print(f'  Already in non-converged set  : {len(overlap)} of 3 PICASO cases')
    print(f'  Total unique cases to rerun   : {len(all_targets)}')
    print()

    if not args.apply:
        print('[DRY RUN] No changes made. Re-run with --apply to mark cases for rerun.')
        return

    backup_path = str(h5_path).replace('.h5', '_pre_rerun_backup.h5')
    print(f'Backing up -> {backup_path} ...')
    shutil.copy2(str(h5_path), backup_path)
    print('Backup complete.')

    print(f'Clearing completed flag for {len(all_targets)} cases ...')
    with h5py.File(str(h5_path), 'a') as f:
        for idx in all_targets:
            f['completed'][idx] = False
    verified = 0
    with h5py.File(str(h5_path), 'r') as f:
        comp = f['completed'][:]
        verified = sum(1 for idx in all_targets if not comp[idx])
    print(f'Verified: {verified}/{len(all_targets)} cases marked as not completed.')
    print()
    print('Next step on Hyak:')
    print('  mpiexec -n <N_RANKS> python MiniNepGrid_Scripts/Photochem_grid_121625.py')


if __name__ == '__main__':
    main()
