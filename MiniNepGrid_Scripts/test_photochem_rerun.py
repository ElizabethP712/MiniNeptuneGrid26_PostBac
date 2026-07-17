"""
Local MPI smoke-test for the photochem grid pipeline.

Copies Photochem_1D_lowmetal_restart.h5 to a temporary test file, marks
all cases as completed except 2 (the Kzz=5 and Kzz=7 PICASO-patched cases),
then runs gridutils.make_grid on that test file so the full master-worker-h5
pipeline can be verified locally before submitting to Hyak.

Run with exactly 2 ranks (1 master + 1 worker):

    conda activate subneptune
    cd <project_root>
    mpiexec -n 2 python MiniNepGrid_Scripts/test_photochem_rerun.py

After it completes, inspect the output:

    python MiniNepGrid_Scripts/test_photochem_rerun.py --check

The test h5 is written to data/grid_results/Photochem_1D_lowmetal_mpitest.h5.
It is safe to delete after the test.
"""

import sys
import os
import shutil
import argparse
import numpy as np
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / 'MiniNepGrid_Scripts'))
sys.path.insert(0, str(_root / 'ReflectedSpectra_Scripts'))
os.chdir(str(_root))

SRC_H5  = 'data/grid_results/Photochem_1D_lowmetal_restart.h5'
TEST_H5  = 'data/grid_results/Photochem_1D_lowmetal_mpitest.h5'
TEST_LOG = TEST_H5.replace('.h5', '.log')

# 2 of the 3 PICASO-patched flat indices (Kzz=5 and Kzz=7) — stopping at 2
# so the local test finishes in a reasonable time.
TEST_INDICES = [1458, 1459]


def setup_test_h5():
    """Copy src h5, mark all completed, then clear just the 2 test cases."""
    import h5py
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_rank() != 0:
        return

    print(f'Copying {SRC_H5} -> {TEST_H5} ...')
    shutil.copy2(SRC_H5, TEST_H5)

    with h5py.File(TEST_H5, 'a') as f:
        f['completed'][:] = True
        for idx in TEST_INDICES:
            f['completed'][idx] = False
    print(f'Test h5 ready: {len(TEST_INDICES)} cases marked for rerun ({TEST_INDICES})')


def check_output():
    """Read the test h5 and verify the 2 test cases have sensible results."""
    import h5py
    if not Path(TEST_H5).exists():
        print(f'ERROR: {TEST_H5} not found — run the MPI test first.')
        return

    grid_shape = (1, 6, 8, 11, 5, 3)

    with h5py.File(TEST_H5, 'r') as f:
        completed = f['completed'][:]
        status    = f['results/status'][...]
        pressure  = f['results/pressure_sol'][...]
        temp      = f['results/temperature_sol'][...]

    print('=== MPI test output check ===')
    for idx in TEST_INDICES:
        nd = np.unravel_index(idx, grid_shape)
        s  = status[nd].flat[0]
        p  = pressure[nd]
        t  = temp[nd]
        print(f'\n  flat index {idx}  nd_index {nd}')
        print(f'  completed   : {completed[idx]}')
        print(f'  status      : {s}')
        print(f'  n_layers    : {len(p)}  (expected 100)')
        print(f'  P range     : {p.max():.3e} – {p.min():.3e} dyn/cm²')
        print(f'  T range     : {t.max():.1f} – {t.min():.1f} K')
        if len(p) != 100:
            print(f'  WARNING: expected 100 layers, got {len(p)}')
        if np.isnan(p).any():
            print(f'  WARNING: NaN values in pressure — case likely errored')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true',
                        help='Inspect test h5 output instead of running the MPI test')
    args = parser.parse_args()

    if args.check:
        check_output()
        sys.exit(0)

    from mpi4py import MPI
    import gridutils
    from Photochem_grid_121625 import (
        Photochem_1D_model, get_gridvals_Photochem, setup_rank_debug_logging
    )

    setup_rank_debug_logging()
    setup_test_h5()
    MPI.COMM_WORLD.Barrier()

    gridutils.make_grid(
        model_func=Photochem_1D_model,
        gridvals=get_gridvals_Photochem(),
        filename=TEST_H5,
        progress_filename=TEST_LOG,
    )
