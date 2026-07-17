"""
Local MPI smoke-test for the new low-metal PICASO grid pipeline.

Runs PICASO_climate_model for 2 cases from the low-metal grid
(rad=2, metal=[0.5, 0.875], tint=50, semi=5, ctoO=0.01) to verify the
full master-worker-h5 pipeline works locally before submitting to Hyak.

Run with exactly 2 ranks (1 master + 1 worker):

    conda activate subneptune_nb_picaso
    cd <project_root>
    mpiexec -n 2 python MiniNepGrid_Scripts/test_picaso_lowmetal.py

After it completes, inspect the output:

    python MiniNepGrid_Scripts/test_picaso_lowmetal.py --check

The test h5 is written to data/grid_results/PICASO_climate_lowmetal_mpitest.h5.
It is safe to delete after the test.

NOTE: Before running, verify that inputs_climate(nstr=...) is supported in
the subneptune_nb_picaso environment:
    python -c "import inspect, picaso.justdoit as jdi; print(inspect.signature(jdi.inputs.inputs_climate))"
If 'nstr' is absent, update PICASO_Climate_grid_121625.py before proceeding.
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / 'MiniNepGrid_Scripts'))
sys.path.insert(0, str(_root / 'ReflectedSpectra_Scripts'))
os.chdir(str(_root))

TEST_H5  = 'data/grid_results/PICASO_climate_lowmetal_mpitest.h5'
TEST_LOG = TEST_H5.replace('.h5', '.log')

# Grid shape for the 2-case test: (1, 2, 1, 1, 1) = 2 cases
# Covers the first two metallicity values of the full low-metal grid.
TEST_GRID_SHAPE = (1, 2, 1, 1, 1)


def get_test_gridvals():
    """Minimal 2-case grid: first two metallicities, one value for everything else."""
    return (
        np.array([2.0]),                    # rad (Earth radii)
        np.linspace(0.5, 0.875, 2),         # metal (first 2 of 6 low-metal values)
        np.array([50.0]),                   # tint (K)
        np.array([5.0]),                    # semi_major (AU)
        np.array([0.01]),                   # ctoO
    )


def check_output():
    """Read the test h5 and verify the 2 cases have sensible PT results."""
    import h5py
    if not Path(TEST_H5).exists():
        print(f'ERROR: {TEST_H5} not found — run the MPI test first.')
        return

    with h5py.File(TEST_H5, 'r') as f:
        completed = f['completed'][:]
        status    = f['results/status'][...]
        pressure  = f['results/pressure'][...]
        temp      = f['results/temperature'][...]
        converged = f['results/converged'][...]

    n_cases = int(np.prod(TEST_GRID_SHAPE))
    metal_vals = np.linspace(0.5, 0.875, 2)

    print('=== PICASO MPI test output check ===')
    for flat_idx in range(n_cases):
        nd = np.unravel_index(flat_idx, TEST_GRID_SHAPE)
        s   = status[nd].flat[0]
        p   = pressure[nd]
        t   = temp[nd]
        c   = converged[nd].flat[0]
        metal = metal_vals[nd[1]]
        print(f'\n  flat index {flat_idx}  nd_index {nd}  (log10Z={metal:.3f})')
        print(f'  completed  : {bool(completed[flat_idx])}')
        print(f'  status     : {s}')
        print(f'  converged  : {c}')
        print(f'  n_layers   : {len(p)}  (expected 91)')
        if len(p) > 0 and not np.isnan(p).all():
            print(f'  P range    : {np.nanmin(p):.3e} – {np.nanmax(p):.3e} bar')
            print(f'  T range    : {np.nanmin(t):.1f} – {np.nanmax(t):.1f} K')
        else:
            print('  WARNING: pressure array is NaN — case failed or errored')


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
    from PICASO_Climate_grid_121625 import PICASO_climate_model, setup_rank_debug_logging

    setup_rank_debug_logging()
    MPI.COMM_WORLD.Barrier()

    gridutils.make_grid(
        model_func=PICASO_climate_model,
        gridvals=get_test_gridvals(),
        filename=TEST_H5,
        progress_filename=TEST_LOG,
    )
