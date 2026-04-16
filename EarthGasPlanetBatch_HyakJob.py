#!/usr/bin/env python
"""
MPI batch runner for earth_spectrum and reflected_spectrum_planet_Sun_adjusted_forcldfrac_molecules.

Cases are loaded from a JSON file (set via BATCH_CASE_LIST env var, default: batch_cases.json).
Each entry in the JSON list is a dict with a required "type" key:

  "earth"      → calls earth_spectrum(**kwargs)
  "gas_planet" → calls reflected_spectrum_gas_planet_Sun(**kwargs)

All other keys are passed directly as kwargs to the function. Set "outputfile" in each
case dict so results are saved as pickle files — workers do not return values to master.

Example batch_cases.json:
[
  {
    "type": "earth",
    "opacity_path": "/path/to/opacities.db",
    "df_mol_earth": {"N2": 0.945, "CO2": 0.05, "CO": 0.0005, "CH4": 0.005, "H2O": 0.003},
    "phase": 0.0,
    "cloud_frac": 0.5,
    "no_rayleigh": false,
    "outputfile": "ArcheanEarth"
  },
  {
    "type": "gas_planet",
    "rad_plan": 2.61,
    "planet_metal": 3.5,
    "tint": 155,
    "semi_major": 0.1,
    "ctoO": 0.01,
    "Kzz": 5,
    "phase_angle": 0.0,
    "Photochem_file": "results/Photochem_1D_fv.h5",
    "cloud_frac": 0.5,
    "no_rayleigh": false,
    "outputfile": "2.61_3.5_155_0.1_0.01_5"
  }
]
"""

import json
import os
import sys
import traceback
import importlib.util
from pathlib import Path

from mpi4py import MPI
import dill

# ── Environment setup ─────────────────────────────────────────────────────────
current_directory = Path.cwd()
references_directory_path = "Installation&Setup_Instructions/picasofiles/reference"
PYSYN_directory_path = "Installation&Setup_Instructions/picasofiles/grp/redcat/trds"

os.environ['picaso_refdata'] = os.path.join(current_directory, references_directory_path)
os.environ['PYSYN_CDBS'] = os.path.join(current_directory, PYSYN_directory_path)

# ── Import FinalPaperFigures&RLSChanges (& in filename requires importlib) ────
_fpf_path = current_directory / "FinalPaperFigures&RLSChanges.py"
_spec = importlib.util.spec_from_file_location("FPF", _fpf_path)
FPF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FPF)

# ── MPI setup ─────────────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

MASTER   = 0
TAG_WORK = 1
TAG_DONE = 2
TAG_STOP = 3

# ── Case file ─────────────────────────────────────────────────────────────────
CASE_FILE = os.environ.get('BATCH_CASE_LIST', 'batch_cases.json')


def load_cases():
    with open(CASE_FILE, 'r') as f:
        return json.load(f)


# ── Dispatch a single case ────────────────────────────────────────────────────
def run_case(case_dict):
    case = case_dict.copy()
    case_type = case.pop('type')

    if case_type == 'earth':
        FPF.earth_spectrum(**case)

    elif case_type == 'gas_planet':
        FPF.reflected_spectrum_gas_planet_Sun(**case)

    else:
        raise ValueError(f"Unknown case type '{case_type}'. Must be 'earth' or 'gas_planet'.")


# ── Master process ────────────────────────────────────────────────────────────
def master():
    cases = load_cases()
    n_cases  = len(cases)
    n_workers = size - 1
    print(f"[Master] {n_cases} cases, {n_workers} workers", flush=True)

    case_idx = 0
    active   = 0

    # Seed each worker with its first case
    for worker in range(1, min(n_workers + 1, n_cases + 1)):
        payload = dill.dumps((case_idx, cases[case_idx]))
        comm.send(payload, dest=worker, tag=TAG_WORK)
        case_idx += 1
        active   += 1

    # As workers finish, hand out remaining cases
    while active > 0:
        status = MPI.Status()
        raw    = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_DONE, status=status)
        worker = status.Get_source()
        done_idx, ok, err = dill.loads(raw)

        if ok:
            print(f"[Master] Case {done_idx} complete (worker {worker})", flush=True)
        else:
            print(f"[Master] Case {done_idx} FAILED (worker {worker}):\n{err}", flush=True)

        active -= 1

        if case_idx < n_cases:
            payload = dill.dumps((case_idx, cases[case_idx]))
            comm.send(payload, dest=worker, tag=TAG_WORK)
            case_idx += 1
            active   += 1

    # Signal all workers to stop
    for worker in range(1, size):
        comm.send(None, dest=worker, tag=TAG_STOP)

    print("[Master] All cases complete.", flush=True)


# ── Worker process ────────────────────────────────────────────────────────────
def worker():
    while True:
        status = MPI.Status()
        raw    = comm.recv(source=MASTER, tag=MPI.ANY_TAG, status=status)
        tag    = status.Get_tag()

        if tag == TAG_STOP:
            break

        case_idx, case_dict = dill.loads(raw)
        label = case_dict.get('outputfile', f'case_{case_idx}')
        print(f"[Worker {rank}] Starting case {case_idx}: {label}", flush=True)

        try:
            run_case(case_dict)
            result = dill.dumps((case_idx, True, None))
            print(f"[Worker {rank}] Finished case {case_idx}: {label}", flush=True)
        except Exception:
            result = dill.dumps((case_idx, False, traceback.format_exc()))

        comm.send(result, dest=MASTER, tag=TAG_DONE)


# ── Entry point ───────────────────────────────────────────────────────────────
if rank == MASTER:
    master()
else:
    worker()
