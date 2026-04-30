#!/usr/bin/env python
"""
Run a single case from batch_cases_finalpap.json without MPI.
Usage: python run_single_case.py [case_index]  (default: 0)
"""
import json
import sys
import importlib.util
import tracemalloc
from pathlib import Path
import os

# --- Paths ---
current_directory = Path.cwd()
os.environ['picaso_refdata'] = str(current_directory / "Installation&Setup_Instructions/picasofiles/reference")
os.environ['PYSYN_CDBS']     = str(current_directory / "Installation&Setup_Instructions/picasofiles/grp/redcat/trds")

# --- Load case from JSON first so we know the type before importing FPF ---
case_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
case_file  = current_directory / "batch_cases_finalpap.json"

with open(case_file, 'r') as f:
    cases = json.load(f)

case = cases[case_index].copy()
case_type = case.pop('type')
print(f"Running case {case_index} (type='{case_type}'): {case.get('outputfile', 'no outputfile')}")

# Tell FPF which opacity to load (earth=[0.3,2.5] only; gas_planet=[0.1,2.5] only)
os.environ['RUN_CASE_TYPE'] = case_type

# --- Load FinalPaperFigures&RLSChanges (& in filename requires importlib) ---
print("Loading FinalPaperFigures&RLSChanges (includes opacity load) ...")
tracemalloc.start()
_fpf_path = current_directory / "FinalPaperFigures&RLSChanges.py"
_spec = importlib.util.spec_from_file_location("FPF", _fpf_path)
FPF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FPF)
_cur, _peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"Memory after import — current: {_cur/1e9:.2f} GB  |  peak: {_peak/1e9:.2f} GB")

# --- Dispatch ---
if case_type == 'earth':
    result = FPF.earth_spectrum(**case)
elif case_type == 'gas_planet':
    result = FPF.reflected_spectrum_gas_planet_Sun(**case)
else:
    raise ValueError(f"Unknown case type: {case_type}")

print("Done.")

