#!/usr/bin/env python
"""
Run a single case from a batch JSON file without MPI.
Usage: python ReflectedSpectra_Scripts/run_single_case.py [case_index] [--file path/to/batch.json]
  case_index  : integer index into the JSON array (default: 0)
  --file      : path to batch JSON, relative to project root
                (default: ReflectedSpectra_Scripts/batch_cases_paper_figures.json)
"""
import json
import sys
import argparse
import importlib.util
import tracemalloc
from pathlib import Path
import os

# --- Paths ---
current_directory = Path.cwd()
os.environ['picaso_refdata'] = str(current_directory / "Installation_Setup_Instructions/picasofiles/reference")
os.environ['PYSYN_CDBS']     = str(current_directory / "Installation_Setup_Instructions/picasofiles/grp/redcat/trds")

# --- Parse args ---
parser = argparse.ArgumentParser()
parser.add_argument('case_index', type=int, nargs='?', default=0)
parser.add_argument('--file', default='ReflectedSpectra_Scripts/batch_cases_paper_figures.json',
                    help='Batch JSON path relative to project root')
args = parser.parse_args()
case_index = args.case_index
case_file  = current_directory / args.file

with open(case_file, 'r') as f:
    cases = json.load(f)

case = cases[case_index].copy()
case_type = case.pop('type')
case.pop('_section', None)  # metadata field — not passed to function
case.pop('_note', None)     # metadata field — not passed to function
print(f"Running case {case_index} (type='{case_type}'): {case.get('outputfile', 'no outputfile')}")

# Tell FPF which opacity to load (earth=[0.3,2.5] only; gas_planet=[0.1,2.5] only)
os.environ['RUN_CASE_TYPE'] = case_type

# --- Load FinalPaperFigures_RLSChanges ---
print("Loading FinalPaperFigures_RLSChanges (includes opacity load) ...")
tracemalloc.start()
_fpf_path = current_directory / "ReflectedSpectra_Scripts/FinalPaperFigures_RLSChanges.py"
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

