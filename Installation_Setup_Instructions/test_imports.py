import sys

failures = []

def try_import(label, code):
    try:
        exec(code, {})
        print(f"  OK  {label}")
    except Exception as e:
        print(f"  FAIL {label}: {e}")
        failures.append(label)

print("Testing imports...\n")

try_import("matplotlib.pyplot", "import matplotlib.pyplot as plt")
try_import("numpy", "import numpy as np")
try_import("h5py", "import h5py")
try_import("copy", "import copy")
try_import("pandas", "import pandas as pd")
try_import("scipy.optimize", "from scipy import optimize")
try_import("pickle", "import pickle")
try_import("itertools.cycle", "from itertools import cycle")
try_import("matplotlib.colors", "import matplotlib.colors as mcolors")
try_import("astropy.units", "import astropy.units as u")
try_import("astropy.constants", "import astropy.constants as const")
try_import("photochem.utils.stars", "from photochem.utils import stars")
try_import("PICASO_Climate_grid_121625", "import PICASO_Climate_grid_121625")
try_import("Photochem_grid_121625", "import Photochem_grid_121625")
try_import("Reflected_Spectra_grid_13026", "import Reflected_Spectra_grid_13026")
try_import("picaso.photochem.EquilibriumChemistry", "from picaso.photochem import EquilibriumChemistry")
try_import("GraphsKey", "import GraphsKey")
try_import("picaso.justdoit", "import picaso.justdoit as jdi")
try_import("picaso.justplotit", "import picaso.justplotit as jpi")

print(f"\n{len(failures)} failure(s)" if failures else "\nAll imports OK")
if failures:
    print("Failed:", ", ".join(failures))
    sys.exit(1)
