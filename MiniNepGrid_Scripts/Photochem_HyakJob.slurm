#!/bin/bash

#SBATCH --job-name=Photochem_grid

#SBATCH --account=stf
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --mem=150G
#SBATCH --time=24:00:00

#SBATCH --export=all
#SBATCH --output=MiniNepGrid_Scripts/Photochem_stderr_stdout_files/photochem_grid_%j.stdout
#SBATCH --error=MiniNepGrid_Scripts/Photochem_stderr_stdout_files/photochem_grid_%j.stderr

# Pre-generate all stellar flux files before MPI workers start
echo "Pre-generating stellar flux files..."
python - <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, 'MiniNepGrid_Scripts')
import PICASO_Climate_grid_121625 as PICASO_Climate_grid
import star_spectrum
import numpy as np

semi_major_values = np.array([0.3, 0.7, 1, 1.5, 2, 3, 4, 5, 6, 8, 10])
for a in semi_major_values:
    Teq = PICASO_Climate_grid.calc_Teq_SUN(distance_AU=a)
    f   = f'sun_flux_file_{Teq}'
    if not Path(f).exists():
        print(f'  Generating {f} ...')
        star_spectrum.solar_spectrum(Teq=Teq, outputfile=f)
    else:
        print(f'  Already exists: {f}')
print('All flux files ready.')
EOF

# Run the script
mpiexec -n 80 python MiniNepGrid_Scripts/Photochem_grid_121625.py