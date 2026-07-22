#!/bin/bash

#SBATCH --job-name=Photochem_grid

#SBATCH --account=vsm
#SBATCH --partition=cpu-g2-mem2x
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=40
#SBATCH --mem=150G
#SBATCH --time=10:00:00

#SBATCH --export=all
#SBATCH --output=MiniNepGrid_Scripts/Photochem_stderr_stdout_files/photochem_grid_%j.stdout
#SBATCH --error=MiniNepGrid_Scripts/Photochem_stderr_stdout_files/photochem_grid_%j.stderr

# Run the script
mpiexec -n 160 python MiniNepGrid_Scripts/Photochem_grid_121625.py