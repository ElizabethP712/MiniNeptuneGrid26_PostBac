#!/bin/bash

#SBATCH --job-name=PICASO_grid

#SBATCH --account=stf
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=50G
#SBATCH --time=24:00:00

#SBATCH --export=all
#SBATCH --output=picaso_grid.stdout
#SBATCH --error=picaso_grid.stderr

# Run the script 
mpiexec -np 40 python PICASO_Climate_grid_121625.py