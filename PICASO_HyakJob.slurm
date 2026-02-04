#!/bin/bash

#SBATCH --job-name=PICASO_grid

#SBATCH --account=stf
#SBATCH --partition=cpu-g2-mem2x
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=33
#SBATCH --mem=50G
#SBATCH --time=24:00:00

#SBATCH --export=all
#SBATCH --output=picaso_grid.stdout
#SBATCH --error=picaso_grid.stderr

# Run the script 
mpiexec -np 66 python PICASO_Climate_grid_121625.py