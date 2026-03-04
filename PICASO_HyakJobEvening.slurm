#!/bin/bash

#SBATCH --job-name=PICASO_grid

#SBATCH --account=vsm
#SBATCH --partition=compute-hugemem
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=40
#SBATCH --mem=700G
#SBATCH --time=8:00:00

#SBATCH --export=all
#SBATCH --output=picaso_grid.stdout
#SBATCH --error=picaso_grid.stderr

# Run the script 
mpiexec -np 160 python PICASO_Climate_grid_121625.py