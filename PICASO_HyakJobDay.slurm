#!/bin/bash

#SBATCH --job-name=PICASO_grid

#SBATCH --account=vsm
#SBATCH --partition=compute-hugemem
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --mem=700G
#SBATCH --time=12:00:00

#SBATCH --export=all
#SBATCH --output=picaso_grid_%j.stdout
#SBATCH --error=picaso_grid_%j.stderr

export PYTHONFAULTHANDLER=1

# Run the script 
mpiexec -l -np 80 python -u PICASO_Climate_grid_121625.py