#!/bin/bash

#SBATCH --job-name=PICASO_grid

#SBATCH --account=vsm
#SBATCH --partition=compute-hugemem
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --mem=700G
#SBATCH --time=7-00:00:00

#SBATCH --export=all
#SBATCH --output=MiniNepGrid_Scripts/Picaso_stderr_stdout_files/picaso_grid_%j.stdout
#SBATCH --error=MiniNepGrid_Scripts/Picaso_stderr_stdout_files/picaso_grid_%j.stderr

export PYTHONFAULTHANDLER=1

# Run the script
mpiexec -l -np 80 python -u MiniNepGrid_Scripts/PICASO_Climate_grid_121625.py