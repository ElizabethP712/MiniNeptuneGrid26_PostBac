#!/bin/bash

#SBATCH --job-name=PICASO_grid
#SBATCH --account=stf
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=10G
#SBATCH --time=5:00
#SBATCH -o log/%x_%j.out

# Load & Initialize conda & necessary packages
module load miniconda3
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate subneptune

# Run the script 
srun python PICASO_Climate_grid_121625.py

# Deactivate conda
conda deactivate 