#!/bin/bash

#SBATCH --job-name=Photochem_grid

#SBATCH --account=stf
#SBATCH --partition=cpu-g2-mem2x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --mem=50G
#SBATCH --time=24:00:00

#SBATCH --export=all
#SBATCH --output=photochem_grid.stdout
#SBATCH --error=photochem_grid.stderr

# Run the script 
mpiexec -np 30 python Photochem_grid_121625.py