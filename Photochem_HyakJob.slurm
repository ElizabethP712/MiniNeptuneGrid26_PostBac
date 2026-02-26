#!/bin/bash

#SBATCH --job-name=Photochem_grid

#SBATCH --account=stf
#SBATCH --partition=cpu-g2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --mem=200G
#SBATCH --time=24:00:00

#SBATCH --export=all
#SBATCH --output=photochem_grid.stdout
#SBATCH --error=photochem_grid.stderr

# Run the script 
mpiexec -np 80 python Photochem_grid_121625.py