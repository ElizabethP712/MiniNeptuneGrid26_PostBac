#!/bin/bash

#SBATCH --job-name=Photochem_grid

#SBATCH --account=stf
#SBATCH --partition=compute
#SBATCH --nodes=7
#SBATCH --ntasks-per-node=40
#SBATCH --mem=150G
#SBATCH --time=24:00:00

#SBATCH --export=all
#SBATCH --output=photochem_grid_%j.stdout
#SBATCH --error=photochem_grid_%j.stderr

# Run the script 
mpiexec -n 280 python Photochem_grid_121625.py