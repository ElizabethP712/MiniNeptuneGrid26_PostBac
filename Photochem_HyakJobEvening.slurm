#!/bin/bash

#SBATCH --job-name=Photochem_grid

#SBATCH --account=vsm
#SBATCH --partition=compute-hugemem
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=40
#SBATCH --mem=150G
#SBATCH --time=12:00:00

#SBATCH --export=all
#SBATCH --output=photochem_grid_%j.stdout
#SBATCH --error=photochem_grid_%j.stderr

# Run the script 
mpiexec -n 160 python Photochem_grid_121625.py