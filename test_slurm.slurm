#!/bin/bash

#SBATCH --job-name=test

#SBATCH --account=stf
#SBATCH --partition=cpu-g2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --mem=10G
#SBATCH --time=30:00

#SBATCH --export=all
#SBATCH --output=test.stdout
#SBATCH --error=test.stderr

# Run the script 
mpiexec -np 80 python test.py