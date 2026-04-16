#!/bin/bash

#SBATCH --job-name=EarthGasBatch

#SBATCH --account=vsm
#SBATCH --partition=compute-hugemem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=14
#SBATCH --mem=500G
#SBATCH --time=24:00:00

#SBATCH --export=all
#SBATCH --output=earthgas_batch_%j.stdout
#SBATCH --error=earthgas_batch_%j.stderr

# --------------------------------------------------------------------------
# Set BATCH_CASE_LIST to the path of your JSON file containing the cases
# to run. Each entry must have a "type" field ("earth" or "gas_planet")
# plus all kwargs for the corresponding function. Set "outputfile" in each
# case so results are saved as pickle files.
#
# See the docstring at the top of EarthGasPlanetBatch_HyakJob.py for the
# full JSON format and an example.
# --------------------------------------------------------------------------
export BATCH_CASE_LIST="batch_cases.json"

# Run the script (nodes * ntasks-per-node = total MPI ranks)
# Rank 0 is the master; all others are workers.
mpiexec -n 14 python EarthGasPlanetBatch_HyakJob.py
