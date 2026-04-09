#!/bin/bash

#SBATCH --job-name=RSM_specific

#SBATCH --account=stf
#SBATCH --partition=compute
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=40
#SBATCH --mem=150G
#SBATCH --time=12:00:00

#SBATCH --export=all
#SBATCH --output=rsm_specific_%j.stdout
#SBATCH --error=rsm_specific_%j.stderr

# --------------------------------------------------------------------------
# Set RSM_CASE_LIST to the path of your .npy, .json, or .csv file containing
# the specific input cases to run.  Each row must have 7 columns:
#   [rad (R_earth), metal (log10 Solar), tint (K), semi_major (AU),
#    ctoO (x Solar), kzz (log10 cm^2/s), phase_angle (radians)]
#
# Leave unset to use the default K2-18b test cases built into the script.
# --------------------------------------------------------------------------
export RSM_CASE_LIST="results/my_specific_cases.npy"

# --------------------------------------------------------------------------
# Optional: set the Photochem results file if different from the default.
# The default inside the script is 'results/Photochem_1D_fv.h5'.
# If you need to override it, edit the Photochem_file argument in
# make_case_RSM() inside Reflected_Spectra_SpecificInputs.py directly.
# --------------------------------------------------------------------------

# Run the script (nodes * ntasks-per-node = total MPI ranks)
mpiexec -n 120 python Reflected_Spectra_SpecificInputs.py
