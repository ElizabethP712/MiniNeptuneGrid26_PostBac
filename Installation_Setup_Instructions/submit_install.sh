#!/bin/bash
#SBATCH --job-name=picaso_install
#SBATCH --account=vsm
#SBATCH --partition=cpu-g2
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=install_%j.out
#SBATCH --error=install_%j.err
#SBATCH --chdir=/gscratch/vsm/elizap/MiniNeptuneGrid26_PostBac/Installation_Setup_Instructions

source /mmfs1/home/elizap77/miniconda3/etc/profile.d/conda.sh

bash "/gscratch/vsm/elizap/MiniNeptuneGrid26_PostBac/Installation_Setup_Instructions/setup_automated.sh"

