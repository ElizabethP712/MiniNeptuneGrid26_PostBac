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

source /mmfs1/home/elizap77/miniconda3/etc/profile.d/conda.sh

bash "/mmfs1/home/elizap77/elizap/MiniNeptuneGrid26_PostBac/Installation_Setup_Instructions/setup_automated.sh"

