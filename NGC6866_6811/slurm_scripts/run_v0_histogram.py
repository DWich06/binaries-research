#!/bin/bash
#SBATCH --job-name=v0
#SBATCH --partition=compute,douglste-laf-lab,unowned
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=gridspecplots_%j.out
#SBATCH --error=gridspecplots_%j.err

cd ../plotting/plot_recreations

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thejoker

python v0_histogram.py
