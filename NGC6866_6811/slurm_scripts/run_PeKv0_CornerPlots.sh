#!/bin/bash
#SBATCH --job-name=corner_plots
#SBATCH --partition=compute,douglste-laf-lab,unowned
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=logs/corner_%j.out
#SBATCH --error=logs/corner_%j.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thejoker

python PeKv0_CornerPlots.py
