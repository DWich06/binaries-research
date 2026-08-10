#!/bin/bash
#SBATCH --job-name=plots
#SBATCH --partition=compute,douglste-laf-lab,unowned
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=logs/gridspecplots_%j.out
#SBATCH --error=logs/gridspecplots_%j.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thejoker

python GridSpecMWE.py
