#!/bin/bash
#SBATCH --job-name=bimodalcheck
#SBATCH --output=bimodal_%j.log
#SBATCH --partition=compute,douglste-laf-lab,unowned
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thejoker

python bimodalcheck.py

