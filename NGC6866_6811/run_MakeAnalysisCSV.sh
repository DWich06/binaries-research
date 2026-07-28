#!/bin/bash
#SBATCH --job-name=spreadsheet
#SBATCH --output=spreadsheet_%j.log
#SBATCH --partition=compute,douglste-laf-lab,unowned
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thejoker

python MakeAnalysisCSV.py
