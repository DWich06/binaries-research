#!/bin/bash
#SBATCH --job-name=RVCurves
#SBATCH --partition=compute,douglste-laf-lab,unowned
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --output=/data/labs/douglste-laf-lab/wichmand/logs/RVCurves/RVCurves_newrun/RVCurves_%A_%a.out
#SBATCH --error=/data/labs/douglste-laf-lab/wichmand/logs/RVCurves/RVCurves_newrun/RVCurves_%A_%a.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thejoker

python RVPlotsPresentation.py
