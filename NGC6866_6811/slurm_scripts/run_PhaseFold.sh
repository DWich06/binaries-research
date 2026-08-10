#!/bin/bash
#SBATCH --job-name=plots
#SBATCH --partition=compute,douglste-laf-lab,unowned
#SBATCH --output=/data/labs/douglste-laf-lab/wichmand/logs/PhaseFold/phasefold_%j.out
#SBATCH --error=/data/labs/douglste-laf-lab/wichmand/logs/PhaseFold/phasefold_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thejoker

python PhaseFold.py
