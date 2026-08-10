#!/bin/bash
#SBATCH --job-name=plots
#SBATCH --partition=compute,douglste-laf-lab,unowned
#SBATCH --output=/data/labs/douglste-laf-lab/wichmand/logs/PePlots/peplots_%A_%a.out
#SBATCH --error=/data/labs/douglste-laf-lab/wichmand/logs/PePlots/peplots_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thejoker

python PePlotsPresentation.py
