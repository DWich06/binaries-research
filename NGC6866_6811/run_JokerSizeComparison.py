#!/bin/bash
#SBATCH --job-name=count_samples
#SBATCH --partition=compute,douglste-laf-lab,unowned
#SBATCH --output=/data/labs/douglste-laf-lab/wichmand/logs/count_samples_%j.out
#SBATCH --error=/data/labs/douglste-laf-lab/wichmand/logs/count_samples_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thejoker

python JokerSizeComparison.py
