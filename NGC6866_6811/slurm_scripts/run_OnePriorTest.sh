#!/bin/bash
#SBATCH --job-name=priortest
#SBATCH --partition=compute,douglste-laf-lab,unowned
#SBATCH --output=/data/labs/douglste-laf-lab/wichmand/logs/priortest_%j.out
#SBATCH --error=/data/labs/douglste-laf-lab/wichmand/logs/priortest_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thejoker

python OnePriorJitter.py
