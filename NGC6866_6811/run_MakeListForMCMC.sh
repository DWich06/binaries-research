#!/bin/bash
#SBATCH --job-name=mcmcdata
#SBATCH --partition=compute,douglste-laf-lab,unowned
#SBATCH --output=/data/labs/douglste-laf-lab/wichmand/logs/mcmcdata_%j.out
#SBATCH --error=/data/labs/douglste-laf-lab/wichmand/logs/mcmcdata_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thejoker

python MakeListForMCMC.py

