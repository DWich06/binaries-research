#!/bin/bash
#SBATCH --job-name=mcmc
#SBATCH --partition=compute,douglste-laf-lab,unowned
#SBATCH --output=/data/labs/douglste-laf-lab/wichmand/logs/mcmc256/mcmcuni_jitter/mcmc_%A_%a.out
#SBATCH --error=/data/labs/douglste-laf-lab/wichmand/logs/mcmc256/mcmcuni_jitter/mcmc_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --array=0,15,17

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thejoker

python MCMCforUniJitter.py
