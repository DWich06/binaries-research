#!/bin/bash
#SBATCH --job-name=priorjitter
#SBATCH --partition=compute,douglste-laf-lab,unowned
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=400G
#SBATCH --output=/data/labs/douglste-laf-lab/wichmand/logs/priorjitter_%j.out
#SBATCH --error=/data/labs/douglste-laf-lab/wichmand/logs/priorjitter_%j.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thejoker

mkdir -p /scratch/david_oneprior_jitter

python OnePriorJitter.py --prior 400000000

mv /scratch/david_oneprior_jitter/prior_samples_*_jitter.hdf5 \
   /data/labs/douglste-laf-lab/wichmand/stardata/

rmdir /scratch/david_oneprior_jitter
