#!/bin/bash
#SBATCH --job-name=unimodal_check
#SBATCH --output=unimodal_check_%j.log
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thejoker

python unimodalcheck_ella.py
