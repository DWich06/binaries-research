#!/bin/bash
#SBATCH --job-name=e_P_1star
#SBATCH --output=e_P_1star_%j.log
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thejoker

python e_P_1starnew.py 
