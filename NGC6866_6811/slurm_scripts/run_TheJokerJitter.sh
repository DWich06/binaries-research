#!/bin/bash
#SBATCH --job-name=jitter
#SBATCH --partition=compute,douglste-laf-lab,unowned
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --output=/data/labs/douglste-laf-lab/wichmand/logs/jitter_%A_%a.out
#SBATCH --error=/data/labs/douglste-laf-lab/wichmand/logs/jitter_%A_%a.err
#SBATCH --array=0-62%20

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thejoker

STAR_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" /data/labs/douglste-laf-lab/wichmand/stardata/mcmc_for_less256_jitter.txt)

SCRATCH=/scratch/david_jitter_run_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH

DATA=/data/labs/douglste-laf-lab/wichmand/stardata

export TMPDIR=$SCRATCH

cp $DATA/rcat_ngc6866_v0.fits $SCRATCH/
cp $DATA/rcat_ngc6811_v0.fits $SCRATCH/
cp $DATA/400.0M_jitter/prior_samples_400M_jitter.hdf5 $SCRATCH/

python TheJokerJitter.py $STAR_ID --prior 400000000

mv $SCRATCH/$STAR_ID \
   $DATA/400.0M_jitter/

rm $SCRATCH/rcat_ngc6866_v0.fits $SCRATCH/rcat_ngc6811_v0.fits  $SCRATCH/prior_samples_400M_jitter.hdf5

rmdir $SCRATCH
