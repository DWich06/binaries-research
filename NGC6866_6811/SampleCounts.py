import os
import glob
import thejoker as tj

workpath = "/data2/labs/douglste-laf-lab/mathewea/200.0M_new"

thresholds = [5, 10, 20, 50, 100, 256]
counts = {t: 0 for t in thresholds}

total_no_mcmc = 0

for filename in glob.glob(f"{workpath}/*/rejection_samples_200.0M_*_new.hdf5"):
    star_id = filename.split("/")[-2]

    mcmc_file = (
        f"{workpath}/{star_id}/"
        f"rejection_samples_MCMC_200.0M_{star_id}_new.hdf5"
    )

    if os.path.exists(mcmc_file):
        continue

    samples = tj.JokerSamples.read(filename)
    n = len(samples)

    total_no_mcmc += 1

    for t in thresholds:
        if n < t:
            counts[t] += 1

print(f"Total stars without MCMC: {total_no_mcmc}")

for t in thresholds:
    print(f"Stars without MCMC and fewer than {t} Joker samples: {counts[t]}")

