import os
import glob
import thejoker as tj

workpath = "/data/labs/douglste-laf-lab/wichmand/stardata/200.0M_new"

outname = "/data/labs/douglste-laf-lab/wichmand/stardata/mcmc_for_less256.txt"

candidates = []

for rej_file in glob.glob(
    f"{workpath}/*/rejection_samples_200.0M_*_new.hdf5"
):
    star_id = os.path.basename(os.path.dirname(rej_file))

    adapt_full_file = (
        f"{workpath}/{star_id}/"
        f"rejection_samples_MCMC_adapt_full_200.0M_{star_id}_new.hdf5"
    )

    if os.path.exists(adapt_full_file):
        continue

    samples = tj.JokerSamples.read(rej_file)
    n = len(samples)

    if n < 256:
        candidates.append((star_id, n))

candidates.sort(key=lambda x: x[1])

with open(outname, "w") as f:
    for star_id, n in candidates:
        f.write(f"{star_id} {n}\n")

print(f"Wrote {len(candidates)} candidates to {outname}")
