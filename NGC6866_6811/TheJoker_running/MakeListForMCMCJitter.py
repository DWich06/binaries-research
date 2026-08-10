import os
import thejoker as tj

workpath = "/data/labs/douglste-laf-lab/wichmand/stardata/200.0M_jitter"
outfile = "/data/labs/douglste-laf-lab/wichmand/stardata/mcmc_for_less256_jitter.txt"

star_ids = [
    name for name in os.listdir(workpath)
    if os.path.isdir(os.path.join(workpath, name)) and name.isdigit()
]

need_mcmc = []

for star_id in star_ids:
    joker_file = f"{workpath}/{star_id}/rejection_samples_200M_jitter_{star_id}.hdf5"

    if not os.path.exists(joker_file):
        print(f"{star_id}: missing")
        continue

    samples = tj.JokerSamples.read(joker_file)

    if len(samples) < 256:
        print(f"{star_id}: {len(samples)} samples")
        need_mcmc.append(star_id)

with open(outfile, "w") as f:
    for star_id in need_mcmc:
        f.write(f"{star_id}\n")

print(f"Wrote {len(need_mcmc)} stars to {outfile}")
