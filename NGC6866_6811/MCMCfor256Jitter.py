import os
import numpy as np
import astropy.units as u
from astropy.table import QTable, vstack
from astropy.time import Time
import thejoker as tj
import thejoker.units as xu
import pymc as pm

workpath = "/data/labs/douglste-laf-lab/wichmand/stardata/200.0M_jitter"

mcmctxt = "/data/labs/douglste-laf-lab/wichmand/stardata/mcmc_for_less256_jitter.txt"

cat_6811 = QTable.read("/data/labs/douglste-laf-lab/wichmand/stardata/rcat_ngc6811_v0.fits")
cat_6866 = QTable.read("/data/labs/douglste-laf-lab/wichmand/stardata/rcat_ngc6866_v0.fits")

rnd = np.random.default_rng(seed=42)


def run_mcmc(star_id):
    star_id = int(star_id)

    star_dir = f"{workpath}/{star_id}"

    joker_file = f"{star_dir}/rejection_samples_200M_jitter_{star_id}.hdf5"

    mcmc_file = f"{star_dir}/rejection_samples_MCMC_adapt_full_200M_jitter_{star_id}.hdf5"

    if os.path.exists(mcmc_file):
        print(f"{star_id}: jitter MCMC already exists, skipping")
        return

    if not os.path.exists(joker_file):
        print(f"{star_id}: Joker file missing, skipping")
        return

    joker_samples = tj.JokerSamples.read(joker_file)

    print(f"{star_id}: {len(joker_samples)} Joker samples")

    if len(joker_samples) >= 256:
        print(f"{star_id}: has >=256 samples, skipping")
        return

    matched_6811 = cat_6811[cat_6811["GAIAEDR3_ID"] == star_id]
    matched_6866 = cat_6866[cat_6866["GAIAEDR3_ID"] == star_id]

    matched = vstack([matched_6811, matched_6866])

    if len(matched) < 3:
        print(f"{star_id}: fewer than 3 RV measurements, skipping")
        return

    data = tj.RVData(
        t=Time(matched["DATE-OBS"], format="fits", scale="tcb"),
        rv=matched["vrad"] * u.km / u.s,
        rv_err=matched["vrad_err"] * u.km / u.s,
    )

    with pm.Model() as model:
        s = xu.with_unit(
            pm.Normal("s", 0.0, 0.5),
            u.km / u.s
        )

        prior = tj.JokerPrior.default(
            P_min=2 * u.day,
            P_max=1e3 * u.day,
            sigma_K0=30 * u.km / u.s,
            sigma_v=100 * u.km / u.s,
            model=model,
            pars={"s": s},
        )

    joker = tj.TheJoker(prior, rng=rnd)

    print(f"{star_id}: starting MCMC")

    with prior.model:
        mcmc_init = joker.setup_mcmc(data, joker_samples)

        trace = pm.sample(
            tune=500,
            draws=500,
            chains=2,
            cores=1,
            start=mcmc_init,
            init="adapt_full",
        )

    mcmc_samples = tj.JokerSamples.from_inference_data(prior, trace, data)
    mcmc_samples.wrap_K()
    mcmc_samples.write(mcmc_file, overwrite=True)

    print(f"{star_id}: wrote {mcmc_file}")


jobid = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))

star_ids = np.loadtxt(mcmctxt, dtype=str, usecols=0)

star_id = str(star_ids[jobid])

print(f"SLURM_ARRAY_TASK_ID = {jobid}")
print(f"Running {star_id}")

run_mcmc(star_id)
