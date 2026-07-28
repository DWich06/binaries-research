#imports
import os
import argparse
import numpy as np
import astropy.units as u
import pymc as pm
import thejoker as tj
import thejoker.units as xu
from astropy.table import QTable, vstack
from astropy.time import Time

#scratch working directory
workpath = f"/scratch/david_jitter_run_{os.environ.get('SLURM_ARRAY_JOB_ID', 'local')}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}/"

os.makedirs(workpath, exist_ok=True)

#random generator to ensure reproducibility
rnd = np.random.default_rng(seed=42)

new_6866 = QTable.read(f"{workpath}/rcat_ngc6866_v0.fits")
new_6811 = QTable.read(f"{workpath}/rcat_ngc6811_v0.fits")


def RunTheJokerOnePriorJitter(id_num, num_priors):

    star_dir = f"{workpath}/{id_num}"
    os.makedirs(star_dir, exist_ok=True)

    new_ids_6811 = new_6811["GAIAEDR3_ID"]
    new_ids_6866 = new_6866["GAIAEDR3_ID"]

    datamatched6811 = new_6811[id_num == new_ids_6811]
    datamatched6866 = new_6866[id_num == new_ids_6866]

    matched = vstack([datamatched6811, datamatched6866])
    print(f"{id_num}: {len(matched)} RV points")

    if len(matched) == 0:
        print("No RV data for this ID")
        return

    if len(matched) < 3:
        print("Not enough RV data")
        return

    t1 = Time(matched["DATE-OBS"], format="fits", scale="tcb")

    data = tj.RVData(
        t=t1,
        rv=matched["vrad"] * (u.km / u.s),
        rv_err=matched["vrad_err"] * (u.km / u.s),
    )

    print("RV data")

    with pm.Model() as model:
        s = xu.with_unit(
            pm.Normal("s", 0.0, 0.5),
            u.km / u.s
        )

        prior = tj.JokerPrior.default(
            P_min=0.1 * u.day,
            P_max=1e4 * u.day,
            sigma_K0=30 * u.km / u.s,
            sigma_v=100 * u.km / u.s,
            model=model,
            pars={"s": s},
        )

    prior_file = f"{workpath}/prior_samples_400M_jitter.hdf5"
    prior_samples = tj.JokerSamples.read(prior_file)

    joker = tj.TheJoker(prior, rng=rnd)

    joker_samples = joker.rejection_sample(
        data,
        prior_samples,
        max_posterior_samples=256,
        return_logprobs=True,
    )

    print("rejection sample created")
    print(len(joker_samples), "samples")
    print("s values:")
    print(joker_samples["s"][:10])

    mils = num_priors / 1_000_000

    outfile = f"{star_dir}/rejection_samples_{mils:.0f}M_jitter_{id_num}.hdf5"

    joker_samples.write(outfile, overwrite=True)

    print(f"joker samples written to {outfile}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("id", help="star id", type=int)
    parser.add_argument("--prior", help="num of prior samples", type=int, default=200000000)

    args = parser.parse_args()

    RunTheJokerOnePriorJitter(args.id, args.prior)
