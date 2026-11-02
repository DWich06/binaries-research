
import os
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.table import QTable, vstack
from astropy.time import Time

import thejoker as tj
import thejoker.units as xu
import pymc as pm


base = "/data/labs/douglaslab/wichmand/stardata"
workpath = f"{base}/joker_runs/200.0M_jitter"

mcmctxt = f"{base}/analysis_csvs/mcmc_unimodal_offset_ids.txt"

cat_6811 = QTable.read(
    f"{base}/star_catalogs/rcat_ngc6811_v0.fits"
)

cat_6866 = QTable.read(
    f"{base}/star_catalogs/rcat_ngc6866_v0.fits"
)

catalog = vstack([cat_6811, cat_6866])

apogee = pd.read_csv(
    f"{base}/star_catalogs/apogee_radial_velocity.csv",
    dtype={"GAIAEDR3_ID": str}
)

apogee["GAIAEDR3_ID"] = apogee["GAIAEDR3_ID"].str.strip()


def run_mcmc(star_id):

    star_id = str(star_id)
    star_dir = f"{workpath}/{star_id}"

    joker_file = (
        f"{star_dir}/"
        f"rejection_samples_200M_jitter_offset_{star_id}.hdf5"
    )

    mcmc_file = (
        f"{star_dir}/"
        f"rejection_samples_MCMC_adapt_full_200M_jitter_offset_{star_id}.hdf5"
    )

    if os.path.exists(mcmc_file):
        print(f"{star_id}: MCMC already exists, skipping", flush=True)
        return

    if not os.path.exists(joker_file):
        print(f"{star_id}: rejection file missing", flush=True)
        return

    joker_samples = tj.JokerSamples.read(joker_file)

    print(
        f"{star_id}: {len(joker_samples)} rejection samples",
        flush=True
    )

    if len(joker_samples) == 0:
        return

    # Hectochelle observations
    matched_hect = catalog[
        catalog["GAIAEDR3_ID"] == int(star_id)
    ]

    hect_rv = np.asarray(matched_hect["vrad"], dtype=float)
    hect_err = np.asarray(matched_hect["vrad_err"], dtype=float)

    hect_good = (
        np.isfinite(hect_rv)
        & np.isfinite(hect_err)
        & (hect_rv > -9000)
        & (hect_err > 0)
    )

    matched_hect = matched_hect[hect_good]

    # APOGEE observations
    matched_apogee = apogee[
        apogee["GAIAEDR3_ID"] == star_id
    ]

    apogee_rv = matched_apogee["v_rad"].to_numpy(dtype=float)
    apogee_err = matched_apogee["e_v_rel"].to_numpy(dtype=float)
    apogee_mjd = matched_apogee["mjd"].to_numpy(dtype=float)
    apogee_flags = matched_apogee["spectrum_flags"].to_numpy(
        dtype=np.int64
    )

    # Filters
    apogee_good = (
        np.isfinite(apogee_rv)
        & np.isfinite(apogee_err)
        & np.isfinite(apogee_mjd)
        & (apogee_rv > -9000)
        & (apogee_err > 0)
        & (apogee_flags == 0)
    )

    matched_apogee = matched_apogee[apogee_good]

    print(
        f"Hectochelle: {len(matched_hect)} observations",
        flush=True
    )
    print(
        f"APOGEE: {len(matched_apogee)} observations",
        flush=True
    )

    if len(matched_hect) < 1 or len(matched_apogee) < 1:
        print("Missing usable data from one instrument", flush=True)
        return

    #  Hectochelle
    data_hect = tj.RVData(
        t=Time(
            matched_hect["DATE-OBS"],
            format="fits",
            scale="tcb"
        ),
        rv=np.asarray(
            matched_hect["vrad"], dtype=float
        ) * u.km / u.s,
        rv_err=np.asarray(
            matched_hect["vrad_err"], dtype=float
        ) * u.km / u.s
    )

    # APOGEE
    data_apogee = tj.RVData(
        t=Time(
            matched_apogee["mjd"].to_numpy(dtype=float),
            format="mjd",
            scale="utc"
        ),
        rv=matched_apogee["v_rad"].to_numpy(dtype=float)
           * u.km / u.s,
        rv_err=matched_apogee["e_v_rel"].to_numpy(dtype=float)
               * u.km / u.s
    )

    data = [data_hect, data_apogee]

    # Prior
    with pm.Model() as model:

        s = xu.with_unit(
            pm.Normal("s", 0.0, 0.5),
            u.km / u.s
        )

        dv0_1 = xu.with_unit(
            pm.Normal("dv0_1", 0.0, 10.0),
            u.km / u.s
        )

        prior = tj.JokerPrior.default(
            P_min=0.1 * u.day,
            P_max=1e4 * u.day,
            sigma_K0=30 * u.km / u.s,
            sigma_v=100 * u.km / u.s,
            model=model,
            pars={"s": s},
            v0_offsets=[dv0_1]
        )

    joker = tj.TheJoker(
        prior,
        rng=np.random.default_rng(seed=42)
    )

    with prior.model:

        mcmc_init = joker.setup_mcmc(
            data,
            joker_samples
        )

        trace = pm.sample(
            tune=500,
            draws=500,
            chains=2,
            cores=1,
            start=mcmc_init,
            init="adapt_full"
        )

    mcmc_samples = tj.JokerSamples.from_inference_data(
        prior,
        trace,
        data
    )

    mcmc_samples.wrap_K()

    mcmc_samples.write(
        mcmc_file,
        overwrite=False
    )

    print(
        f"{star_id}: saved {len(mcmc_samples)} MCMC samples",
        flush=True
    )
    print(f"Output: {mcmc_file}", flush=True)


with open(mcmctxt) as f:
    star_ids = [line.strip() for line in f if line.strip()]

jobid = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))

if jobid >= len(star_ids):
    raise IndexError("SLURM array index exceeds ID list")

star_id = star_ids[jobid]

print(f"SLURM_ARRAY_TASK_ID = {jobid}", flush=True)
print(f"Running Gaia ID {star_id}", flush=True)

run_mcmc(star_id)
