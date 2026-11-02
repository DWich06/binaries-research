
import os
import argparse
import numpy as np
import astropy.units as u
import pymc as pm
import thejoker as tj
import thejoker.units as xu

from astropy.table import QTable, vstack
from astropy.time import Time


# Paths
base_dir = "/data/labs/douglaslab/wichmand/stardata"

catalog_dir = f"{base_dir}/star_catalogs"

output_dir = f"{base_dir}/joker_runs/200.0M_jitter"

prior_file = (
    f"{base_dir}/joker_runs/200.0M_jitter_offset/"
    "prior_samples_200M_jitter_offset.hdf5"
)

apogee_file = f"{catalog_dir}/apogee_radial_velocity.csv"

hect6811_file = f"{catalog_dir}/rcat_ngc6811_v0.fits"
hect6866_file = f"{catalog_dir}/rcat_ngc6866_v0.fits"


#Load catalogs
new_6811 = QTable.read(hect6811_file)
new_6866 = QTable.read(hect6866_file)

apogee_catalog = QTable.read(
    apogee_file,
    format="ascii.csv"
)


# Run the joker
def RunTheJokerJitterOffset(id_num, num_priors):
    star_dir = f"{output_dir}/{id_num}"

    mils = num_priors / 1_000_000

    outfile = (
        f"{star_dir}/rejection_samples_"
        f"{mils:.0f}M_jitter_offset_{id_num}.hdf5"
    )

    # Do not overwrite previous results
    if os.path.exists(outfile):
        print(f"Output already exists: {outfile}")
        print("Skipping star.")
        return

    matched6811 = new_6811[
        new_6811["GAIAEDR3_ID"] == id_num
    ]

    matched6866 = new_6866[
        new_6866["GAIAEDR3_ID"] == id_num
    ]

    matched_hect = vstack([
        matched6811,
        matched6866
    ])

    print(
        f"{id_num}: {len(matched_hect)} Hectochelle RV points",
        flush=True
    )

    matched_apogee = apogee_catalog[
        apogee_catalog["GAIAEDR3_ID"] == id_num
    ]

    print(
        f"{id_num}: {len(matched_apogee)} APOGEE RV points",
        flush=True
    )

    if len(matched_hect) == 0 or len(matched_apogee) == 0:
        print("Missing observations from one instrument.")
        return

    # Clean data
    hect_rv = np.asarray(
        matched_hect["vrad"],
        dtype=float
    )

    hect_err = np.asarray(
        matched_hect["vrad_err"],
        dtype=float
    )

    hect_good = (
        np.isfinite(hect_rv)
        & np.isfinite(hect_err)
        & (hect_rv > -9000)
        & (hect_err > 0)
    )

    matched_hect = matched_hect[hect_good]

    apogee_rv = np.asarray(
        matched_apogee["v_rad"],
        dtype=float
    )

    apogee_err = np.asarray(
        matched_apogee["e_v_rel"],
        dtype=float
    )

    apogee_mjd = np.asarray(
        matched_apogee["mjd"],
        dtype=float
    )

    apogee_flags = np.asarray(
        matched_apogee["spectrum_flags"],
        dtype=np.int64
    )

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
        f"Valid Hectochelle observations: {len(matched_hect)}",
        flush=True
    )

    print(
        f"Valid APOGEE observations: {len(matched_apogee)}",
        flush=True
    )

    if len(matched_hect) == 0 or len(matched_apogee) == 0:
        print("No usable data from both instruments.")
        print("Skipping star.")
        return

    if len(matched_hect) + len(matched_apogee) < 3:
        print("Fewer than 3 combined RV observations.")
        print("Skipping star.")
        return


    t_hect = Time(
        matched_hect["DATE-OBS"],
        format="fits",
        scale="tcb"
    )

    data_hect = tj.RVData(
        t=t_hect,
        rv=(
            np.asarray(
                matched_hect["vrad"],
                dtype=float
            ) * u.km / u.s
        ),
        rv_err=(
            np.asarray(
                matched_hect["vrad_err"],
                dtype=float
            ) * u.km / u.s
        )
    )

    t_apogee = Time(
        np.asarray(
            matched_apogee["mjd"],
            dtype=float
        ),
        format="mjd",
        scale="utc"
    )

    data_apogee = tj.RVData(
        t=t_apogee,
        rv=(
            np.asarray(
                matched_apogee["v_rad"],
                dtype=float
            ) * u.km / u.s
        ),
        rv_err=(
            np.asarray(
                matched_apogee["e_v_rel"],
            ) * u.km / u.s
        )
    )

    data = [
        data_hect,
        data_apogee
    ]

    print("Combined RV data created.", flush=True)

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

    prior_samples = tj.JokerSamples.read(prior_file)

    print(
        f"Loaded {len(prior_samples)} prior samples.",
        flush=True
    )

    rnd = np.random.default_rng(seed=42)

    joker = tj.TheJoker(
        prior,
        rng=rnd
    )

    print("Starting rejection sampling", flush=True)

    joker_samples = joker.rejection_sample(
        data,
        prior_samples,
        max_posterior_samples=256,
        return_logprobs=True
    )

    print("Rejection sampling completed.", flush=True)

    print(
        f"Accepted posterior samples: {len(joker_samples)}",
        flush=True
    )


    if len(joker_samples) == 0:
        print("No posterior samples accepted.")
        return

    if "dv0_1" not in joker_samples.tbl.colnames:
        raise RuntimeError(
            "Instrumental offset dv0_1 missing from posterior."
        )

    print("First 10 jitter values:")
    print(joker_samples["s"][:10])

    print("First 10 offset values:")
    print(joker_samples["dv0_1"][:10])


    os.makedirs(star_dir, exist_ok=True)

    joker_samples.write(
        outfile,
        overwrite=False
    )

    print(f"Joker samples written to {outfile}", flush=True)

    return



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "id",
        help="Gaia EDR3 star ID",
        type=int
    )

    parser.add_argument(
        "--prior",
        help="Number of prior samples",
        type=int,
        default=200000000
    )

    args = parser.parse_args()

    RunTheJokerJitterOffset(
        args.id,
        args.prior
    )
