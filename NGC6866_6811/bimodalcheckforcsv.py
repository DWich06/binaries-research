import os
import traceback
import astropy.units as u
import numpy as np
import thejoker as tj
from astropy.table import QTable, Table, vstack
from astropy.time import Time


stardata = "/data/labs/douglste-laf-lab/wichmand/stardata"

idlist = QTable.read(
    "/data2/labs/douglste-laf-lab/mathewea/"
    "Summer-Research/GAIADR3_IDs.csv"
)

cat_6811 = QTable.read(f"{stardata}/rcat_ngc6811_v0.fits")
cat_6866 = QTable.read(f"{stardata}/rcat_ngc6866_v0.fits")


runs = {
    "200": {
        "directory": f"{stardata}/200.0M_new",
        "filename": "rejection_samples_200.0M_{id}_new.hdf5",
        "output": "bimodalcheck_200_forcsv.csv",
    },

    "200_mcmc": {
        "directory": f"{stardata}/200.0M_new",
        "filename": "rejection_samples_MCMC_200.0M_{id}_new.hdf5",
        "output": "bimodalcheck_200_mcmc_forcsv.csv",
    },

    "200_mcmc_adapt_full": {
        "directory": f"{stardata}/200.0M_new",
        "filename": "rejection_samples_MCMC_adapt_full_200.0M_{id}_new.hdf5",
        "output": "bimodalcheck_200_mcmc_adapt_full_forcsv.csv",
    },

    "200_jitter": {
        "directory": f"{stardata}/200.0M_jitter",
        "filename": "rejection_samples_200M_jitter_{id}.hdf5",
        "output": "bimodalcheck_200_jitter_forcsv.csv",
    },

    "200_jitter_mcmc": {
        "directory": f"{stardata}/200.0M_jitter",
        "filename": (
            "rejection_samples_MCMC_adapt_full_"
            "200M_jitter_{id}.hdf5"
        ),
        "output": "bimodalcheck_200_jitter_mcmc_forcsv.csv",
    },

    "400_jitter": {
        "directory": f"{stardata}/400.0M_jitter",
        "filename": "rejection_samples_400M_jitter_{id}.hdf5",
        "output": "bimodalcheck_400_jitter_forcsv.csv",
    },
}


def get_cluster(idnum):
    
    if np.any(cat_6811["GAIAEDR3_ID"] == idnum):
        return "NGC6811"

    if np.any(cat_6866["GAIAEDR3_ID"] == idnum):
        return "NGC6866"

    return "Unknown"


def get_rv_data(idnum):
    matched_6811 = cat_6811[
        cat_6811["GAIAEDR3_ID"] == idnum
    ]

    matched_6866 = cat_6866[
        cat_6866["GAIAEDR3_ID"] == idnum
    ]

    matched = vstack([matched_6811, matched_6866])

    if len(matched) < 3:
        return matched, None

    times = Time(
        matched["DATE-OBS"],
        format="fits",
        scale="tcb",
    )

    data = tj.RVData(
        t=times,
        rv=matched["vrad"] * (u.kilometer / u.second),
        rv_err=matched["vrad_err"] * (u.kilometer / u.second),
    )

    return matched, data


def fix_ln_prior(joker_samples):
    if "ln_prior" not in joker_samples.tbl.colnames:
        return

    if joker_samples.tbl["ln_prior"].dtype.names is not None:
        joker_samples.tbl["ln_prior"] = (
            joker_samples.tbl["ln_prior"]["ln_prior"]
        )


def classify_modality(joker_samples, data):
    try:
        if tj.is_P_unimodal(joker_samples, data):
            return "Unimodal"

    except Exception as exc:
        print(f"is_P_unimodal failed: {exc}")
        traceback.print_exc()
        return "Other"

    if len(joker_samples) < 10:
        return "Too Few Samples"

    try:
        is_bimodal, _, _ = tj.is_P_Kmodal(
            joker_samples,
            data,
            n_clusters=2,
        )

        if is_bimodal:
            return "Bimodal"

        return "Other"

    except Exception as exc:
        print(f"is_P_Kmodal failed: {exc}")
        traceback.print_exc()
        return "Other"


def process_run(run_name, run_info):
    print()
    print("=" * 70)
    print(f"Processing run: {run_name}")
    print("=" * 70)

    rows = []

    for idnum in idlist["GAIAEDR3_ID"]:
        idnum = int(idnum)

        matched, data = get_rv_data(idnum)
        num_rvs = len(matched)

        if data is None:
            print(f"{idnum}: fewer than 3 RVs, skipping")
            continue

        filepath = os.path.join(
            run_info["directory"],
            str(idnum),
            run_info["filename"].format(id=idnum),
        )

        if not os.path.exists(filepath):
            print(f"{idnum}: file does not exist")
            continue

        try:
            joker_samples = tj.JokerSamples.read(filepath)
            fix_ln_prior(joker_samples)

            sample_count = len(joker_samples)

            modality = classify_modality(
                joker_samples,
                data,
            )

            rows.append(
                {
                    "GAIAEDR3_ID": idnum,
                    "cluster": get_cluster(idnum),
                    "sample_count": sample_count,
                    "modality": modality,
                }
            )

            print(
                f"{idnum}: "
                f"samples={sample_count}, "
                f"modality={modality}"
            )

        except Exception as exc:
            print(f"{idnum}: failed to process file")
            print(f"File: {filepath}")
            print(f"Error: {exc}")
            traceback.print_exc()

            rows.append(
                {
                    "GAIAEDR3_ID": idnum,
                    "cluster": get_cluster(idnum),
                    "sample_count": 0,
                    "modality": "Other",
                }
            )

    output_table = Table(
        rows=rows,
        names=[
            "GAIAEDR3_ID",
            "cluster",
            "sample_count",
            "modality",
        ],
    )

    output_table.write(
        run_info["output"],
        format="csv",
        overwrite=True,
    )

    print()
    print(f"Wrote {run_info['output']}")
    print(f"Number of stars: {len(output_table)}")


for run_name, run_info in runs.items():
    process_run(run_name, run_info)

print()
print("Finished all runs.")
