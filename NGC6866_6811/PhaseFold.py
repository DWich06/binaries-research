import os
import numpy as np
import matplotlib.pyplot as plt
import thejoker as tj
import astropy.units as u
from astropy.table import QTable, Table, vstack
from astropy.time import Time


workpath = "/data/labs/douglste-laf-lab/wichmand"

samplepath = f"{workpath}/stardata/200.0M_new"
bimodal_path = f"{workpath}/stardata/bimodalcheck_200M_phase.csv"

outdir = f"{workpath}/plots/PhaseFold"
os.makedirs(outdir, exist_ok=True)


print("Reading 6811", flush=True)
new_6811 = QTable.read("/data2/labs/douglste-laf-lab/mathewea/rcat_ngc6811_v0.fits")
print("Finished 6811", flush=True)

print("Reading 6866", flush=True)
new_6866 = QTable.read("/data2/labs/douglste-laf-lab/mathewea/rcat_ngc6866_v0.fits")
print("Finished 6866", flush=True)


print("Reading bimodal table", flush=True)
modality_data = Table.read(bimodal_path, format="csv")

id_list = modality_data["id"][
    modality_data["unimodal"] == 1
]

id_list = sorted(np.array(id_list, dtype=np.int64))

print(f"Unimodal ID list has {len(id_list)} IDs", flush=True)

def plot_phase_fold(id_num):

    print(f"Plotting {id_num}", flush=True)

    rej_path = f"{samplepath}/{id_num}/rejection_samples_200.0M_{id_num}_new.hdf5"
    mcmc_path = f"{samplepath}/{id_num}/rejection_samples_MCMC_200.0M_{id_num}_new.hdf5"

    print("Checking to see if sample files exist", flush=True)

    if os.path.exists(mcmc_path):
        print("Using MCMC samples", flush=True)
        joker_samples = tj.JokerSamples.read(mcmc_path)
        filetype = "MCMC"

    elif os.path.exists(rej_path):
        print("Using rejection samples", flush=True)
        joker_samples = tj.JokerSamples.read(rej_path)
        filetype = "rejection"

    else:
        print("No Joker samples", flush=True)
        return

    print("Loaded Joker samples", flush=True)
    print(f"N samples = {len(joker_samples)}", flush=True)

    if len(joker_samples) == 0:
        print("No samples in file", flush=True)
        return

    matched = vstack([
        new_6811[new_6811["GAIAEDR3_ID"] == id_num],
        new_6866[new_6866["GAIAEDR3_ID"] == id_num]
    ])

    if len(matched) < 3:
        print("Less than 3 data points", flush=True)
        return

    print("Building Time", flush=True)

    t = Time(matched["DATE-OBS"], format="fits", scale="tcb")

    print("Building RV Data", flush=True)

    data = tj.RVData(
        t=t,
        rv=matched["vrad"] * (u.km / u.s),
        rv_err=matched["vrad_err"] * (u.km / u.s),
    )

    print("Creating phase fold plot", flush=True)

    sample = tj.MAP_sample(joker_samples)

    fig = tj.plot_phase_fold(
        sample,
        data=data,
    )

    fig.axes[0].set_title(f"ID: {id_num}")

    print(f"Saving {id_num}", flush=True)

    fig.savefig(
        f"{outdir}/PhaseFold_{id_num}_{filetype}.png",
        dpi=200,
        bbox_inches="tight"
    )

    print(f"Saved {id_num}", flush=True)

    plt.close("all")

    del joker_samples
    del matched
    del data
    del t
    del fig


for id_num in id_list:
    plot_phase_fold(id_num)

Print("Done")
