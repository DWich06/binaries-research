import os
import numpy as np
import matplotlib.pyplot as plt
import thejoker as tj
import astropy.units as u
from astropy.table import QTable, vstack
from astropy.time import Time


workpath = "/data/labs/douglste-laf-lab/wichmand/stardata/200.0M_new"


print("Reading 6811", flush=True)
new_6811 = QTable.read("/data2/labs/douglste-laf-lab/mathewea/rcat_ngc6811_v0.fits")
print("Finished 6811", flush=True)

print("Reading 6866", flush=True)
new_6866 = QTable.read("/data2/labs/douglste-laf-lab/mathewea/rcat_ngc6866_v0.fits")
print("Finished 6866", flush=True)

id_list = []

for name in os.listdir(workpath):
    fullpath = os.path.join(workpath, name)

    if os.path.isdir(fullpath):
        try:
            id_list.append(np.int64(name))
        except ValueError:
            pass

id_list = sorted(id_list)

print(id_list[:10])

print(f"Built ID list with {len(id_list)} IDs", flush=True)

outdir = "/data/labs/douglste-laf-lab/wichmand/plots/RVCurves"
os.makedirs(outdir, exist_ok=True)

print("Directory:", os.getcwd(), flush=True)

def plot_rvcurves(id_num):

    print(f"Plotting {id_num}", flush=True)

    rej_path = f"{workpath}/{id_num}/rejection_samples_200.0M_{id_num}_new.hdf5"
    mcmc_path = f"{workpath}/{id_num}/rejection_samples_MCMC_200.0M_{id_num}_new.hdf5"

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

    matched = vstack([
        new_6811[new_6811["GAIAEDR3_ID"] == id_num],
        new_6866[new_6866["GAIAEDR3_ID"] == id_num]
    ])

    if len(matched) < 3:
        print("Less than 3 data points", flush=True)
        return

    print("Creating plot", flush=True)

    print("Building Time", flush=True)

    t = Time(matched["DATE-OBS"], format="fits", scale="tcb")

    print("Building RV Data", flush=True)

    data = tj.RVData(
        t=t,
        rv=matched["vrad"] * (u.km / u.s),
        rv_err=matched["vrad_err"] * (u.km / u.s),
    )

    print("Creating figure", flush=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    print("Calling plot_rv_curves", flush=True)

    _ = tj.plot_rv_curves(joker_samples, data=data, ax=ax)
    print(joker_samples, flush=True)
    print(f"N samples = {len(joker_samples)}", flush=True)

   # _ = tj.plot_rv_curves(joker_samples[:1], data=data, ax=ax)

    print("Finished plot_rv_curves", flush=True)

    ax.set_title(f"ID: {id_num}")

    print(f"Saving {id_num}", flush=True)

    fig.savefig(f"{outdir}/RVCurves_{id_num}_{filetype}.png", dpi=200)

    print(f"Saved {id_num}", flush=True)

    plt.close("all")

    print("Saved RV curve", flush=True)

    del joker_samples
    del matched
    del data
    del t
    del fig
    del ax


jobid = int(os.getenv("SLURM_ARRAY_TASK_ID"))
plot_rvcurves(id_list[jobid])
