import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import thejoker as tj

workpath = "/data/labs/douglste-laf-lab/wichmand/stardata/200.0M_jitter"
outdir = "/data/labs/douglste-laf-lab/wichmand/plots/PePlots"

os.makedirs(outdir, exist_ok=True)

star_ids = [
    name for name in os.listdir(workpath)
    if os.path.isdir(os.path.join(workpath, name)) and name.isdigit()
]

star_ids = sorted(star_ids)

print(f"Found {len(star_ids)} stars", flush=True)


def make_pe_plot(star_id):
    star_id = int(star_id)

    rejection_file = (
        f"{workpath}/{star_id}/"
        f"rejection_samples_MCMC_adapt_full_200M_jitter_{star_id}.hdf5"
    )

    mcmc_file = (
        f"{workpath}/{star_id}/"
        f"rejection_samples_MCMC_adapt_full_200M_jitter_{star_id}.hdf5"
    )

   # if os.path.exists(mcmc_file):
    #    sample_file = mcmc_file
     #   filetype = "MCMC_adapt_full"
    if  os.path.exists(rejection_file):
        sample_file = rejection_file
        filetype = "rejection"
    else:
        print(f"{star_id}: no sample file", flush=True)
        return

    try:
        samples = tj.JokerSamples.read(sample_file)
    except Exception as e:
        print(f"{star_id}: failed to read samples: {e}", flush=True)
        return

    P = samples["P"].to_value(u.day)
    e = samples["e"]

    mask = (
        np.isfinite(P) &
        np.isfinite(e) &
        (P > 0) &
        (e >= 0) &
        (e <= 1)
    )

    P = P[mask]
    e = e[mask]

    if len(P) == 0:
        print(f"{star_id}: no valid samples", flush=True)
        return

    logP = np.log10(P)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(logP, e, s=8, alpha=0.5)

    ax.set_xlabel("log10(P/day)")
    ax.set_ylabel("Eccentricity")
    ax.set_ylim(0, 1.02)

    ax.set_title(
        f"Gaia ID: {star_id}\n"
        f"MCMC, N samples = {len(P)}"
    )

    outname = f"{outdir}/Pe_{star_id}_400M_jitter.png"
    fig.savefig(outname, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"{star_id}: saved {outname}", flush=True)

make_pe_plot(2128120221745116416)
