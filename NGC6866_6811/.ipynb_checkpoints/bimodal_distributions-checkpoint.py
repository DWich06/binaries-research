import thejoker as tj
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.table import Table

# only take in -1 stars
data = Table.read("bimodalcheck_200M_phase.csv")
failed_ids = data["id"][data["bimodal"] == -1]

print(failed_ids)

workpath = "/data2/labs/douglste-laf-lab/mathewea/200.0M_new"

for idnum in failed_ids:

    rejection_file = (
        f"{workpath}/{idnum}/"
        f"rejection_samples_200.0M_{idnum}_new.hdf5"
    )

    mcmc_file = (
        f"{workpath}/{idnum}/"
        f"rejection_samples_MCMC_200.0M_{idnum}_new.hdf5"
    )

    # choose file
    if os.path.exists(mcmc_file):
        sample_file = mcmc_file
    else:
        sample_file = rejection_file

    # load samples
    samples = tj.JokerSamples.read(sample_file)

    # extract periods
    P = samples["P"].to_value("day")

    # make plot
    plt.figure(figsize=(8,5))
    plt.hist(np.log10(P), bins=150)

    plt.xlabel(r"$\log_{10}(P/\mathrm{day})$")
    plt.ylabel("Samples")
    plt.title(str(idnum))

    plt.savefig(f"{idnum}_distribution.png")
    plt.close()

    print(f"saved {idnum}")