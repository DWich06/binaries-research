import numpy as np
import thejoker as tj
import corner
import matplotlib.pyplot as plt
import os

star_id = 2128127506009807232

workpath = "/data2/labs/douglste-laf-lab/mathewea/200.0M_new"

sample_file = (
    f"{workpath}/{star_id}/"
    f"rejection_samples_200.0M_{star_id}_new.hdf5"
)

samples = tj.JokerSamples.read(sample_file)

P = samples["P"].to_value("day")
e = samples["e"]

mask = (e >= 0) & (e <= 1)
                  
data = np.vstack([
    np.log10(P[mask]),
    e[mask]
]).T

fig = corner.corner(
    data,
    labels=[r"$\log_{10}(P/\mathrm{day})$", "e"],
    bins=50
)

outname = f"{star_id}_Pe_corner.png"
fig.savefig(outname, dpi=200, bbox_inches="tight")

print(f"saved {outname}")

plt.close()

