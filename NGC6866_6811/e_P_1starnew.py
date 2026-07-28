import numpy as np
import thejoker as tj
import corner
import matplotlib.pyplot as plt
import os
from astropy.table import Table
from astropy.table import QTable
from astropy.table import vstack

star_id = 2128173032662959232

workpath = "/data2/labs/douglste-laf-lab/mathewea/200.0M_new"

bimodal_table = Table.read(
    "bimodalcheck_200M_phase.csv",
    format="csv"
)

cat_6811 = QTable.read("rcat_ngc6811_v0.fits")
cat_6866 = QTable.read("rcat_ngc6866_v0.fits")
catalog = vstack([cat_6811, cat_6866])

rejection_file = (
    f"{workpath}/{star_id}/"
    f"rejection_samples_200.0M_{star_id}_new.hdf5"
)

mcmc_file = (
    f"{workpath}/{star_id}/"
    f"rejection_samples_MCMC_200.0M_{star_id}_new.hdf5"
)


if os.path.exists(mcmc_file):
    sample_file = mcmc_file
    print("Using MCMC samples")
else:
    sample_file = rejection_file
    print("Using rejection samples")


samples = tj.JokerSamples.read(sample_file)

row = bimodal_table[bimodal_table["id"] == star_id]

cat_row = catalog[catalog["GAIAEDR3_ID"] == star_id]

ruwe_val = np.nan

if len(cat_row) > 0:
    try:
        ruwe_val = float(cat_row["GAIAEDR3_RUWE"][0])
    except Exception:
        ruwe_val = np.nan

if np.isfinite(ruwe_val):
    ruwe_text = f"RUWE: {ruwe_val:.2f}\n"
else:
    ruwe_text = "RUWE: N/A\n"

nrv_val = np.count_nonzero(catalog["GAIAEDR3_ID"] == star_id)

if len(row) > 0:

    unimodal_raw = row["unimodal"][0]
    bimodal_raw = row["bimodal"][0]

    if unimodal_raw == 1:
        unimodal_val = "Yes"
    elif unimodal_raw == 0:
        unimodal_val = "No"
    else:
        unimodal_val = "Unsure"

    if bimodal_raw == 1:
        bimodal_val = "Yes"
    elif bimodal_raw == 0:
        bimodal_val = "No"
    else:
        bimodal_val = "Unsure"

else:
    unimodal_val = "N/A"
    bimodal_val = "N/A"

print(f"Number of samples: {len(samples)}")


P = samples["P"].to_value("day")
e = samples["e"]


mask = (
    np.isfinite(P) &
    np.isfinite(e) &
    (P > 0) &
    (e >= 0) &
    (e <= 1)
)

v0 = samples["v0"].to_value("km/s")
K = samples["K"].to_value("km/s")

data = np.vstack([
    np.log10(P[mask]),
    e[mask],
    v0[mask],
    K[mask]
]).T

print("data shape:", data.shape)

if len(data) < 3:
    print("Not enough samples for corner plot")
    exit()

levelvalues = [0.196, 0.683, 0.954]
binvalues = 20
smooth1dvalues = 1.0
smoothvalues = 1.0

fig = corner.corner(
    data,
    levels = levelvalues,
    labels=[
    r"$\log_{10}(P/\mathrm{day})$",
    "e",
    r"$v_0\ (\mathrm{km/s})$",
    r"$K\ (\mathrm{km/s})$"
    ],
    bins=binvalues,
    smooth1d=smooth1dvalues,
    smooth=smoothvalues,
    plot_density=True,
    plot_datapoints=True
)

axes = np.array(fig.axes).reshape((4, 4))

ax = axes[0, 3]

ax.axis("off")

HDBcluster_val = "N/A"

if len(cat_row) > 0:
    HDBcluster_val = cat_row["HDBscan_Cluster"][0]

MemBool_val = "N/A"

if len(cat_row) > 0:

    membool_raw = cat_row["MemBool"][0]

    if membool_raw == 1:
        MemBool_val = "Yes"
    elif membool_raw == 0:
        MemBool_val = "No"
    else:
        MemBool_val = "N/A"

cluster_name = "Unknown"

if np.count_nonzero(cat_6811["GAIAEDR3_ID"] == star_id) > 0:
    cluster_name = "NGC 6811"

elif np.count_nonzero(cat_6866["GAIAEDR3_ID"] == star_id) > 0:
    cluster_name = "NGC 6866"

info_text = (
    f"Gaia ID: {star_id}\n"
    f"Cluster: {cluster_name}\n"
    f"N RV data points: {nrv_val}\n"
    f"N samples: {len(samples)}\n"
    f"Ran MCMC: {'Yes' if os.path.exists(mcmc_file) else 'No'}\n"
    f"Unimodal: {unimodal_val}\n"
    f"Bimodal: {bimodal_val}\n"
    f"Member: {MemBool_val}\n"
    f"HDBscan_Cluster: {HDBcluster_val}\n"
    f"{ruwe_text}"
    f"Prior size: 200M\n"
    f"Bins: {binvalues}\n"
    f"Smooth1d: {smooth1dvalues}\n"
    f"Smooth: {smoothvalues}\n"
    f"Levels: {levelvalues}"
)


ax.text(
    0.05,
    0.95,
    info_text,
    transform=ax.transAxes,
    va="top",
    fontsize=10
)

outname = f"plots/{star_id}_PeKv0_corner_Levels.png"
fig.savefig(outname, dpi=200, bbox_inches="tight")

print(f"saved {outname}")

plt.close()