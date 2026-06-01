import numpy as np
import thejoker as tj
import corner
import matplotlib.pyplot as plt
import os
from astropy.table import Table, QTable, vstack

workpath = "/data2/labs/douglste-laf-lab/mathewea/200.0M_new"

outdir = "plots/PeKv0_CornerPlots"
os.makedirs(outdir, exist_ok=True)

bimodal_table = Table.read(
    "bimodalcheck_200M_phase.csv",
    format="csv"
)

cat_6811 = QTable.read("rcat_ngc6811_v0.fits")
cat_6866 = QTable.read("rcat_ngc6866_v0.fits")
catalog = vstack([cat_6811, cat_6866])

ids_6811 = set(cat_6811["GAIAEDR3_ID"])
ids_6866 = set(cat_6866["GAIAEDR3_ID"])

star_ids = [
    name for name in os.listdir(workpath)
    if os.path.isdir(os.path.join(workpath, name)) and name.isdigit()
]

print(f"Found {len(star_ids)} stars")


# Create the function for the corner plots

def make_corner_plot(star_id):

    star_id = int(star_id)

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
    elif os.path.exists(rejection_file):
        sample_file = rejection_file
    else:
        print(f"[{star_id}] No sample file, skipping")
        return

    try:
        samples = tj.JokerSamples.read(sample_file)
    except Exception as e:
        print(f"[{star_id}] Failed to read samples: {e}")
        return


    row = bimodal_table[bimodal_table["id"] == star_id]
    cat_row = catalog[catalog["GAIAEDR3_ID"] == star_id]


    ruwe_val = np.nan
    if len(cat_row) > 0:
        try:
            ruwe_val = float(cat_row["GAIAEDR3_RUWE"][0])
        except Exception:
            ruwe_val = np.nan

    ruwe_text = f"RUWE: {ruwe_val:.2f}\n" if np.isfinite(ruwe_val) else "RUWE: N/A\n"


    nrv_val = len(cat_row)


    unimodal_val = "N/A"
    bimodal_val = "N/A"

    if len(row) > 0:
        u = row["unimodal"][0]
        b = row["bimodal"][0]

        unimodal_val = "Yes" if u == 1 else "No" if u == 0 else "Unsure"
        bimodal_val = "Yes" if b == 1 else "No" if b == 0 else "Unsure"


    P = samples["P"].to_value("day")
    e = samples["e"]
    v0 = samples["v0"].to_value("km/s")
    K = samples["K"].to_value("km/s")

    mask = (
        np.isfinite(P) &
        np.isfinite(e) &
        (P > 0) &
        (e >= 0) &
        (e <= 1)
    )

    data = np.vstack([
        np.log10(P[mask]),
        e[mask],
        v0[mask],
        K[mask]
    ]).T

    if len(data) < 3:
        print(f"[{star_id}] Not enough samples")
        return

    print(f"[{star_id}] samples: {len(data)}")


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

    axes = np.array(fig.axes).reshape(4, 4)
    ax = axes[0, 3]
    ax.axis("off")


    HDBcluster_val = "N/A"
    MemBool_val = "N/A"

    if len(cat_row) > 0:
        HDBcluster_val = cat_row["HDBscan_Cluster"][0]

        mb = cat_row["MemBool"][0]
        MemBool_val = "Yes" if mb == 1 else "No" if mb == 0 else "N/A"

    cluster_name = "Unknown"
    if star_id in ids_6811:
        cluster_name = "NGC 6811"
    elif star_id in ids_6866:
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


    outname = f"{outdir}/{star_id}_PeKv0_corner.png"
    fig.savefig(outname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    del fig

    print(f"[{star_id}] saved")

for i, star_id in enumerate(star_ids):
    try:
        make_corner_plot(star_id)
    except Exception as e:
        print(f"[{star_id}] FAILED: {e}")
        continue

    if i % 50 == 0:
        print(f"Progress: {i}/{len(star_ids)}")