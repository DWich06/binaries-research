import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import astropy.units as u
from astropy.table import Table, QTable, vstack
from astropy.time import Time
import thejoker as tj
import corner
import traceback


# Data paths
stardata = "/data/labs/douglste-laf-lab/wichmand/stardata"
outdir = "/data/labs/douglste-laf-lab/wichmand/plots/GridSpecPlots"
os.makedirs(outdir, exist_ok=True)


# Star catalogs
cat_6811 = QTable.read(f"{stardata}/rcat_ngc6811_v0.fits")
cat_6866 = QTable.read(f"{stardata}/rcat_ngc6866_v0.fits")
catalog = vstack([cat_6811, cat_6866])

# Corner Plot Info
levelvalues = [0.196, 0.683, 0.954]
binvalues = 20
smooth1dvalues = 1.0
smoothvalues = 1.0

# All joker run definitions

jokerruns = {
    "200M": {
        "workpath": f"{stardata}/200.0M_new",
        "filename": "rejection_samples_200.0M_{id}_new.hdf5",
        "has_jitter": False,
        "bimodal_file": f"{stardata}/bimodalcheck_200M_phase.csv",
        "prior_size": "200M",
        "sample_type": "Rejection",
    },

    "200M_MCMC": {
        "workpath": f"{stardata}/200.0M_new",
        "filename": "rejection_samples_MCMC_200.0M_{id}_new.hdf5",
        "has_jitter": False,
        "bimodal_file": f"{stardata}/bimodalcheck_200M_phase.csv",
        "prior_size": "200M",
        "sample_type": "MCMC",
    },

    "200M_MCMC_adapt_full": {
        "workpath": f"{stardata}/200.0M_new",
        "filename": "rejection_samples_MCMC_adapt_full_200.0M_{id}_new.hdf5",
        "has_jitter": False,
        "bimodal_file": f"{stardata}/bimodalcheck_200M_adapt_full.csv",
        "prior_size": "200M",
        "sample_type": "MCMC adapt_full",
    },

    "200M_jitter": {
        "workpath": f"{stardata}/200.0M_jitter",
        "filename": "rejection_samples_200M_jitter_{id}.hdf5",
        "has_jitter": True,
        "bimodal_file": f"{stardata}/bimodalcheck_200M_jitter_adapt_full.csv",
        "prior_size": "200M",
        "sample_type": "Rejection + jitter",
    },

    "200M_jitter_MCMC": {
        "workpath": f"{stardata}/200.0M_jitter",
        "filename": "rejection_samples_MCMC_adapt_full_200M_jitter_{id}.hdf5",
        "has_jitter": True,
        "bimodal_file": f"{stardata}/bimodalcheck_200M_jitter_adapt_full.csv",
        "prior_size": "200M",
        "sample_type": "MCMC adapt_full + jitter",
    },

    "400M_jitter": {
        "workpath": f"{stardata}/400.0M_jitter",
        "filename": "rejection_samples_400M_jitter_{id}.hdf5",
        "has_jitter": True,
        "bimodal_file": f"{stardata}/bimodalcheck_400M_jitter.csv",
        "prior_size": "400M",
        "sample_type": "Rejection + jitter",
    },
}


# Helper functions

bimodal_tables = {}


def get_bimodal_table(run_name):

    cfg = jokerruns[run_name]
    filename = cfg["bimodal_file"]

    if filename in bimodal_tables:
        return bimodal_tables[filename]

    table = Table.read(filename, format="csv")
    bimodal_tables[filename] = table

    return table

def get_bimodal_row(star_id, run_name):

    table = get_bimodal_table(run_name)

    row = table[
        np.asarray(table["id"], dtype=np.int64) == int(star_id)
    ]

    if len(row) == 0:
        return None

    return row

def get_cluster_name(star_id):
    star_id = int(star_id)

    if star_id in set(cat_6811["GAIAEDR3_ID"]):
        return "NGC 6811"

    if star_id in set(cat_6866["GAIAEDR3_ID"]):
        return "NGC 6866"

    return "Unknown"

def get_catalog_row(star_id):
    row = catalog[catalog["GAIAEDR3_ID"] == int(star_id)]

    if len(row) == 0:
        return None

    return row

def get_corner_info(star_id, run_name, samples):

    cfg = jokerruns[run_name]

    cluster_name = get_cluster_name(star_id)
    cat_row = get_catalog_row(star_id)
    bimodal_row = get_bimodal_row(star_id, run_name)

    nrv_val = 0
    ruwe_text = "N/A"
    member_text = "N/A"
    hdbscan_text = "N/A"

    if cat_row is not None:
        nrv_val = len(cat_row)

        try:
            ruwe_val = float(cat_row["GAIAEDR3_RUWE"][0])

            if np.isfinite(ruwe_val):
                ruwe_text = f"{ruwe_val:.2f}"

        except Exception:
            ruwe_text = "N/A"

        try:
            member_val = int(cat_row["MemBool"][0])

            if member_val == 1:
                member_text = "Yes"
            elif member_val == 0:
                member_text = "No"

        except Exception:
            member_text = "N/A"

        try:
            hdbscan_text = str(cat_row["HDBscan_Cluster"][0])

        except Exception:
            hdbscan_text = "N/A"

    unimodal_text = "N/A"
    bimodal_text = "N/A"

    if bimodal_row is not None:

        try:
            unimodal_val = int(bimodal_row["unimodal"][0])

            if unimodal_val == 1:
                unimodal_text = "Yes"
            elif unimodal_val == 0:
                unimodal_text = "No"
            else:
                unimodal_text = "Unsure"

        except Exception:
            unimodal_text = "N/A"

        try:
            bimodal_val = int(bimodal_row["bimodal"][0])

            if bimodal_val == 1:
                bimodal_text = "Yes"
            elif bimodal_val == 0:
                bimodal_text = "No"
            elif bimodal_val == -1:
                bimodal_text = "P-K modal exception"
            elif bimodal_val == -2:
                bimodal_text = "Less than 10 samples"
            else:
                bimodal_text = "Error"

        except Exception:
            bimodal_text = "N/A"

    levelvalues = [0.196, 0.683, 0.954]
    binvalues = 20
    smooth1dvalues = 1.0
    smoothvalues = 1.0

    info_text = (
        f"Gaia ID: {star_id}\n"
        f"Cluster: {cluster_name}\n"
        f"N RV data points: {nrv_val}\n"
        f"N samples: {len(samples)}\n"
        f"Sample type: {cfg['sample_type']}\n"
        f"Unimodal: {unimodal_text}\n"
        f"Bimodal: {bimodal_text}\n"
        f"Member: {member_text}\n"
        f"HDBscan Cluster: {hdbscan_text}\n"
        f"RUWE: {ruwe_text}\n"
        f"Prior size: {cfg['prior_size']}\n"
        f"Bins: {binvalues}\n"
        f"Smooth1d: {smooth1dvalues}\n"
        f"Smooth: {smoothvalues}\n"
        f"Levels: {levelvalues}"
    )

    return info_text

def find_sample_file(star_id, run_name):
    star_id = int(star_id)
    cfg = jokerruns[run_name]

    path = os.path.join(
        cfg["workpath"],
        str(star_id),
        cfg["filename"].format(id=star_id)
    )

    if os.path.exists(path):
        return path

    return None


def get_star_rows(star_id):
    star_id = int(star_id)

    rows = catalog[catalog["GAIAEDR3_ID"] == star_id]

    return rows

def get_rvdata(star_id):
    row = get_star_rows(star_id)

    if len(row) < 3:
        return None

    t = Time(row["DATE-OBS"], format="fits", scale="tcb")

    data = tj.RVData(
        t=t,
        rv=row["vrad"] * u.km / u.s,
        rv_err=row["vrad_err"] * u.km / u.s,
    )

    return data


# Plotting functions

def plot_cmd(ax, star_id):
    star_id = int(star_id)

    if star_id in set(cat_6811["GAIAEDR3_ID"]):
        cluster_cat = cat_6811
        cluster_name = "NGC 6811"
    elif star_id in set(cat_6866["GAIAEDR3_ID"]):
        cluster_cat = cat_6866
        cluster_name = "NGC 6866"
    else:
        cluster_cat = catalog
        cluster_name = "Unknown cluster"

    members = cluster_cat["MemBool"] == 1

    bp_rp = (
        cluster_cat["GAIAEDR3_BP"][members]
        - cluster_cat["GAIAEDR3_RP"][members]
    )
    G = cluster_cat["GAIAEDR3_G"][members]
    dist = cluster_cat["dist"][members]
    MG = G - 5 * np.log10(dist) + 5

    ax.scatter(bp_rp, MG, s=5, alpha=0.4)

    row = cluster_cat[cluster_cat["GAIAEDR3_ID"] == star_id]

    if len(row) > 0:

        try:

            star_bp_rp = float(row["GAIAEDR3_BP"][0] - row["GAIAEDR3_RP"][0])
            star_G = float(row["GAIAEDR3_G"][0])
            star_dist = float(row["dist"][0])
            star_MG = star_G - 5 * np.log10(star_dist) + 5

            ax.scatter(
                star_bp_rp,
                star_MG,
                s=130,
                marker="*",
                edgecolor="k",
                zorder=5,
            )

        except Exception as err:

            print(f"Failed to plot CMD for star {star_id}: {err}", flush=True,)

    ax.invert_yaxis()
    ax.set_xlabel("BP - RP")
    ax.set_ylabel(r"$M_G$")
    ax.set_title(f"{cluster_name}")

def plot_rv(ax, star_id, samples, has_jitter=False, jitter_multiplier=1.0):
    data = get_rvdata(star_id)

    if data is None:
        ax.text(0.5, 0.5, "Less than 3 RV points", ha="center", va="center")
        ax.set_axis_off()
        return

    try:
        tj.plot_rv_curves(samples, data=data, ax=ax)

        if has_jitter and "s" in samples.par_names and len(samples) > 0:

            jitter = float(np.median(samples["s"].to_value(u.km / u.s))) * jitter_multiplier
            rv_err = data.rv_err.to_value(u.km / u.s)
            rv_val = np.asarray(data.rv.to_value(u.km / u.s), dtype=float)
            n_obs = len(rv_err)

            total_err = np.sqrt(rv_err**2 + jitter**2)

            # Find the container that holds the actual data points
            # (matches on both length and y-values, in case thejoker
            # draws other same-length containers, e.g. resampled draws)
            data_x = None
            for container in ax.containers:
                x_vals = np.asarray(container.lines[0].get_xdata(), dtype=float)
                y_vals = np.asarray(container.lines[0].get_ydata(), dtype=float)
                if len(y_vals) == n_obs and np.allclose(np.sort(y_vals), np.sort(rv_val), atol=1e-6):
                    data_x = x_vals
                    break

            if data_x is not None:
                eb = ax.errorbar(
                    data_x,
                    rv_val,
                    yerr=total_err,
                    fmt="none",
                    ecolor="red",
                    elinewidth=0.8,
                    capsize=0,
                    zorder=0,
                )

                for line in eb.lines:
                    if line is None:
                        continue
                    try:
                        for l in line:
                            l.set_snap(False)
                    except TypeError:
                        line.set_snap(False)

            else:
                print(f"{star_id}: could not find matching data container for jitter bars", flush=True)


    except Exception as e:
        print(f"RV curve failed for {star_id}:", flush=True)
        print(traceback.format_exc(), flush=True)

        ax.text(
            0.5,
            0.5,
            f"RV curve failed\n{e}",
            ha="center",
            va="center",
            fontsize=8,
        )
        ax.set_axis_off()



def plot_phase_fold(ax, star_id, samples):
    data = get_rvdata(star_id)

    if data is None:
        ax.text(0.5, 0.5, "Less than 3 RV points", ha="center", va="center")
        ax.set_axis_off()
        return

    if len(samples) == 0:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center")
        ax.set_axis_off()
        return

    try:
        sample = tj.MAP_sample(samples)

        tj.plot_phase_fold(
            sample,
            data=data,
            ax=ax,
        )

        ax.set_xlabel("Orbital phase")
        ax.set_ylabel(r"RV (km s$^{-1}$)")
        ax.set_title("")

    except Exception as e:
        ax.text(
            0.5,
            0.5,
            "Phase fold\nreserved space",
            ha="center",
            va="center",
            fontsize=10,
        )

def plot_pe(ax, samples):
    P = samples["P"].to_value(u.day)
    e = samples["e"]

    mask = (
        np.isfinite(P)
        & np.isfinite(e)
        & (P > 0)
        & (e >= 0)
        & (e <= 1)
    )

    P = P[mask]
    e = e[mask]

    if len(P) == 0:
        ax.text(0.5, 0.5, "No valid samples", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.scatter(np.log10(P), e, s=6, alpha=0.4)

    ax.set_xlabel(r"$\log_{10}(P/\mathrm{day})$")
    ax.set_ylabel("e")
    ax.set_ylim(0, 1.02)

def get_corner_data(samples, has_jitter=False):
    P = samples["P"].to_value(u.day)
    e = samples["e"]
    v0 = samples["v0"].to_value(u.km / u.s)
    K = samples["K"].to_value(u.km / u.s)

    values = [
        np.log10(P),
        e,
        v0,
        K,
    ]

    labels = [
        r"$\log_{10}(P/\mathrm{day})$",
        "e",
        r"$v_0\ (\mathrm{km/s})$",
        r"$K\ (\mathrm{km/s})$",
    ]

    mask = (
        np.isfinite(P)
        & np.isfinite(e)
        & np.isfinite(v0)
        & np.isfinite(K)
        & (P > 0)
        & (e >= 0)
        & (e <= 1)
    )

    if has_jitter:
        s = samples["s"].to_value(u.km / u.s)

        values.append(s)
        labels.append(r"$s\ (\mathrm{km/s})$")

        mask = mask & np.isfinite(s)

    data = np.vstack([v[mask] for v in values]).T

    return data, labels


# GridSpec plot

def make_gridspec_plot(star_id, run_name):
    star_id = int(star_id)

    sample_file = find_sample_file(star_id, run_name)

    if sample_file is None:
        print(f"{star_id} {run_name}: no sample file", flush=True)
        return

    try:
        samples = tj.JokerSamples.read(sample_file)

    except Exception as e:
        print(f"{star_id} {run_name}: failed to read samples: {e}", flush=True)
        return

    has_jitter = jokerruns[run_name]["has_jitter"]
    corner_data, corner_labels = get_corner_data(samples, has_jitter=has_jitter)

    if len(corner_data) < 3:
        print(f"{star_id} {run_name}: not enough samples", flush=True)

    fig = plt.figure(figsize=(8.5, 11), constrained_layout=False)

    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1.75, 0.75, 0.75],
        width_ratios=[0.62, 1.18],
        left=0.09,
        right=0.98,
        bottom=0.06,
        top=0.94,
        hspace=0.28,
        wspace=0.28,
    )

    ax_cmd = fig.add_subplot(gs[0, 0])
    corner_subfig = fig.add_subfigure(gs[0, 1])

    ax_phase = fig.add_subplot(gs[1, 0])
    ax_pe = fig.add_subplot(gs[1, 1])

    ax_rv = fig.add_subplot(gs[2, :])

    plot_cmd(ax_cmd, star_id)
    plot_rv(ax_rv, star_id, samples, has_jitter=has_jitter, jitter_multiplier=1.0)
    plot_phase_fold(ax_phase, star_id, samples)
    plot_pe(ax_pe, samples)

    corner_failed = False

    try:
        corner.corner(
            corner_data,
            fig=corner_subfig,
            labels=corner_labels,
            levels=levelvalues,
            bins=binvalues,
            smooth=smoothvalues,
            smooth1d=smooth1dvalues,
            plot_density=True,
            plot_datapoints=True,
            show_titles=False,
        )

    except Exception as err:
        corner_failed = True
        print(f"Skipping corner plot for star {star_id}: {err}", flush=True,)

        ax_corner_blank = corner_subfig.subplots()
        ax_corner_blank.axis("off")

        ax_corner_blank.text(
            0.5,
            0.5,
            "Corner plot unavailable",
            ha="center",
            va="center",
            fontsize=8,
            transform=ax_corner_blank.transAxes,
        )
    if not corner_failed:

        for ax in corner_subfig.axes:
            ax.tick_params(axis="both", labelsize=5)
            ax.xaxis.label.set_fontsize(6)
            ax.yaxis.label.set_fontsize(6)

        if has_jitter:
            labelpad = 40
        else:
            labelpad = 10

        for ax in corner_subfig.axes:
            ax.xaxis.labelpad = labelpad
            ax.yaxis.labelpad = labelpad

        corner_subfig.subplots_adjust(
            left=0.12,
            right=0.96,
            bottom=0.22,
            top=0.89,
            wspace=0.08,
            hspace=0.08,
        )

    info_text = get_corner_info(
        star_id,
        run_name,
        samples,
    )

    corner_subfig.text(
        0.65,
        0.93,
        info_text,
        transform=corner_subfig.transSubfigure,
        ha="left",
        va="top",
        fontsize=7,
        zorder=100,
    )

    #fig.suptitle(
     #   f"Gaia ID: {star_id} | {run_name} | N samples = {len(samples)}",
      #  fontsize=16,
    #)

    run_outdir = os.path.join(outdir, run_name)
    os.makedirs(run_outdir, exist_ok=True)

    outname = os.path.join(
        run_outdir,
        f"Summary_{star_id}_{run_name}.png",
    )

    fig.savefig(outname, dpi=300)
    plt.close(fig)
    print(f"{star_id} {run_name}: saved {outname}", flush=True)


allrunsfile = "/data/labs/douglste-laf-lab/wichmand/stardata/gridspec_star_runs.txt"

array_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
task_offset = int(os.getenv("TASK_OFFSET", "0"))

jobid = array_id + task_offset

allruns = np.loadtxt(allrunsfile, dtype=str)

star_id = allruns[jobid, 0]
run_name = allruns[jobid, 1]

print(f"SLURM_ARRAY_TASK_ID = {jobid}")
print(f"Running {star_id} ({run_name})")

make_gridspec_plot(int(star_id), run_name)
