import os
import traceback

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import QTable, vstack
from astropy.time import Time

import thejoker as tj


star_id = 2128120221745116416

stardata = "/data/labs/douglste-laf-lab/wichmand/stardata"

outdir = (
    f"/data/labs/douglste-laf-lab/wichmand/"
    f"plots/RVCurves/Comparison_{star_id}"
)

os.makedirs(outdir, exist_ok=True)


print("Reading NGC 6811 catalog", flush=True)
cat_6811 = QTable.read(
    f"{stardata}/rcat_ngc6811_v0.fits"
)

print("Reading NGC 6866 catalog", flush=True)
cat_6866 = QTable.read(
    f"{stardata}/rcat_ngc6866_v0.fits"
)

catalog = vstack([cat_6811, cat_6866])

print("Finished reading catalogs", flush=True)


jokerruns = {
    "200M": {
        "workpath": f"{stardata}/200.0M_new",
        "filename": (
            "rejection_samples_200.0M_{id}_new.hdf5"
        ),
        "title": "200M Rejection Sampling",
        "has_jitter": False,
    },

    "200M_MCMC": {
        "workpath": f"{stardata}/200.0M_new",
        "filename": (
            "rejection_samples_MCMC_200.0M_{id}_new.hdf5"
        ),
        "title": "200M MCMC",
        "has_jitter": False,
    },

    "200M_MCMC_adapt_full": {
        "workpath": f"{stardata}/200.0M_new",
        "filename": (
            "rejection_samples_MCMC_adapt_full_"
            "200.0M_{id}_new.hdf5"
        ),
        "title": "200M MCMC adapt_full",
        "has_jitter": False,
    },

    "200M_jitter": {
        "workpath": f"{stardata}/200.0M_jitter",
        "filename": (
            "rejection_samples_200M_jitter_{id}.hdf5"
        ),
        "title": "200M Rejection Sampling with Jitter",
        "has_jitter": True,
    },

    "200M_jitter_MCMC": {
        "workpath": f"{stardata}/200.0M_jitter",
        "filename": (
            "rejection_samples_MCMC_adapt_full_"
            "200M_jitter_{id}.hdf5"
        ),
        "title": "200M Jitter MCMC adapt_full",
        "has_jitter": True,
    },

    "400M_jitter": {
        "workpath": f"{stardata}/400.0M_jitter",
        "filename": (
            "rejection_samples_400M_jitter_{id}.hdf5"
        ),
        "title": "400M Rejection Sampling with Jitter",
        "has_jitter": True,
    },
}



def find_sample_file(star_id, run_name):

    cfg = jokerruns[run_name]

    return os.path.join(
        cfg["workpath"],
        str(star_id),
        cfg["filename"].format(id=star_id),
    )


def get_rvdata(star_id):

    rows = catalog[
        catalog["GAIAEDR3_ID"] == star_id
    ]

    if len(rows) < 3:
        return None

    rows.sort("DATE-OBS")

    t = Time(
        rows["DATE-OBS"],
        format="fits",
        scale="tcb",
    )

    return tj.RVData(
        t=t,
        rv=rows["vrad"] * (u.km / u.s),
        rv_err=rows["vrad_err"] * (u.km / u.s),
    )


def add_jitter_bars(
    ax,
    samples,
    data,
    jitter_multiplier=1.0,
):

    if len(samples) == 0:
        return

    if "s" not in samples.par_names:
        print(
            "Samples are marked as jitter samples, "
            "but parameter 's' is missing.",
            flush=True,
        )
        return

    jitter_values = samples["s"].to_value(u.km / u.s)

    jitter = (
        float(np.median(jitter_values))
        * jitter_multiplier
    )

    jitter = abs(jitter)

    rv_err = np.asarray(
        data.rv_err.to_value(u.km / u.s),
        dtype=float,
    )

    rv_val = np.asarray(
        data.rv.to_value(u.km / u.s),
        dtype=float,
    )

    total_err = np.sqrt(
        rv_err**2 + jitter**2
    )

    n_obs = len(rv_val)
    data_x = None

    for container in ax.containers:

        try:
            point_line = container.lines[0]

            x_vals = np.asarray(
                point_line.get_xdata(),
                dtype=float,
            )

            y_vals = np.asarray(
                point_line.get_ydata(),
                dtype=float,
            )

        except (AttributeError, IndexError, TypeError):
            continue

        if (
            len(y_vals) == n_obs
            and np.allclose(
                np.sort(y_vals),
                np.sort(rv_val),
                atol=1e-6,
            )
        ):
            data_x = x_vals
            break

    if data_x is None:
        print(
            "Could not find The Joker's observed-data "
            "error-bar container.",
            flush=True,
        )
        return

    jitter_errorbars = ax.errorbar(
        data_x,
        rv_val,
        yerr=total_err,
        fmt="none",
        ecolor="red",
        elinewidth=0.8,
        capsize=0,
        zorder=0,
    )

    for line_group in jitter_errorbars.lines:

        if line_group is None:
            continue

        try:
            for line in line_group:
                line.set_snap(False)

        except TypeError:
            line_group.set_snap(False)

    print(
        f"Median jitter: {jitter:.4f} km/s",
        flush=True,
    )


def make_rv_plot(
    star_id,
    run_name,
    data,
    jitter_multiplier=1.0,
):
    """Create and save one RV-curve plot."""

    cfg = jokerruns[run_name]
    sample_file = find_sample_file(star_id, run_name)

    print(f"\nChecking {run_name}", flush=True)
    print(sample_file, flush=True)

    if not os.path.exists(sample_file):
        print(
            f"{run_name}: sample file does not exist",
            flush=True,
        )
        return

    try:
        samples = tj.JokerSamples.read(sample_file)

    except Exception as error:
        print(
            f"{run_name}: failed to read samples: {error}",
            flush=True,
        )
        return

    if len(samples) == 0:
        print(
            f"{run_name}: no samples",
            flush=True,
        )
        return

    print(
        f"{run_name}: loaded {len(samples)} samples",
        flush=True,
    )

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    try:
        tj.plot_rv_curves(
            samples,
            data=data,
            ax=ax,
        )

        if cfg["has_jitter"]:
            add_jitter_bars(
                ax=ax,
                samples=samples,
                data=data,
                jitter_multiplier=jitter_multiplier,
            )

        ax.set_title(
            f"Gaia ID: {star_id}\n"
            f"{cfg['title']}, N = {len(samples)}"
        )

        ax.grid(False)

        outname = os.path.join(
            outdir,
            f"RVCurves_{star_id}_{run_name}.png",
        )

        fig.savefig(
            outname,
            dpi=200,
            bbox_inches="tight",
        )

        print(
            f"{run_name}: saved {outname}",
            flush=True,
        )

    except Exception:
        print(
            f"{run_name}: RV plotting failed",
            flush=True,
        )
        print(
            traceback.format_exc(),
            flush=True,
        )

    finally:
        plt.close(fig)



rv_data = get_rvdata(star_id)

if rv_data is None:
    raise RuntimeError(
        f"Star {star_id} has fewer than 3 RV observations."
    )

print(
    f"Found {len(rv_data)} RV observations "
    f"for star {star_id}",
    flush=True,
)

for run_name in jokerruns:
    make_rv_plot(
        star_id=star_id,
        run_name=run_name,
        data=rv_data,
        jitter_multiplier=1.0,
    )

print("\nFinished all available runs.", flush=True)
