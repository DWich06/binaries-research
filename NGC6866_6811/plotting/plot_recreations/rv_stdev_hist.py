import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.modeling import models, fitting


workpath = "/data/labs/douglaslab/wichmand/stardata"

catalogs = {
    "NGC6811": os.path.join(
        workpath,
        "star_catalogs",
        "rcat_ngc6811_v0.fits",
    ),
    "NGC6866": os.path.join(
        workpath,
        "star_catalogs",
        "rcat_ngc6866_v0.fits",
    ),
}

output_dir = (
    "/data/labs/douglaslab/wichmand/plots/plot_recreations"
)

bins = np.linspace(0, 4, 21)


def get_rv_standard_deviations(catalog):
    star_ids = np.unique(
        catalog["GAIAEDR3_ID"]
    )

    rv_stdevs = []

    for star_id in star_ids:
        rows = catalog[
            catalog["GAIAEDR3_ID"] == star_id
        ]

        rv = np.asarray(
            rows["vrad"],
            dtype=float,
        )

        rv_err = np.asarray(
            rows["vrad_err"],
            dtype=float,
        )

        mjd = np.asarray(
            rows["MJD"],
            dtype=float,
        )

        valid = (
            np.isfinite(rv)
            & np.isfinite(rv_err)
            & (rv_err > 0)
            & np.isfinite(mjd)
        )

        rv = rv[valid]
        mjd = mjd[valid]

        if len(rv) < 3:
            continue

        order = np.argsort(mjd)

        rv = rv[order]

        first_three_rvs = rv[:3]

        rv_stdev = np.std(
            first_three_rvs,
            ddof=1,
        )

        rv_stdevs.append(
            rv_stdev
        )

    return np.asarray(
        rv_stdevs
    )


def fit_gaussian(rv_stdevs):
    counts, edges = np.histogram(
        rv_stdevs,
        bins=bins,
    )

    bin_centers = (
        edges[:-1] + edges[1:]
    ) / 2

    amplitude_guess = np.max(
        counts
    )

    mean_guess = bin_centers[
        np.argmax(counts)
    ]

    sigma_guess = 0.5

    gaussian_init = models.Gaussian1D(
        amplitude=amplitude_guess,
        mean=mean_guess,
        stddev=sigma_guess,
    )

    fitter = fitting.LevMarLSQFitter()

    gaussian_fit = fitter(
        gaussian_init,
        bin_centers,
        counts,
    )

    return gaussian_fit


def make_plot(
    cluster_name,
    catalog_path,
):
    catalog = Table.read(
        catalog_path
    )

    rv_stdevs = get_rv_standard_deviations(
        catalog
    )

    gaussian_fit = fit_gaussian(
        rv_stdevs
    )

    x_fit = np.linspace(
        0,
        4,
        1000,
    )

    y_fit = gaussian_fit(
        x_fit
    )

    fig, ax = plt.subplots(
        figsize=(5.5, 4.5),
    )

    ax.hist(
        rv_stdevs,
        bins=bins,
        histtype="step",
        color="black",
        linewidth=1.2,
        label="RV standard deviation",
    )

    ax.plot(
        x_fit,
        y_fit,
        color="red",
        linestyle=":",
        linewidth=2,
        label="Gaussian fit",
    )

    ax.set_xlabel(
        r"$\sigma_{\rm obs}$ (km s$^{-1}$)"
    )

    ax.set_ylabel(
        "N Stars"
    )

    ax.set_xlim(
        bins[0],
        bins[-1],
    )

    ax.set_title(
        cluster_name
    )

    ax.legend()

    fig.tight_layout()

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    filename = (
        f"{cluster_name}_rv_stdev_hist.png"
    )

    output_path = os.path.join(
        output_dir,
        filename,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    for (
        cluster_name,
        catalog_path,
    ) in catalogs.items():

        make_plot(
            cluster_name,
            catalog_path,
        )


if __name__ == "__main__":
    main()
