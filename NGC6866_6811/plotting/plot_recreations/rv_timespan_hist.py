import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table


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

bins = np.arange(0, 4400, 100)


def get_rv_time_spans(catalog):
    star_ids = np.unique(
        catalog["GAIAEDR3_ID"]
    )

    time_spans = []

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

        mjd = mjd[valid]

        if len(mjd) < 2:
            continue

        time_span = (
            np.max(mjd)
            - np.min(mjd)
        )

        time_spans.append(
            time_span
        )

    return np.asarray(
        time_spans
    )


def make_plot(
    cluster_name,
    catalog_path,
):
    catalog = Table.read(
        catalog_path
    )

    time_spans = get_rv_time_spans(
        catalog
    )

    fig, ax = plt.subplots(
        figsize=(5.5, 4.5),
    )

    ax.hist(
        time_spans,
        bins=bins,
        histtype="step",
        color="black",
        linewidth=1.2,
    )

    ax.set_xlabel(
        r"$\Delta$MJD (days)"
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

    fig.tight_layout()

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    filename = (
        f"{cluster_name}_rv_timespan_hist.png"
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
