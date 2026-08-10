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

bins = np.linspace(0, 4, 21)


def get_rv_standard_deviations(catalog):

    star_ids = np.unique(
        catalog["GAIAEDR3_ID"]
    )

    rv_stdevs = []

    n_three_rvs = 0
    n_too_few = 0

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
            n_too_few += 1
            continue

        n_three_rvs += 1

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

    return (
        np.asarray(rv_stdevs),
        n_three_rvs,
        n_too_few,
    )


def make_plot(
    cluster_name,
    catalog_path,
):

    catalog = Table.read(
        catalog_path
    )

    (
        rv_stdevs,
        n_three_rvs,
        n_too_few,
    ) = get_rv_standard_deviations(
        catalog
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

    print()
    print(cluster_name)

    print(
        f"Total unique stars: "
        f"{len(np.unique(catalog['GAIAEDR3_ID']))}"
    )

    print(
        f"Stars with >=3 valid RVs: "
        f"{n_three_rvs}"
    )

    print(
        f"Stars with <3 valid RVs: "
        f"{n_too_few}"
    )

    print(
        f"Median sigma_obs: "
        f"{np.median(rv_stdevs):.3f} km/s"
    )

    print(
        f"Min sigma_obs: "
        f"{np.min(rv_stdevs):.3f} km/s"
    )

    print(
        f"Max sigma_obs: "
        f"{np.max(rv_stdevs):.3f} km/s"
    )

    percentiles = np.percentile(
        rv_stdevs,
        [50, 75, 90, 95, 99],
    )

    print()
    print("sigma_obs percentiles:")

    print(
        f"50th: {percentiles[0]:.3f} km/s"
    )

    print(
        f"75th: {percentiles[1]:.3f} km/s"
    )

    print(
        f"90th: {percentiles[2]:.3f} km/s"
    )

    print(
        f"95th: {percentiles[3]:.3f} km/s"
    )

    print(
        f"99th: {percentiles[4]:.3f} km/s"
    )

    print()
    print("High-sigma counts:")

    for limit in [
        4,
        5,
        10,
        20,
        50,
        100,
    ]:

        count = np.sum(
            rv_stdevs > limit
        )

        fraction = (
            100 * count / len(rv_stdevs)
        )

        print(
            f"sigma_obs > {limit:3} km/s: "
            f"{count:3} "
            f"({fraction:.1f}%)"
        )

    print()
    print(
        f"Saved: {output_path}"
    )


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
