import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table
from thejoker import JokerSamples


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

joker_dir = os.path.join(
    workpath,
    "joker_runs",
    "200.0M_jitter",
)

output_dir = (
    "/data/labs/douglaslab/wichmand/"
    "plots/plot_recreations"
)

v0_bins = np.linspace(
    -150,
    150,
    101,
)


def get_sample_file(star_id):

    star_folder = os.path.join(
        joker_dir,
        str(star_id),
    )

    mcmc_file = os.path.join(
        star_folder,
        (
            "rejection_samples_MCMC_adapt_full_"
            f"200M_jitter_{star_id}.hdf5"
        ),
    )

    rejection_file = os.path.join(
        star_folder,
        (
            "rejection_samples_200M_jitter_"
            f"{star_id}.hdf5"
        ),
    )

    if os.path.exists(mcmc_file):
        return mcmc_file, "MCMC"

    if os.path.exists(rejection_file):
        return rejection_file, "Rejection"

    return None, None


def get_v0(sample_file):

    samples = JokerSamples.read(
        sample_file
    )

    if len(samples) == 0:
        raise ValueError(
            "No Joker samples"
        )

    v0_values = samples[
        "v0"
    ].to_value(
        u.km / u.s
    )

    v0_values = np.asarray(
        v0_values,
        dtype=float,
    )

    v0_values = v0_values[
        np.isfinite(v0_values)
    ]

    if len(v0_values) == 0:
        raise ValueError(
            "No valid v0 samples"
        )

    v016, v050, v084 = np.percentile(
        v0_values,
        [16, 50, 84],
    )

    e_dn_v0 = (
        v050 - v016
    )

    e_up_v0 = (
        v084 - v050
    )

    return (
        v050,
        e_dn_v0,
        e_up_v0,
    )


def get_star_ids(catalog):

    star_ids = []

    for value in catalog[
        "GAIAEDR3_ID"
    ]:

        try:
            star_id = int(
                value
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            continue

        if star_id > 0:
            star_ids.append(
                star_id
            )

    return np.unique(
        star_ids
    )


def collect_v0_values(
    cluster_name,
    catalog,
):

    star_ids = get_star_ids(
        catalog
    )

    v0_values = []

    mcmc_count = 0
    rejection_count = 0
    missing_count = 0
    failed_count = 0

    for star_id in star_ids:

        (
            sample_file,
            run_type,
        ) = get_sample_file(
            star_id
        )

        if sample_file is None:
            missing_count += 1
            continue

        try:

            (
                v0,
                e_dn_v0,
                e_up_v0,
            ) = get_v0(
                sample_file
            )

        except Exception as error:

            print(
                f"Failed {star_id}: "
                f"{error}"
            )

            failed_count += 1
            continue

        v0_values.append(
            v0
        )

        if run_type == "MCMC":
            mcmc_count += 1

        else:
            rejection_count += 1

    v0_values = np.asarray(
        v0_values,
        dtype=float,
    )

    print()
    print(cluster_name)

    print(
        f"Catalog stars: "
        f"{len(star_ids)}"
    )

    print(
        f"Stars with v0 values: "
        f"{len(v0_values)}"
    )

    print(
        f"MCMC: "
        f"{mcmc_count}"
    )

    print(
        f"Rejection: "
        f"{rejection_count}"
    )

    print(
        f"Missing: "
        f"{missing_count}"
    )

    print(
        f"Failed: "
        f"{failed_count}"
    )

    if len(v0_values) > 0:

        print()
        print("v0 distribution:")

        print(
            f"Min: "
            f"{np.min(v0_values):.3f} km/s"
        )

        print(
            f"16th percentile: "
            f"{np.percentile(v0_values, 16):.3f} km/s"
        )

        print(
            f"Median: "
            f"{np.percentile(v0_values, 50):.3f} km/s"
        )

        print(
            f"84th percentile: "
            f"{np.percentile(v0_values, 84):.3f} km/s"
        )

        print(
            f"Max: "
            f"{np.max(v0_values):.3f} km/s"
        )

    return v0_values


def save_histogram(
    cluster_name,
    v0_values,
    y_max,
):

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.hist(
        v0_values,
        bins=v0_bins,
        histtype="step",
        linewidth=1.2,
    )

    ax.set_xlim(
        v0_bins[0],
        v0_bins[-1],
    )

    ax.set_ylim(
        0,
        y_max,
    )

    ax.set_xlabel(
        r"$v_0$ (km/s)"
    )

    ax.set_ylabel(
        "N Stars"
    )

    ax.set_title(
        cluster_name
    )

    fig.tight_layout()

    filename = (
        f"{cluster_name}_v0_histogram.png"
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

    plt.close(
        fig
    )

    print(
        f"Saved: "
        f"{output_path}"
    )


def main():

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    plot_data = []

    for (
        cluster_name,
        catalog_path,
    ) in catalogs.items():

        if not os.path.exists(
            catalog_path
        ):

            print(
                f"Catalog not found: "
                f"{catalog_path}"
            )

            continue

        catalog = Table.read(
            catalog_path
        )

        v0_values = collect_v0_values(
            cluster_name,
            catalog,
        )

        counts, _ = np.histogram(
            v0_values,
            bins=v0_bins,
        )

        highest_bin = (
            int(
                np.max(counts)
            )
            if len(counts) > 0
            else 0
        )

        plot_data.append(
            {
                "cluster": cluster_name,
                "v0_values": v0_values,
                "highest_bin": highest_bin,
            }
        )

    if len(plot_data) == 0:
        print(
            "No plots to make"
        )
        return

    largest_bin = max(
        item[
            "highest_bin"
        ]
        for item in plot_data
    )

    y_max = max(
        1,
        int(
            np.ceil(
                largest_bin * 1.1
            )
        ),
    )

    for item in plot_data:

        save_histogram(
            item[
                "cluster"
            ],
            item[
                "v0_values"
            ],
            y_max,
        )


if __name__ == "__main__":
    main()
