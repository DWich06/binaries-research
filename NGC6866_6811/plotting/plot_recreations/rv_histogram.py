import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table
from thejoker import JokerSamples


workpath = "/data/labs/douglste-laf-lab/wichmand/stardata"

catalogs = {
    "NGC6811": os.path.join(
        workpath,
        "rcat_ngc6811_v0.fits",
    ),
    "NGC6866": os.path.join(
        workpath,
        "rcat_ngc6866_v0.fits",
    ),
}

output_dir = (
    "/data/labs/douglste-laf-lab/wichmand/"
    "plots/plot_recreations"
)

rv_bins = np.linspace(-150, 150, 101)

percentiles = [50, 75, 90]


def get_star_ids(catalog):
    star_ids = []

    for value in catalog["GAIAEDR3_ID"]:
        try:
            star_id = int(value)
        except (TypeError, ValueError, OverflowError):
            continue

        if star_id > 0:
            star_ids.append(star_id)

    return np.unique(star_ids)


def get_sample_file(star_id):
    star_folder = os.path.join(
        workpath,
        "200.0M_new",
        str(star_id),
    )

    mcmc_file = os.path.join(
        star_folder,
        (
            "rejection_samples_MCMC_adapt_full_"
            f"200.0M_{star_id}_new.hdf5"
        ),
    )

    rejection_file = os.path.join(
        star_folder,
        f"rejection_samples_200.0M_{star_id}_new.hdf5",
    )

    if os.path.exists(mcmc_file):
        return mcmc_file, "MCMC"

    if os.path.exists(rejection_file):
        return rejection_file, "Rejection"

    return None, None


def get_ln_k(sample_file):
    samples = JokerSamples.read(sample_file)

    if len(samples) == 0:
        raise ValueError("No samples")

    try:
        k_values = samples["K"].to_value(u.km / u.s)
    except (
        AttributeError,
        TypeError,
        ValueError,
        u.UnitConversionError,
    ):
        k_values = np.asarray(
            samples["K"],
            dtype=float,
        )

    k_values = np.asarray(
        k_values,
        dtype=float,
    )

    k_values = k_values[
        np.isfinite(k_values) & (k_values > 0)
    ]

    if len(k_values) == 0:
        raise ValueError("No valid K values")

    ln_k_values = np.log(k_values)

    return np.percentile(ln_k_values, 1)


def collect_ln_k_values(
    cluster_name,
    catalog,
):
    star_ids = get_star_ids(catalog)

    values = {}

    mcmc_count = 0
    rejection_count = 0
    missing_count = 0
    failed_count = 0

    for star_id in star_ids:
        sample_file, run_type = get_sample_file(star_id)

        if sample_file is None:
            missing_count += 1
            continue

        try:
            values[star_id] = get_ln_k(sample_file)
        except Exception as error:
            print(f"Failed {star_id}: {error}")
            failed_count += 1
            continue

        if run_type == "MCMC":
            mcmc_count += 1
        else:
            rejection_count += 1

    print()
    print(f"{cluster_name} - Non-jitter")
    print(f"Catalog stars: {len(star_ids)}")
    print(f"Stars with lnK values: {len(values)}")
    print(f"MCMC: {mcmc_count}")
    print(f"Rejection: {rejection_count}")
    print(f"Missing: {missing_count}")
    print(f"Failed: {failed_count}")

    return values


def get_selected_rvs(
    catalog,
    ln_k_by_star,
    cut,
):
    selected_ids = {
        star_id
        for star_id, ln_k in ln_k_by_star.items()
        if ln_k <= cut
    }

    weighted_mean_rvs = []

    for star_id in selected_ids:
        rows = catalog[
            catalog["GAIAEDR3_ID"] == star_id
        ]

        rv_values = np.asarray(
            rows["vrad"],
            dtype=float,
        )

        rv_errors = np.asarray(
            rows["vrad_err"],
            dtype=float,
        )

        valid = (
            np.isfinite(rv_values)
            & np.isfinite(rv_errors)
            & (rv_errors > 0)
        )

        rv_values = rv_values[valid]
        rv_errors = rv_errors[valid]

        if len(rv_values) == 0:
            continue

        weights = 1 / rv_errors**2

        weighted_mean_rv = np.average(
            rv_values,
            weights=weights,
        )

        weighted_mean_rvs.append(weighted_mean_rv)

    return np.asarray(
        weighted_mean_rvs,
        dtype=float,
    )


def save_histogram(
    cluster_name,
    percentile,
    rv_values,
    y_max,
):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(
        rv_values,
        bins=rv_bins,
        histtype="step",
        linewidth=1.2,
    )

    ax.set_xlim(
        rv_bins[0],
        rv_bins[-1],
    )

    ax.set_ylim(0, y_max)

    ax.set_xlabel("RV (km/s)")
    ax.set_ylabel("N Stars")

    ax.set_title(
        f"{cluster_name} - {percentile}th percentile cut"
    )

    fig.tight_layout()

    filename = (
        f"{cluster_name}_RVhist_"
        f"{percentile}percent_nonjitter.png"
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

    print(f"Saved: {output_path}")


def main():
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    catalog_data = {}
    ln_k_data = {}

    for cluster_name, catalog_path in catalogs.items():
        if not os.path.exists(catalog_path):
            print(f"Catalog not found: {catalog_path}")
            continue

        catalog = Table.read(catalog_path)

        catalog_data[cluster_name] = catalog

        ln_k_data[cluster_name] = collect_ln_k_values(
            cluster_name,
            catalog,
        )

    if len(catalog_data) != 2:
        print("Both catalogs are required")
        return

    all_ln_k_values = np.concatenate(
        [
            np.asarray(
                list(ln_k_data["NGC6811"].values()),
                dtype=float,
            ),
            np.asarray(
                list(ln_k_data["NGC6866"].values()),
                dtype=float,
            ),
        ]
    )

    cut_values = np.percentile(
        all_ln_k_values,
        percentiles,
    )

    cuts = dict(
        zip(
            percentiles,
            cut_values,
        )
    )

    print()
    print("Combined non-jitter cuts")

    for percentile in percentiles:
        print(
            f"{percentile}th percentile: "
            f"{cuts[percentile]:.3f}"
        )

    plot_data = []

    for cluster_name in catalogs:
        for percentile in percentiles:
            rv_values = get_selected_rvs(
                catalog_data[cluster_name],
                ln_k_data[cluster_name],
                cuts[percentile],
            )

            counts, _ = np.histogram(
                rv_values,
                bins=rv_bins,
            )

            highest_bin = (
                int(np.max(counts))
                if len(counts) > 0
                else 0
            )

            plot_data.append(
                {
                    "cluster": cluster_name,
                    "percentile": percentile,
                    "rvs": rv_values,
                    "highest_bin": highest_bin,
                }
            )

            print()
            print(
                f"{cluster_name}, "
                f"{percentile}th-percentile cut"
            )
            print(
                f"Stars in histogram: "
                f"{len(rv_values)}"
            )

    largest_bin = max(
        item["highest_bin"]
        for item in plot_data
    )

    y_max = max(
        1,
        int(np.ceil(largest_bin * 1.1)),
    )

    for item in plot_data:
        save_histogram(
            item["cluster"],
            item["percentile"],
            item["rvs"],
            y_max,
        )


if __name__ == "__main__":
    main()
