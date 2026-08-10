import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table
from astropy.modeling import models, fitting
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

rv_bins = np.linspace(-150, 150, 100)


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


def collect_ln_k_values(cluster_name, catalog):
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


def get_weighted_mean_rvs(
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

        if np.isfinite(weighted_mean_rv):
            weighted_mean_rvs.append(weighted_mean_rv)

    return np.asarray(
        weighted_mean_rvs,
        dtype=float,
    )


def fit_double_gaussian(rv_values):
    counts, edges = np.histogram(
        rv_values,
        bins=rv_bins,
    )

    bin_centers = (
        edges[:-1] + edges[1:]
    ) / 2

    peak_index = np.argmax(counts)
    cluster_mean_guess = bin_centers[peak_index]
    cluster_amplitude_guess = counts[peak_index]

    outside_cluster = (
        np.abs(bin_centers - cluster_mean_guess) > 10
    )

    if np.any(outside_cluster):
        field_counts = counts[outside_cluster]
        field_centers = bin_centers[outside_cluster]

        field_peak_index = np.argmax(field_counts)
        field_amplitude_guess = max(
            field_counts[field_peak_index],
            1,
        )
        field_mean_guess = field_centers[field_peak_index]
    else:
        field_amplitude_guess = max(
            cluster_amplitude_guess / 5,
            1,
        )
        field_mean_guess = np.median(rv_values)

    cluster_guess = models.Gaussian1D(
        amplitude=cluster_amplitude_guess,
        mean=cluster_mean_guess,
        stddev=5,
        bounds={
            "amplitude": (0, None),
            "mean": (
                cluster_mean_guess - 10,
                cluster_mean_guess + 10,
            ),
            "stddev": (0.2, 10),
        },
    )

    field_guess = models.Gaussian1D(
        amplitude=field_amplitude_guess,
        mean=field_mean_guess,
        stddev=40,
        bounds={
            "amplitude": (0, None),
            "mean": (-150, 150),
            "stddev": (10, 100),
        },
    )

    initial_model = cluster_guess + field_guess

    fitter = fitting.TRFLSQFitter()

    fitted_model = fitter(
        initial_model,
        bin_centers,
        counts,
    )

    return (
        counts,
        bin_centers,
        fitted_model,
    )


def main():
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    catalog_data = {}
    ln_k_data = {}

    for cluster_name, catalog_path in catalogs.items():
        catalog = Table.read(catalog_path)

        catalog_data[cluster_name] = catalog

        ln_k_data[cluster_name] = collect_ln_k_values(
            cluster_name,
            catalog,
        )

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

    cut_90 = np.percentile(
        all_ln_k_values,
        90,
    )

    print()
    print(f"Combined 90th-percentile cut: {cut_90:.3f}")

    plot_data = {}

    for cluster_name in catalogs:
        rv_values = get_weighted_mean_rvs(
            catalog_data[cluster_name],
            ln_k_data[cluster_name],
            cut_90,
        )

        counts, bin_centers, fitted_model = (
            fit_double_gaussian(rv_values)
        )

        x_fit = np.linspace(
            rv_bins[0],
            rv_bins[-1],
            2000,
        )

        cluster_fit = fitted_model[0](x_fit)
        field_fit = fitted_model[1](x_fit)

        plot_data[cluster_name] = {
            "rv_values": rv_values,
            "counts": counts,
            "bin_centers": bin_centers,
            "fitted_model": fitted_model,
            "x_fit": x_fit,
            "cluster_fit": cluster_fit,
            "field_fit": field_fit,
        }

        print()
        print(cluster_name)
        print(f"Stars in histogram: {len(rv_values)}")
        print(
            "Cluster mean: "
            f"{fitted_model[0].mean.value:.3f} km/s"
        )
        print(
            "Cluster standard deviation: "
            f"{fitted_model[0].stddev.value:.3f} km/s"
        )
        print(
            "Field mean: "
            f"{fitted_model[1].mean.value:.3f} km/s"
        )
        print(
            "Field standard deviation: "
            f"{fitted_model[1].stddev.value:.3f} km/s"
        )

    y_max = 0

    for cluster_name in plot_data:
        data = plot_data[cluster_name]

        y_max = max(
            y_max,
            np.max(data["counts"]),
            np.max(data["cluster_fit"]),
            np.max(data["field_fit"]),
        )

    y_max = int(np.ceil(y_max * 1.1))

    for cluster_name in catalogs:
        data = plot_data[cluster_name]

        fig, ax = plt.subplots(
            figsize=(8, 6),
        )

        ax.hist(
            data["rv_values"],
            bins=rv_bins,
            histtype="step",
            color="black",
            linewidth=1.2,
            label="RV histogram",
        )

        ax.plot(
            data["x_fit"],
            data["field_fit"],
            color="blue",
            linestyle=":",
            linewidth=2,
            label="Field fit",
        )

        ax.plot(
            data["x_fit"],
            data["cluster_fit"],
            color="red",
            linestyle=":",
            linewidth=2,
            label="Cluster fit",
        )

        ax.set_xlim(
            rv_bins[0],
            rv_bins[-1],
        )

        ax.set_ylim(
            0,
            y_max,
        )

        ax.set_xlabel("RV (km/s)")
        ax.set_ylabel("N Stars")
        ax.set_title(
            f"{cluster_name} - 90th percentile cut"
        )

        ax.legend()

        fig.tight_layout()

        filename = (
            f"{cluster_name}_RVhist_"
            "90percent_5_40_nonjitter_double_gaussian.png"
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


if __name__ == "__main__":
    main()
