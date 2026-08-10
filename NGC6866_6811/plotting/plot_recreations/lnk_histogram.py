import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table
from thejoker import JokerSamples


workpath = "/data/labs/douglste-laf-lab/wichmand/stardata"

catalogs = {
    "NGC6811": os.path.join(workpath, "rcat_ngc6811_v0.fits"),
    "NGC6866": os.path.join(workpath, "rcat_ngc6866_v0.fits"),
}

output_dir = "/data/labs/douglste-laf-lab/wichmand/plots/plot_recreations"

x_min = -7
x_max = 6
bins = np.linspace(x_min, x_max, 30)


def get_star_ids(catalog_path):
    catalog = Table.read(catalog_path)

    star_ids = []

    for value in catalog["GAIAEDR3_ID"]:
        try:
            star_id = int(value)
        except (TypeError, ValueError, OverflowError):
            continue

        if star_id > 0:
            star_ids.append(star_id)

    return np.unique(star_ids)


def get_sample_file(star_id, jitter):
    if jitter:
        star_folder = os.path.join(
            workpath,
            "200.0M_jitter",
            str(star_id),
        )

        mcmc_file = os.path.join(
            star_folder,
            f"rejection_samples_MCMC_adapt_full_200M_jitter_{star_id}.hdf5",
        )

        rejection_file = os.path.join(
            star_folder,
            f"rejection_samples_200M_jitter_{star_id}.hdf5",
        )

    else:
        star_folder = os.path.join(
            workpath,
            "200.0M_new",
            str(star_id),
        )

        mcmc_file = os.path.join(
            star_folder,
            f"rejection_samples_MCMC_adapt_full_200.0M_{star_id}_new.hdf5",
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
        k_values = np.asarray(samples["K"], dtype=float)

    k_values = np.asarray(k_values, dtype=float)
    k_values = k_values[
        np.isfinite(k_values) & (k_values > 0)
    ]

    if len(k_values) == 0:
        raise ValueError("No valid K values")

    ln_k_values = np.log(k_values)

    return np.percentile(ln_k_values, 1)


def collect_values(cluster_name, catalog_path, jitter):
    star_ids = get_star_ids(catalog_path)

    values = []

    mcmc_count = 0
    rejection_count = 0
    missing_count = 0
    failed_count = 0

    for star_id in star_ids:
        sample_file, run_type = get_sample_file(
            star_id,
            jitter,
        )

        if sample_file is None:
            missing_count += 1
            continue

        try:
            ln_k = get_ln_k(sample_file)
        except Exception as error:
            print(f"Failed {star_id}: {error}")
            failed_count += 1
            continue

        values.append(ln_k)

        if run_type == "MCMC":
            mcmc_count += 1
        else:
            rejection_count += 1

    values = np.asarray(values, dtype=float)

    run_name = "Jitter" if jitter else "Non-jitter"

    print()
    print(f"{cluster_name} - {run_name}")
    print(f"Catalog stars: {len(star_ids)}")
    print(f"Included: {len(values)}")
    print(f"MCMC: {mcmc_count}")
    print(f"Rejection: {rejection_count}")
    print(f"Missing: {missing_count}")
    print(f"Failed: {failed_count}")
    print(
        f"Outside histogram range: "
        f"{np.sum((values < x_min) | (values > x_max))}"
    )

    return values


def make_histogram(ngc6811_values, ngc6866_values, jitter):
    if len(ngc6811_values) == 0 or len(ngc6866_values) == 0:
        print("One of the clusters has no values")
        return

    os.makedirs(output_dir, exist_ok=True)

    all_values = np.concatenate(
        (ngc6811_values, ngc6866_values)
    )

    percentile_99, percentile_90, percentile_50 = np.percentile(
        all_values,
        [99, 90, 50],
    )

    plt.figure(figsize=(8, 6))

    plt.hist(
        [ngc6811_values, ngc6866_values],
        bins=bins,
        density=True,
        histtype="barstacked",
        label=["NGC6811", "NGC6866"],
    )

    ax = plt.gca()

    ax.axvline(
        percentile_99,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=f"99th percentile: {percentile_99:.2f}",
    )

    ax.axvline(
        percentile_90,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"90th percentile: {percentile_90:.2f}",
    )

    ax.axvline(
        percentile_50,
        color="black",
        linestyle="-.",
        linewidth=1.5,
        label=f"50th percentile: {percentile_50:.2f}",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_yscale("log")

    ax.set_xlabel(
        r"$\ln\left(K/\mathrm{km\,s^{-1}}\right)$"
    )
    ax.set_ylabel("Density")

    if jitter:
        ax.set_title("Jitter")
        filename = "lnk_histogram_jitter.png"
    else:
        ax.set_title("Non-jitter")
        filename = "lnk_histogram_nonjitter.png"

    ax.legend()

    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        filename,
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print(f"99th percentile: {percentile_99:.3f}")
    print(f"90th percentile: {percentile_90:.3f}")
    print(f"50th percentile: {percentile_50:.3f}")
    print(f"Saved: {output_path}")


def main():
    jitter_values = {}
    nonjitter_values = {}

    for cluster_name, catalog_path in catalogs.items():
        jitter_values[cluster_name] = collect_values(
            cluster_name,
            catalog_path,
            jitter=True,
        )

        nonjitter_values[cluster_name] = collect_values(
            cluster_name,
            catalog_path,
            jitter=False,
        )

    make_histogram(
        jitter_values["NGC6811"],
        jitter_values["NGC6866"],
        jitter=True,
    )

    make_histogram(
        nonjitter_values["NGC6811"],
        nonjitter_values["NGC6866"],
        jitter=False,
    )


if __name__ == "__main__":
    main()
