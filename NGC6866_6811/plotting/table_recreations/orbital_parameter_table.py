import os
import numpy as np
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

output_dir = os.path.join(
    workpath,
    "analysis_csvs",
    "table_recreations",
)

output_file = os.path.join(
    output_dir,
    "joker_orbital_parameters.csv",
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
        return mcmc_file

    if os.path.exists(rejection_file):
        return rejection_file

    return None


def get_percentiles(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    p16, p50, p84 = np.percentile(
        values,
        [16, 50, 84],
    )

    median = p50
    e_dn = p50 - p16
    e_up = p84 - p50

    return (
        median,
        e_dn,
        e_up,
    )


def get_angle_percentiles(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    # Circular mean used only as the
    # reference point for unwrapping
    center = np.arctan2(
        np.mean(np.sin(values)),
        np.mean(np.cos(values)),
    )

    # Measure each angle relative to
    # the circular center and wrap the
    # difference into [-pi, pi)
    shifted = np.arctan2(
        np.sin(values - center),
        np.cos(values - center),
    )

    p16, p50, p84 = np.percentile(
        shifted,
        [16, 50, 84],
    )

    # Convert the median back to its
    # actual position on the circle
    median = center + p50

    # Wrap final median into [-pi, pi)
    median = np.arctan2(
        np.sin(median),
        np.cos(median),
    )

    e_dn = p50 - p16
    e_up = p84 - p50

    # Convert radians to degrees
    median = np.degrees(median)
    e_dn = np.degrees(e_dn)
    e_up = np.degrees(e_up)

    return (
        median,
        e_dn,
        e_up,
    )


def get_orbital_parameters(sample_file):

    samples = JokerSamples.read(
        sample_file
    )

    if len(samples) == 0:
        raise ValueError(
            "No Joker samples"
        )

    # Force K to the positive representation.
    # wrap_K also shifts omega consistently
    # when K changes sign.
    wrapped_samples = samples.wrap_K()

    # Handle versions where wrap_K returns
    # a new JokerSamples object versus
    # modifying the existing one in place.
    if wrapped_samples is not None:
        samples = wrapped_samples

    # --------------------------
    # Period
    # --------------------------

    P, e_dn_P, e_up_P = get_percentiles(
        samples["P"].to_value(
            u.day
        )
    )

    # --------------------------
    # Systemic velocity
    # --------------------------

    v0, e_dn_v0, e_up_v0 = get_percentiles(
        samples["v0"].to_value(
            u.km / u.s
        )
    )

    # --------------------------
    # RV semi-amplitude
    # --------------------------

    K, e_dn_K, e_up_K = get_percentiles(
        samples["K"].to_value(
            u.km / u.s
        )
    )

    # --------------------------
    # Eccentricity
    # --------------------------

    e, e_dn_e, e_up_e = get_percentiles(
        samples["e"]
    )

    # --------------------------
    # Argument of periastron
    # --------------------------

    omega_values = samples[
        "omega"
    ].to_value(
        u.rad
    )

    (
        omega,
        e_dn_omega,
        e_up_omega,
    ) = get_angle_percentiles(
        omega_values
    )

    # --------------------------
    # Mean anomaly
    # --------------------------

    M0_values = samples[
        "M0"
    ].to_value(
        u.rad
    )

    (
        M0,
        e_dn_M0,
        e_up_M0,
    ) = get_angle_percentiles(
        M0_values
    )

    # --------------------------
    # Jitter
    # --------------------------

    s, e_dn_s, e_up_s = get_percentiles(
        samples["s"].to_value(
            u.km / u.s
        )
    )

    return (
        P,
        e_dn_P,
        e_up_P,

        v0,
        e_dn_v0,
        e_up_v0,

        K,
        e_dn_K,
        e_up_K,

        e,
        e_dn_e,
        e_up_e,

        omega,
        e_dn_omega,
        e_up_omega,

        M0,
        e_dn_M0,
        e_up_M0,

        s,
        e_dn_s,
        e_up_s,
    )


def main():

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    rows_out = []

    missing_count = 0
    failed_count = 0

    for (
        cluster_name,
        catalog_path,
    ) in catalogs.items():

        print()
        print(cluster_name)

        catalog = Table.read(
            catalog_path
        )

        star_ids = np.unique(
            catalog["GAIAEDR3_ID"]
        )

        cluster_count = 0

        for star_id in star_ids:

            try:
                star_id = int(
                    star_id
                )

            except (
                TypeError,
                ValueError,
                OverflowError,
            ):
                continue

            if star_id <= 0:
                continue

            sample_file = get_sample_file(
                star_id
            )

            if sample_file is None:
                missing_count += 1
                continue

            star_rows = catalog[
                catalog["GAIAEDR3_ID"]
                == star_id
            ]

            if len(star_rows) == 0:
                continue

            first_row = star_rows[0]

            try:
                member = float(
                    first_row["MemBool"]
                )

            except (
                TypeError,
                ValueError,
            ):
                member = np.nan

            try:

                parameters = (
                    get_orbital_parameters(
                        sample_file
                    )
                )

            except Exception as error:

                print(
                    f"Failed {star_id}: "
                    f"{error}"
                )

                failed_count += 1
                continue

            rows_out.append(
                (
                    star_id,
                    cluster_name,
                    member,
                    *parameters,
                )
            )

            cluster_count += 1

        print(
            f"Stars added: "
            f"{cluster_count}"
        )

    output_table = Table(
        rows=rows_out,
        names=[
            "GAIAEDR3_ID",
            "Cluster",
            "Member",

            "P (days)",
            "e_dn_P (days)",
            "e_up_P (days)",

            "v0 (km/s)",
            "e_dn_v0 (km/s)",
            "e_up_v0 (km/s)",

            "K (km/s)",
            "e_dn_K (km/s)",
            "e_up_K (km/s)",

            "e",
            "e_dn_e",
            "e_up_e",

            "omega (deg)",
            "e_dn_omega (deg)",
            "e_up_omega (deg)",

            "M0 (deg)",
            "e_dn_M0 (deg)",
            "e_up_M0 (deg)",

            "s (km/s)",
            "e_dn_s (km/s)",
            "e_up_s (km/s)",
        ],
    )

    output_table.sort(
        [
            "Cluster",
            "GAIAEDR3_ID",
        ]
    )

    output_table.write(
        output_file,
        format="ascii.csv",
        overwrite=True,
    )

    print()
    print("-----------------------------")
    print("Finished")
    print("-----------------------------")

    print(
        f"Stars in table: "
        f"{len(output_table)}"
    )

    print(
        f"Missing Joker files: "
        f"{missing_count}"
    )

    print(
        f"Failed: "
        f"{failed_count}"
    )

    print()

    print(
        f"Saved: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()
