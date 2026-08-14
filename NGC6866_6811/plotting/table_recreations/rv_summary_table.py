import os
import numpy as np
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

output_dir = os.path.join(
    workpath,
    "analysis_csvs",
)


def weighted_mean_and_error(values, errors):

    values = np.asarray(
        values,
        dtype=float,
    )

    errors = np.asarray(
        errors,
        dtype=float,
    )

    valid = (
        np.isfinite(values)
        & np.isfinite(errors)
        & (errors > 0)
    )

    values = values[valid]
    errors = errors[valid]

    if len(values) == 0:
        return np.nan, np.nan, 0

    weights = 1 / errors**2

    weighted_mean = np.average(
        values,
        weights=weights,
    )

    weighted_error = np.sqrt(
        1 / np.sum(weights)
    )

    return (
        weighted_mean,
        weighted_error,
        len(values),
    )


def make_summary_table(
    cluster_name,
    catalog_path,
):

    catalog = Table.read(
        catalog_path
    )

    star_ids = np.unique(
        catalog["GAIAEDR3_ID"]
    )

    rows_out = []

    for star_id in star_ids:

        rows = catalog[
            catalog["GAIAEDR3_ID"] == star_id
        ]

        first_row = rows[0]

        ra = float(
            first_row["GAIAEDR3_RA"]
        )

        dec = float(
            first_row["GAIAEDR3_DEC"]
        )

        g_mag = float(
            first_row["GAIAEDR3_G"]
        )

        bp = float(
            first_row["GAIAEDR3_BP"]
        )

        rp = float(
            first_row["GAIAEDR3_RP"]
        )

        if (
            np.isfinite(bp)
            and np.isfinite(rp)
        ):
            bp_rp = bp - rp
        else:
            bp_rp = np.nan

        hdbscan_cluster = float(
            first_row["HDBscan_Cluster"]
        )

        (
            weighted_rv,
            weighted_rv_err,
            nobs,
        ) = weighted_mean_and_error(
            rows["vrad"],
            rows["vrad_err"],
        )

        (
            weighted_vsini,
            weighted_vsini_err,
            _,
        ) = weighted_mean_and_error(
            rows["vstar"],
            rows["vstar_err"],
        )

        rows_out.append(
            (
                int(star_id),

                (
                    round(ra, 6)
                    if np.isfinite(ra)
                    else np.nan
                ),

                (
                    round(dec, 6)
                    if np.isfinite(dec)
                    else np.nan
                ),

                (
                    round(g_mag, 3)
                    if np.isfinite(g_mag)
                    else np.nan
                ),

                (
                    round(bp_rp, 3)
                    if np.isfinite(bp_rp)
                    else np.nan
                ),

                nobs,

                (
                    round(weighted_rv, 3)
                    if np.isfinite(weighted_rv)
                    else np.nan
                ),

                (
                    round(weighted_rv_err, 3)
                    if np.isfinite(weighted_rv_err)
                    else np.nan
                ),

                (
                    round(weighted_vsini, 3)
                    if np.isfinite(weighted_vsini)
                    else np.nan
                ),

                (
                    round(weighted_vsini_err, 3)
                    if np.isfinite(weighted_vsini_err)
                    else np.nan
                ),

                hdbscan_cluster,
            )
        )

    output_table = Table(
        rows=rows_out,
        names=[
            "GAIAEDR3_ID",
            "RA (deg)",
            "DEC (deg)",
            "G (mag)",
            "BP-RP (mag)",
            "Nobs",
            "RV (km/s)",
            "RV_e (km/s)",
            "vsini (km/s)",
            "vsini_e (km/s)",
            "HDBscan_Cluster",
        ],
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    filename = (
        f"{cluster_name}_rv_summary_table.csv"
    )

    output_path = os.path.join(
        output_dir,
        filename,
    )

    output_table.write(
        output_path,
        format="ascii.csv",
        overwrite=True,
    )

    print()
    print(cluster_name)
    print(
        f"Stars in table: "
        f"{len(output_table)}"
    )
    print(
        f"Saved: {output_path}"
    )


def main():

    for (
        cluster_name,
        catalog_path,
    ) in catalogs.items():

        make_summary_table(
            cluster_name,
            catalog_path,
        )


if __name__ == "__main__":
    main()
