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
    "table_recreations",
)


def make_rv_data_table(
    cluster_name,
    catalog_path,
):

    catalog = Table.read(
        catalog_path
    )

    rows_out = []

    for row in catalog:

        star_id = row["GAIAEDR3_ID"]
        mjd = row["MJD"]
        rv = row["vrad"]
        rv_err = row["vrad_err"]
        teff = row["Teff"]
        logg = row["log(g)"]
        vmic = row["vmic"]
        vstar = row["vstar"]
        vstar_err = row["vstar_err"]

        rows_out.append(
            (
                int(star_id),
                float(mjd),
                float(rv),
                float(rv_err),
                float(teff),
                float(logg),
                float(vmic),
                float(vstar),
                float(vstar_err),
            )
        )

    output_table = Table(
        rows=rows_out,
        names=[
            "Star ID",
            "MJD",
            "RV (km/s)",
            "e_RV (km/s)",
            "Teff (K)",
            "log(g)",
            "Vmic (km/s)",
            "Vstar (km/s)",
            "e_Vstar (km/s)",
        ],
    )

    # Sort first by Star ID, then chronologically by MJD
    output_table.sort(
        [
            "Star ID",
            "MJD",
        ]
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    filename = (
        f"{cluster_name}_rv_data_table.csv"
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
        f"Observations in table: "
        f"{len(output_table)}"
    )
    print(
        f"Unique stars: "
        f"{len(np.unique(output_table['Star ID']))}"
    )
    print(
        f"Saved: {output_path}"
    )


def main():

    for (
        cluster_name,
        catalog_path,
    ) in catalogs.items():

        make_rv_data_table(
            cluster_name,
            catalog_path,
        )


if __name__ == "__main__":
    main()
