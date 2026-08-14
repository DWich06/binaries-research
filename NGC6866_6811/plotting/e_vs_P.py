import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from astropy.table import Table
from thejoker import JokerSamples
from sklearn.cluster import KMeans


workpath = "/data/labs/douglaslab/wichmand/stardata"

modality_file = os.path.join(
    workpath,
    "analysis_csvs",
    "csvruns",
    "all_runs_sample_count_modality.csv",
)

orbital_file = os.path.join(
    workpath,
    "analysis_csvs",
    "table_recreations",
    "joker_orbital_parameters.csv",
)

joker_dir = os.path.join(
    workpath,
    "joker_runs",
    "200.0M_jitter",
)

output_dir = "/data/labs/douglaslab/wichmand/plots"

output_file = os.path.join(
    output_dir,
    "e_vs_P_unimodal_bimodal_members_200M_jitter.png",
)


def valid_string(value):

    if np.ma.is_masked(value):
        return None

    value = str(value).strip()

    if value in [
        "",
        "--",
        "nan",
        "None",
    ]:
        return None

    return value


def get_selected_modality(row):

    mcmc_modality = valid_string(
        row["Modality_200_Jitter_MCMC"]
    )

    if mcmc_modality is not None:
        return mcmc_modality, "MCMC"

    rejection_modality = valid_string(
        row["Modality_200_Jitter"]
    )

    return rejection_modality, "Rejection"


def get_sample_file(
    star_id,
    run_type,
):

    star_folder = os.path.join(
        joker_dir,
        str(star_id),
    )

    if run_type == "MCMC":

        filename = os.path.join(
            star_folder,
            (
                "rejection_samples_MCMC_adapt_full_"
                f"200M_jitter_{star_id}.hdf5"
            ),
        )

        if os.path.exists(filename):
            return filename

        return None

    filename = os.path.join(
        star_folder,
        (
            "rejection_samples_200M_jitter_"
            f"{star_id}.hdf5"
        ),
    )

    if os.path.exists(filename):
        return filename

    return None


def summarize_samples(samples):

    P = samples[
        "P"
    ].to_value(
        u.day
    )

    e = np.asarray(
        samples["e"],
        dtype=float,
    )

    valid = (
        np.isfinite(P)
        & np.isfinite(e)
        & (P > 0)
    )

    P = P[valid]
    e = e[valid]

    if len(P) == 0:
        return None

    P16, P50, P84 = np.percentile(
        P,
        [16, 50, 84],
    )

    e16, e50, e84 = np.percentile(
        e,
        [16, 50, 84],
    )

    return {
        "P": P50,
        "P_dn": P50 - P16,
        "P_up": P84 - P50,
        "e": e50,
        "e_dn": e50 - e16,
        "e_up": e84 - e50,
    }


def split_bimodal_samples(samples):

    P = samples[
        "P"
    ].to_value(
        u.day
    )

    valid = (
        np.isfinite(P)
        & (P > 0)
    )

    samples = samples[
        valid
    ]

    P = samples[
        "P"
    ].to_value(
        u.day
    )

    lnP = np.log(
        P
    ).reshape(
        -1,
        1,
    )

    clf = KMeans(
        n_clusters=2,
        random_state=0,
        n_init=10,
    )

    labels = clf.fit_predict(
        lnP
    )

    modes = []

    for label in np.unique(
        labels
    ):

        mode_samples = samples[
            labels == label
        ]

        summary = summarize_samples(
            mode_samples
        )

        if summary is None:
            continue

        modes.append(
            summary
        )

    modes.sort(
        key=lambda x: x["P"]
    )

    return modes


def main():

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    modality_table = Table.read(
        modality_file
    )

    orbital_table = Table.read(
        orbital_file
    )

    member_ids = set()

    for row in orbital_table:

        try:
            star_id = int(
                row["GAIAEDR3_ID"]
            )

            cluster = str(
                row["Cluster"]
            ).strip()

            member = int(
                float(
                    row["Member"]
                )
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            continue

        if member == 1:

            member_ids.add(
                (
                    star_id,
                    cluster,
                )
            )

    plot_rows = []

    unimodal_count = 0
    bimodal_star_count = 0
    bimodal_point_count = 0
    missing_count = 0
    failed_count = 0

    for row in modality_table:

        modality, run_type = (
            get_selected_modality(
                row
            )
        )

        if modality not in [
            "Unimodal",
            "Bimodal",
        ]:
            continue

        try:
            star_id = int(
                row["GAIAEDR3_ID"]
            )

            cluster = str(
                row["Cluster"]
            ).strip()

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            continue

        if (
            star_id,
            cluster,
        ) not in member_ids:
            continue

        sample_file = get_sample_file(
            star_id,
            run_type,
        )

        if sample_file is None:

            print(
                f"Missing {star_id} "
                f"({run_type})"
            )

            missing_count += 1
            continue

        try:

            samples = JokerSamples.read(
                sample_file
            )

            if modality == "Unimodal":

                summary = summarize_samples(
                    samples
                )

                if summary is None:
                    continue

                plot_rows.append(
                    {
                        "star_id": star_id,
                        "modality": "Unimodal",
                        **summary,
                    }
                )

                unimodal_count += 1

            elif modality == "Bimodal":

                modes = split_bimodal_samples(
                    samples
                )

                if len(modes) != 2:

                    print(
                        f"{star_id}: "
                        f"expected 2 modes, "
                        f"found {len(modes)}"
                    )

                    continue

                for summary in modes:

                    plot_rows.append(
                        {
                            "star_id": star_id,
                            "modality": "Bimodal",
                            **summary,
                        }
                    )

                bimodal_star_count += 1
                bimodal_point_count += 2

        except Exception as error:

            print(
                f"Failed {star_id}: "
                f"{error}"
            )

            failed_count += 1

    print()
    print(
        f"Unimodal member stars: "
        f"{unimodal_count}"
    )

    print(
        f"Bimodal member stars: "
        f"{bimodal_star_count}"
    )

    print(
        f"Bimodal member points: "
        f"{bimodal_point_count}"
    )

    print(
        f"Total plotted points: "
        f"{len(plot_rows)}"
    )

    print(
        f"Missing files: "
        f"{missing_count}"
    )

    print(
        f"Failed: "
        f"{failed_count}"
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    unimodal_rows = [
        row
        for row in plot_rows
        if row["modality"] == "Unimodal"
    ]

    bimodal_rows = [
        row
        for row in plot_rows
        if row["modality"] == "Bimodal"
    ]

    if len(unimodal_rows) > 0:

        P = np.array(
            [
                row["P"]
                for row in unimodal_rows
            ]
        )

        e = np.array(
            [
                row["e"]
                for row in unimodal_rows
            ]
        )

        P_dn = np.array(
            [
                row["P_dn"]
                for row in unimodal_rows
            ]
        )

        P_up = np.array(
            [
                row["P_up"]
                for row in unimodal_rows
            ]
        )

        e_dn = np.array(
            [
                row["e_dn"]
                for row in unimodal_rows
            ]
        )

        e_up = np.array(
            [
                row["e_up"]
                for row in unimodal_rows
            ]
        )

        ax.errorbar(
            P,
            e,
            xerr=np.vstack(
                (
                    P_dn,
                    P_up,
                )
            ),
            yerr=np.vstack(
                (
                    e_dn,
                    e_up,
                )
            ),
            fmt="o",
            color="tab:blue",
            linestyle="none",
            markersize=5,
            capsize=2,
            label="Unimodal",
        )

    if len(bimodal_rows) > 0:

        P = np.array(
            [
                row["P"]
                for row in bimodal_rows
            ]
        )

        e = np.array(
            [
                row["e"]
                for row in bimodal_rows
            ]
        )

        P_dn = np.array(
            [
                row["P_dn"]
                for row in bimodal_rows
            ]
        )

        P_up = np.array(
            [
                row["P_up"]
                for row in bimodal_rows
            ]
        )

        e_dn = np.array(
            [
                row["e_dn"]
                for row in bimodal_rows
            ]
        )

        e_up = np.array(
            [
                row["e_up"]
                for row in bimodal_rows
            ]
        )

        ax.errorbar(
            P,
            e,
            xerr=np.vstack(
                (
                    P_dn,
                    P_up,
                )
            ),
            yerr=np.vstack(
                (
                    e_dn,
                    e_up,
                )
            ),
            fmt="o",
            color="tab:orange",
            markerfacecolor="none",
            linestyle="none",
            markersize=6,
            capsize=2,
            label="Bimodal",
        )

    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        "P (days)"
    )

    ax.set_ylabel(
        "e"
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_title(
        "e vs P for NGC6811 and NGC6866"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print()
    print(
        f"Saved: {output_file}"
    )


if __name__ == "__main__":
    main()
