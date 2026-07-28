import os
import pandas as pd
from astropy.table import QTable

stardata = "/data/labs/douglaslab/wichmand/stardata"
csvdir = f"{stardata}/csvruns"

idlist_path = "/data2/labs/douglste-laf-lab/mathewea/Summer-Research/GAIADR3_IDs.csv"

catalog_6811_path = f"{stardata}/rcat_ngc6811_v0.fits"
catalog_6866_path = f"{stardata}/rcat_ngc6866_v0.fits"

output_file = f"{csvdir}/all_runs_sample_count_modality.csv"

runs = {
    "200": {
        "file": "bimodalcheck_200_forcsv.csv",
        "count_column": "Count_200",
        "modality_column": "Modality_200",
    },
    "200_MCMC": {
        "file": "bimodalcheck_200_mcmc_forcsv.csv",
        "count_column": "Count_200_MCMC",
        "modality_column": "Modality_200_MCMC",
    },
    "200_MCMC_Adapt": {
        "file": "bimodalcheck_200_mcmc_adapt_full_forcsv.csv",
        "count_column": "Count_200_MCMC_Adapt",
        "modality_column": "Modality_200_MCMC_Adapt",
    },
    "200_Jitter": {
        "file": "bimodalcheck_200_jitter_forcsv.csv",
        "count_column": "Count_200_Jitter",
        "modality_column": "Modality_200_Jitter",
    },
    "200_Jitter_MCMC": {
        "file": "bimodalcheck_200_jitter_mcmc_forcsv.csv",
        "count_column": "Count_200_Jitter_MCMC",
        "modality_column": "Modality_200_Jitter_MCMC",
    },
    "400_Jitter": {
        "file": "bimodalcheck_400_jitter_forcsv.csv",
        "count_column": "Count_400_Jitter",
        "modality_column": "Modality_400_Jitter",
    },
}

id_table = QTable.read(idlist_path)

summary_table = pd.DataFrame({
    "GAIAEDR3_ID": [int(x) for x in id_table["GAIAEDR3_ID"]]
})

summary_table = summary_table.drop_duplicates(
    subset="GAIAEDR3_ID"
).reset_index(drop=True)

cat_6811 = QTable.read(catalog_6811_path)
cat_6866 = QTable.read(catalog_6866_path)

ids_6811 = {int(x) for x in cat_6811["GAIAEDR3_ID"]}
ids_6866 = {int(x) for x in cat_6866["GAIAEDR3_ID"]}

def get_cluster(gaia_id):
    if gaia_id in ids_6811:
        return "NGC6811"
    if gaia_id in ids_6866:
        return "NGC6866"
    return "Unknown"

summary_table["Cluster"] = summary_table["GAIAEDR3_ID"].apply(get_cluster)

for run_name, run_info in runs.items():

    filepath = os.path.join(csvdir, run_info["file"])

    print(f"Reading {run_name}")

    if not os.path.exists(filepath):
        summary_table[run_info["count_column"]] = pd.NA
        summary_table[run_info["modality_column"]] = pd.NA
        continue

    run_df = pd.read_csv(filepath)

    run_df = run_df[
        [
            "GAIAEDR3_ID",
            "sample_count",
            "modality",
        ]
    ].copy()

    run_df = run_df.rename(
        columns={
            "sample_count": run_info["count_column"],
            "modality": run_info["modality_column"],
        }
    )

    run_df = run_df.drop_duplicates(subset="GAIAEDR3_ID")

    summary_table = summary_table.merge(
        run_df,
        on="GAIAEDR3_ID",
        how="left",
    )

summary_table = summary_table.dropna(
    subset=[
        "Count_200",
        "Count_200_MCMC",
        "Count_200_MCMC_Adapt",
        "Count_200_Jitter",
        "Count_200_Jitter_MCMC",
        "Count_400_Jitter",
    ],
    how="all",
)

summary_table = summary_table[
    [
        "GAIAEDR3_ID",
        "Cluster",
        "Count_200",
        "Modality_200",
        "Count_200_MCMC",
        "Modality_200_MCMC",
        "Count_200_MCMC_Adapt",
        "Modality_200_MCMC_Adapt",
        "Count_200_Jitter",
        "Modality_200_Jitter",
        "Count_200_Jitter_MCMC",
        "Modality_200_Jitter_MCMC",
        "Count_400_Jitter",
        "Modality_400_Jitter",
    ]
]

summary_table.to_csv(
    output_file,
    index=False,
    na_rep="",
)

print(f"Wrote {output_file}")
print(f"Total stars: {len(summary_table)}")
