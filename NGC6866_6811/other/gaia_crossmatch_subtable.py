import os
import pandas as pd

stardata = "/data/labs/douglste-laf-lab/wichmand/stardata"

input_file = os.path.join(
    stardata,
    "rcat_gaia_crossmatch.csv"
)

output_file = os.path.join(
    stardata,
    "nss_gaia_crossmatch.csv"
)

print("Reading crossmatch table...", flush=True)

df = pd.read_csv(input_file)

print(f"Total rows: {len(df)}", flush=True)

nss_df = df[
    df["Catalog_Match"] == "I355 and I357"
].copy()

nss_df = nss_df.sort_values(
    by=["Cluster", "GAIAEDR3_ID"]
).reset_index(drop=True)

print(f"NSS rows: {len(nss_df)}", flush=True)

nss_df.to_csv(
    output_file,
    index=False
)

print(f"\nSaved to:\n{output_file}", flush=True)
