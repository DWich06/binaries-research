import os
import glob
import pandas as pd
from astropy.table import QTable

stardata = "/data/labs/douglste-laf-lab/wichmand/stardata"

cat_6811 = QTable.read(
    os.path.join(stardata, "rcat_ngc6811_v0.fits")
)

cat_6866 = QTable.read(
    os.path.join(stardata, "rcat_ngc6866_v0.fits")
)

input_file = os.path.join(
    stardata,
    "All_NSS_Matches.csv"
)

output_file = os.path.join(
    stardata,
    "NSS_TheJoker_Summary.csv"
)


def clean_id(value):
    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


def clean_value(value):
    value = str(value).strip()

    if value.lower() in [
        "",
        "nan",
        "none",
        "--",
        "no data",
    ]:
        return None

    return value


def first_available_value(group, possible_columns):
    column_lookup = {
        str(column).strip().lower(): column
        for column in group.columns
    }

    for possible_column in possible_columns:
        actual_column = column_lookup.get(
            possible_column.lower()
        )

        if actual_column is None:
            continue

        for value in group[actual_column]:
            cleaned = clean_value(value)

            if cleaned is not None:
                return cleaned

    return ""


def combine_unique_values(group, column_name):
    if column_name not in group.columns:
        return ""

    values = []

    for value in group[column_name]:
        cleaned = clean_value(value)

        if cleaned is not None and cleaned not in values:
            values.append(cleaned)

    return "; ".join(values)

def get_num_rvs(star_id, cluster):
    if cluster == "NGC6811":
        catalog = cat_6811
    elif cluster == "NGC6866":
        catalog = cat_6866
    else:
        return 0

    return sum(
        str(value).strip() == star_id
        for value in catalog["GAIAEDR3_ID"]
    )

def file_exists(pattern):
    return len(glob.glob(pattern)) > 0


def find_thejoker_runs(star_id):
    runs = []

    new_200_dir = os.path.join(
        stardata,
        "200.0M_new",
        star_id
    )

    jitter_200_dir = os.path.join(
        stardata,
        "200.0M_jitter",
        star_id
    )

    jitter_400_dir = os.path.join(
        stardata,
        "400.0M_jitter",
        star_id
    )

    rejection_200 = os.path.join(
        new_200_dir,
        f"rejection_samples_200.0M_{star_id}_new.hdf5"
    )

    mcmc_200 = os.path.join(
        new_200_dir,
        f"rejection_samples_MCMC_200.0M_{star_id}_new.hdf5"
    )

    mcmc_adapt_200 = os.path.join(
        new_200_dir,
        f"rejection_samples_MCMC_adapt_full_200.0M_{star_id}_new.hdf5"
    )

    rejection_200_jitter = os.path.join(
        jitter_200_dir,
        f"rejection_samples_200M_jitter_{star_id}.hdf5"
    )

    mcmc_200_jitter_patterns = [
        os.path.join(
            jitter_200_dir,
            f"rejection_samples_MCMC_200M_jitter_{star_id}.hdf5"
        ),
        os.path.join(
            jitter_200_dir,
            f"rejection_samples_MCMC_adapt_full_200M_jitter_{star_id}.hdf5"
        ),
        os.path.join(
            jitter_200_dir,
            f"*MCMC*{star_id}*.hdf5"
        ),
    ]

    rejection_400_jitter_patterns = [
        os.path.join(
            jitter_400_dir,
            f"rejection_samples_400M_jitter_{star_id}.hdf5"
        ),
        os.path.join(
            jitter_400_dir,
            f"*400M*jitter*{star_id}*.hdf5"
        ),
        os.path.join(
            jitter_400_dir,
            f"*{star_id}*.hdf5"
        ),
    ]

    if os.path.exists(rejection_200):
        runs.append("200M")

    if os.path.exists(mcmc_200):
        runs.append("200M_MCMC")

    if os.path.exists(mcmc_adapt_200):
        runs.append("200M_MCMC_Adapt_Full")

    if os.path.exists(rejection_200_jitter):
        runs.append("200M_Jitter")

    if any(file_exists(pattern) for pattern in mcmc_200_jitter_patterns):
        runs.append("200M_Jitter_MCMC")

    if any(file_exists(pattern) for pattern in rejection_400_jitter_patterns):
        runs.append("400M_Jitter")

    return runs


data = pd.read_csv(
    input_file,
    dtype=str,
    keep_default_na=False
)

print(f"Read {len(data)} NSS match rows")
print(f"Columns found: {list(data.columns)}")


if "Source" in data.columns:
    id_column = "Source"

elif "GAIAEDR3_ID" in data.columns:
    id_column = "GAIAEDR3_ID"

elif "GAIADR3_ID" in data.columns:
    id_column = "GAIADR3_ID"

else:
    raise KeyError(
        "Could not find a Gaia source ID column.\n"
        f"Available columns: {list(data.columns)}"
    )


data[id_column] = data[id_column].apply(clean_id)

data = data[
    data[id_column] != ""
].copy()


period_columns = [
    "Period",
    "period",
    "Per",
    "P",
    "P1",
    "OrbitalPeriod",
    "orbital_period",
]

eccentricity_columns = [
    "Eccentricity",
    "eccentricity",
    "Ecc",
    "ecc",
    "e",
]


summary_rows = []


for star_id, group in data.groupby(
    id_column,
    sort=False
):
    cluster = first_available_value(
        group,
        ["Cluster"]
    )

    num_rvs = get_num_rvs(star_id, cluster)

    nss_tables = combine_unique_values(
        group,
        "NSS_Table"
    )

    gaia_period = first_available_value(
        group,
        period_columns
    )

    gaia_eccentricity = first_available_value(
        group,
        eccentricity_columns
    )

    thejoker_runs = find_thejoker_runs(star_id)

    summary_rows.append(
        {
            "GAIAEDR3_ID": star_id,
            "Cluster": cluster,
            "NSS_Table": nss_tables,
            "Gaia_Period_days": gaia_period,
            "Gaia_Eccentricity": gaia_eccentricity,
            "N_RV_Measurements": num_rvs,
            "TheJoker_Runs": "; ".join(thejoker_runs),
        }
    )


summary = pd.DataFrame(summary_rows)

summary = summary.sort_values(
    by=[
        "Cluster",
        "GAIAEDR3_ID",
    ]
).reset_index(drop=True)

summary.to_csv(
    output_file,
    index=False
)



