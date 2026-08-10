import os
import glob
from io import StringIO

import pandas as pd
from astropy.table import QTable


stardata = "/data/labs/douglste-laf-lab/wichmand/stardata"

rcat_6811_file = os.path.join(stardata, "rcat_ngc6811_v0.fits")
rcat_6866_file = os.path.join(stardata, "rcat_ngc6866_v0.fits")

i357_6811_file = os.path.join(stardata, "NGC6811_nonsingle.tsv")
i357_6866_file = os.path.join(stardata, "NGC6866_nonsingle.tsv")

i355_6811_file = os.path.join(stardata, "NGC6811_i355.tsv")
i355_6866_file = os.path.join(stardata, "NGC6866_i355.tsv")

output_file = os.path.join(stardata, "rcat_gaia_crossmatch.csv")


def clean_id(value):
    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    if value.lower() in {"", "nan", "none", "--"}:
        return ""

    return value


def clean_value(value):
    value = str(value).strip()

    if value.lower() in {"", "nan", "none", "--", "no data"}:
        return ""

    return value


def first_nonblank(values):
    for value in values:
        value = clean_value(value)

        if value:
            return value

    return ""


def combine_unique(values):
    output = []

    for value in values:
        value = clean_value(value)

        if value and value not in output:
            output.append(value)

    return "; ".join(output)


def find_column(dataframe, possible_names):
    lookup = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for possible_name in possible_names:
        matched = lookup.get(possible_name.lower())

        if matched is not None:
            return matched

    return None


def read_rcat(filename, cluster_name):
    print(f"Reading {filename}", flush=True)

    table = QTable.read(filename)

    if "GAIAEDR3_ID" not in table.colnames:
        raise KeyError(f"{filename} has no GAIAEDR3_ID column")

    dataframe = pd.DataFrame({
        "GAIAEDR3_ID": [
            clean_id(value)
            for value in table["GAIAEDR3_ID"]
        ],
        "Cluster": cluster_name,
    })

    dataframe = dataframe[
        dataframe["GAIAEDR3_ID"] != ""
    ].copy()

    counts = (
        dataframe
        .groupby(["GAIAEDR3_ID", "Cluster"], as_index=False)
        .size()
        .rename(columns={"size": "N_Hectochelle_RV_Measurements"})
    )

    print(
        f"{cluster_name}: {len(dataframe)} RV rows, "
        f"{len(counts)} unique stars",
        flush=True
    )

    return counts


def split_i357_file(filename):
    print(f"Reading {filename}", flush=True)

    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    starts = [
        index
        for index, line in enumerate(lines)
        if line.startswith("#Table")
    ]

    print(f"Found {len(starts)} I/357 tables", flush=True)

    sections = []

    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        sections.append(lines[start:end])

    return sections


def get_i357_name(section):
    for line in section:
        stripped = line.strip()

        if stripped.startswith("#Name:"):
            return stripped.replace("#Name:", "", 1).strip()

    return ""


def parse_i357_section(section):
    table_name = get_i357_name(section)

    noncomment_lines = []

    for line in section:
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        noncomment_lines.append(line.rstrip("\n"))

    if len(noncomment_lines) < 4:
        return None, table_name

    header_line = noncomment_lines[0]
    data_lines = noncomment_lines[3:]

    if ";" not in header_line or not data_lines:
        return None, table_name

    table_text = header_line + "\n" + "\n".join(data_lines)

    try:
        table = pd.read_csv(
            StringIO(table_text),
            sep=";",
            dtype=str,
            keep_default_na=False,
            na_filter=False,
        )
    except Exception as error:
        print(f"Could not parse {table_name}: {error}", flush=True)
        return None, table_name

    table.columns = [
        str(column).strip()
        for column in table.columns
    ]

    for column in table.columns:
        table[column] = table[column].astype(str).str.strip()

    source_column = find_column(table, ["Source", "source_id"])

    if source_column is None:
        return None, table_name

    table[source_column] = table[source_column].apply(clean_id)

    table = table[
        table[source_column].str.fullmatch(r"\d{15,20}", na=False)
    ].copy()

    if source_column != "Source":
        table = table.rename(columns={source_column: "Source"})

    return table, table_name


def read_i357(filename, cluster_name):
    sections = split_i357_file(filename)
    tables = []

    for number, section in enumerate(sections, start=1):
        table, table_name = parse_i357_section(section)

        if table is None or table.empty:
            print(f"{cluster_name} table {number}: {table_name} no data", flush=True)
            continue

        print(
            f"{cluster_name} table {number}: {table_name} "
            f"{len(table)} rows",
            flush=True
        )

        table = table.copy()
        table["Cluster"] = cluster_name
        table["NSS_Table"] = table_name
        tables.append(table)

    if not tables:
        return pd.DataFrame(
            columns=[
                "GAIAEDR3_ID",
                "Cluster",
                "NSS_Table",
                "Gaia_Period_days",
                "Gaia_Eccentricity",
            ]
        )

    combined = pd.concat(tables, ignore_index=True, sort=False)

    period_column = find_column(
        combined,
        ["Per", "Period", "period", "orbital_period"],
    )

    eccentricity_column = find_column(
        combined,
        ["ecc", "Ecc", "eccentricity", "Eccentricity"],
    )

    rows = []

    for source_id, group in combined.groupby("Source", sort=False):
        period = (
            first_nonblank(group[period_column])
            if period_column is not None
            else ""
        )

        eccentricity = (
            first_nonblank(group[eccentricity_column])
            if eccentricity_column is not None
            else ""
        )

        rows.append({
            "GAIAEDR3_ID": clean_id(source_id),
            "Cluster": cluster_name,
            "NSS_Table": combine_unique(group["NSS_Table"]),
            "Gaia_Period_days": period,
            "Gaia_Eccentricity": eccentricity,
        })

    output = pd.DataFrame(rows)

    print(
        f"{cluster_name}: {len(output)} unique I/357 sources",
        flush=True
    )

    return output


def read_i355(filename, cluster_name):
    print(f"Reading {filename}", flush=True)

    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    noncomment_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        noncomment_lines.append(line.rstrip("\n"))

    header_index = None

    for index, line in enumerate(noncomment_lines):
        columns = [
            value.strip()
            for value in line.split(";")
        ]

        if "Source" in columns:
            header_index = index
            break

    if header_index is None:
        raise KeyError(f"Could not find Source in {filename}")

    remaining = noncomment_lines[header_index + 1:]

    if len(remaining) < 3:
        raise ValueError(f"No data rows found in {filename}")

    table_text = (
        noncomment_lines[header_index]
        + "\n"
        + "\n".join(remaining[2:])
    )

    table = pd.read_csv(
        StringIO(table_text),
        sep=";",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )

    table.columns = [
        str(column).strip()
        for column in table.columns
    ]

    for column in table.columns:
        table[column] = table[column].astype(str).str.strip()

    source_column = find_column(table, ["Source", "source_id"])

    if source_column is None:
        raise KeyError(f"No Source column found in {filename}")

    table[source_column] = table[source_column].apply(clean_id)

    table = table[
        table[source_column].str.fullmatch(r"\d{15,20}", na=False)
    ].copy()

    rv_column = find_column(
        table,
        ["RV", "radial_velocity"],
    )

    rv_error_column = find_column(
        table,
        ["e_RV", "radial_velocity_error", "RV_error"],
    )

    rv_transits_column = find_column(
        table,
        ["o_RV", "rv_nb_transits"],
    )

    rv_visibility_column = find_column(
        table,
        ["RVperiods", "rv_visibility_periods_used", "o_RVp"],
    )

    rv_duration_column = find_column(
        table,
        ["RVduration", "rv_time_duration"],
    )

    ruwe_column = find_column(
        table,
        ["RUWE", "ruwe"],
    )

    print(
        f"{cluster_name} I/355 mapping: "
        f"Source={source_column}, "
        f"RV={rv_column}, "
        f"e_RV={rv_error_column}, "
        f"transits={rv_transits_column}, "
        f"visibility={rv_visibility_column}, "
        f"duration={rv_duration_column}, "
        f"RUWE={ruwe_column}",
        flush=True
    )

    output = pd.DataFrame()

    output["GAIAEDR3_ID"] = table[source_column]
    output["Cluster"] = cluster_name

    output["Gaia_Radial_Velocity_kms"] = (
        table[rv_column]
        if rv_column is not None
        else ""
    )

    output["Gaia_RV_Error_kms"] = (
        table[rv_error_column]
        if rv_error_column is not None
        else ""
    )

    output["Gaia_RV_Transits"] = (
        table[rv_transits_column]
        if rv_transits_column is not None
        else ""
    )

    output["Gaia_RV_Visibility_Periods"] = (
        table[rv_visibility_column]
        if rv_visibility_column is not None
        else ""
    )

    output["Gaia_RV_Time_Baseline_days"] = (
        table[rv_duration_column]
        if rv_duration_column is not None
        else ""
    )

    output["Gaia_RUWE"] = (
        table[ruwe_column]
        if ruwe_column is not None
        else ""
    )

    output = output.drop_duplicates(
        subset=["GAIAEDR3_ID", "Cluster"],
        keep="first",
    ).reset_index(drop=True)

    print(
        f"{cluster_name}: {len(output)} unique I/355 sources",
        flush=True
    )

    return output


def file_exists(pattern):
    return bool(glob.glob(pattern))


def find_thejoker_runs(star_id):
    runs = []

    run_200_dir = os.path.join(
        stardata,
        "200.0M_new",
        star_id,
    )

    run_200_jitter_dir = os.path.join(
        stardata,
        "200.0M_jitter",
        star_id,
    )

    run_400_jitter_dir = os.path.join(
        stardata,
        "400.0M_jitter",
        star_id,
    )

    patterns = [
        (
            "200M",
            os.path.join(
                run_200_dir,
                f"rejection_samples_200.0M_{star_id}_new.hdf5",
            ),
        ),
        (
            "200M_MCMC",
            os.path.join(
                run_200_dir,
                f"rejection_samples_MCMC_200.0M_{star_id}_new.hdf5",
            ),
        ),
        (
            "200M_MCMC_Adapt_Full",
            os.path.join(
                run_200_dir,
                f"rejection_samples_MCMC_adapt_full_200.0M_{star_id}_new.hdf5",
            ),
        ),
        (
            "200M_Jitter",
            os.path.join(
                run_200_jitter_dir,
                f"rejection_samples_200M_jitter_{star_id}.hdf5",
            ),
        ),
        (
            "200M_Jitter_MCMC",
            os.path.join(
                run_200_jitter_dir,
                f"*MCMC*{star_id}*.hdf5",
            ),
        ),
        (
            "400M_Jitter",
            os.path.join(
                run_400_jitter_dir,
                f"*{star_id}*.hdf5",
            ),
        ),
    ]

    for run_name, pattern in patterns:
        if file_exists(pattern):
            runs.append(run_name)

    return "; ".join(runs)


print("=" * 90, flush=True)
print("READING RCAT FILES", flush=True)
print("=" * 90, flush=True)

rcat_data = pd.concat(
    [
        read_rcat(rcat_6811_file, "NGC6811"),
        read_rcat(rcat_6866_file, "NGC6866"),
    ],
    ignore_index=True,
)


print("\n" + "=" * 90, flush=True)
print("READING I/357 FILES", flush=True)
print("=" * 90, flush=True)

i357_data = pd.concat(
    [
        read_i357(i357_6811_file, "NGC6811"),
        read_i357(i357_6866_file, "NGC6866"),
    ],
    ignore_index=True,
)


print("\n" + "=" * 90, flush=True)
print("READING I/355 FILES", flush=True)
print("=" * 90, flush=True)

i355_data = pd.concat(
    [
        read_i355(i355_6811_file, "NGC6811"),
        read_i355(i355_6866_file, "NGC6866"),
    ],
    ignore_index=True,
)


i355_keys = i355_data[
    ["GAIAEDR3_ID", "Cluster"]
].drop_duplicates()

i357_keys = i357_data[
    ["GAIAEDR3_ID", "Cluster"]
].drop_duplicates()

catalog_keys = i355_keys.merge(
    i357_keys,
    on=["GAIAEDR3_ID", "Cluster"],
    how="outer",
    indicator=True,
)

catalog_keys["Catalog_Match"] = (
    catalog_keys["_merge"]
    .map({
        "left_only": "I355 only",
        "right_only": "I357 only",
        "both": "I355 and I357",
    })
    .astype(str)
)

catalog_keys = catalog_keys.drop(columns="_merge")

crossmatch = rcat_data.merge(
    catalog_keys,
    on=["GAIAEDR3_ID", "Cluster"],
    how="inner",
)

crossmatch = crossmatch.merge(
    i357_data,
    on=["GAIAEDR3_ID", "Cluster"],
    how="left",
)

crossmatch = crossmatch.merge(
    i355_data,
    on=["GAIAEDR3_ID", "Cluster"],
    how="left",
)

for column in crossmatch.select_dtypes(include="category").columns:
    crossmatch[column] = crossmatch[column].astype(str)

crossmatch = crossmatch.fillna("")

print("\nChecking TheJoker runs...", flush=True)

crossmatch["TheJoker_Runs"] = crossmatch[
    "GAIAEDR3_ID"
].apply(find_thejoker_runs)

desired_columns = [
    "GAIAEDR3_ID",
    "Cluster",
    "Catalog_Match",
    "NSS_Table",
    "Gaia_Period_days",
    "Gaia_Eccentricity",
    "Gaia_Radial_Velocity_kms",
    "Gaia_RV_Error_kms",
    "Gaia_RV_Transits",
    "Gaia_RV_Visibility_Periods",
    "Gaia_RV_Time_Baseline_days",
    "Gaia_RUWE",
    "N_Hectochelle_RV_Measurements",
    "TheJoker_Runs",
]

existing_columns = [
    column
    for column in desired_columns
    if column in crossmatch.columns
]

other_columns = [
    column
    for column in crossmatch.columns
    if column not in existing_columns
]

crossmatch = crossmatch[
    existing_columns + other_columns
]

crossmatch = crossmatch.sort_values(
    by=["Cluster", "GAIAEDR3_ID"]
).reset_index(drop=True)

crossmatch.to_csv(output_file, index=False)

print("\n" + "=" * 90, flush=True)
print("FINISHED", flush=True)
print("=" * 90, flush=True)

print(f"Total matched RCAT stars: {len(crossmatch)}", flush=True)
print(
    f"I355 only: {(crossmatch['Catalog_Match'] == 'I355 only').sum()}",
    flush=True
)
print(
    f"I357 only: {(crossmatch['Catalog_Match'] == 'I357 only').sum()}",
    flush=True
)
print(
    "I355 and I357: "
    f"{(crossmatch['Catalog_Match'] == 'I355 and I357').sum()}",
    flush=True
)
print(
    "Stars with Gaia RV: "
    f"{(crossmatch['Gaia_Radial_Velocity_kms'] != '').sum()}",
    flush=True
)
print(
    "Stars with Gaia RV transit counts: "
    f"{(crossmatch['Gaia_RV_Transits'] != '').sum()}",
    flush=True
)
print(
    "Stars with NSS periods: "
    f"{(crossmatch['Gaia_Period_days'] != '').sum()}",
    flush=True
)
print(
    "Stars with NSS eccentricities: "
    f"{(crossmatch['Gaia_Eccentricity'] != '').sum()}",
    flush=True
)
print(
    "Stars with TheJoker runs: "
    f"{(crossmatch['TheJoker_Runs'] != '').sum()}",
    flush=True
)
print(f"Saved to: {output_file}", flush=True)
