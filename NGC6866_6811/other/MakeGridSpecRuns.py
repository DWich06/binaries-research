import os

stardata = "/data/labs/douglste-laf-lab/wichmand/stardata"

outfile = os.path.join(stardata, "gridspec_star_runs.txt")

jokerruns = {
    "200M": {
        "workpath": f"{stardata}/200.0M_new",
        "filename": "rejection_samples_200.0M_{id}_new.hdf5",
    },
    "200M_MCMC": {
        "workpath": f"{stardata}/200.0M_new",
        "filename": "rejection_samples_MCMC_200.0M_{id}_new.hdf5",
    },
    "200M_MCMC_adapt_full": {
        "workpath": f"{stardata}/200.0M_new",
        "filename": "rejection_samples_MCMC_adapt_full_200.0M_{id}_new.hdf5",
    },
    "200M_jitter": {
        "workpath": f"{stardata}/200.0M_jitter",
        "filename": "rejection_samples_200M_jitter_{id}.hdf5",
    },
    "200M_jitter_MCMC": {
        "workpath": f"{stardata}/200.0M_jitter",
        "filename": "rejection_samples_MCMC_adapt_full_200M_jitter_{id}.hdf5",
    },
    "400M_jitter": {
        "workpath": f"{stardata}/400.0M_jitter",
        "filename": "rejection_samples_400M_jitter_{id}.hdf5",
    },
}

count = 0

with open(outfile, "w") as f:

    for run_name, run in jokerruns.items():

        workpath = run["workpath"]

        if not os.path.isdir(workpath):
            continue

        for folder in sorted(os.listdir(workpath)):

            if not folder.isdigit():
                continue

            star_id = int(folder)

            sample = os.path.join(
                workpath,
                folder,
                run["filename"].format(id=star_id),
            )

            if os.path.exists(sample):
                f.write(f"{star_id} {run_name}\n")
                count += 1

print(f"Wrote {count} tasks")
print(outfile)
