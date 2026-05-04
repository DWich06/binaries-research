import astropy.units as u
import numpy as np
import thejoker as tj
from astropy.table import QTable, Table, vstack
from astropy.time import Time
import os
import traceback


#Set paths and get files, just using Ella's for now because I have no gotten these files locally yet
workpath = "/data2/labs/douglste-laf-lab/mathewea/200.0M_new"
idlist = QTable.read("/data2/labs/douglste-laf-lab/mathewea/Summer-Research/GAIADR3_IDs.csv")

new_6866 = QTable.read("/data2/labs/douglste-laf-lab/mathewea/rcat_ngc6866_v0.fits")
new_6811 = QTable.read("/data2/labs/douglste-laf-lab/mathewea/rcat_ngc6811_v0.fits")

# Creates an empty table for outputs
datatable = Table()

# Creates empty lists which will become csv columns
ids = []
deltaT = []
mcmc = []
num_samples = []
unimodal = []
bimodal = []
numRVs = []
maxgap = []
phasecov = []
mode1_P = []
mode2_P = [] 
mode1_count = []
mode2_count = []

# Pick 10 IDs to test
failed_kmodal_ids = []
max_failed_kmodal_ids = 10


# Pulls gaia IDs
new_ids_6811 = new_6811["GAIAEDR3_ID"]
new_ids_6866 = new_6866["GAIAEDR3_ID"]

# Finds RV measurements for this star in each catalog
for idnum in idlist["GAIAEDR3_ID"]:
    datamatched6811 = new_6811[idnum == new_ids_6811]
    datamatched6866 = new_6866[idnum == new_ids_6866]
    matched = vstack([datamatched6811, datamatched6866])
    
    RV = len(matched)

    # Stars with fewer than 3 RVs means the constraints are weak
    if RV < 3:
        print(f"{idnum}: skipping, fewer than 3 RVs")
        continue

    # Converts observation dates into astropy Time objects
    t1 = Time(matched["DATE-OBS"], format="fits", scale="tcb")

    # Builds a TheJoker radial velocity dataset using times, RVs, and RV errors
    data = tj.RVData(
        t=t1,
        rv=matched["vrad"] * (u.kilometer / u.second),
        rv_err=matched["vrad_err"] * (u.kilometer / u.second),
    )

    # Paths for TheJoker output files
    rejection_file = f"{workpath}/{idnum}/rejection_samples_200.0M_{idnum}_new.hdf5"
    mcmc_file = f"{workpath}/{idnum}/rejection_samples_MCMC_200.0M_{idnum}_new.hdf5"

    # Only processes the star if the rejection sample file exists, adds the ID to the table, counts rejection sample and records that it isnt MCMC unless proven otherwise
    if os.path.exists(rejection_file):
        ids.append(idnum)

        joker_samples = tj.JokerSamples.read(rejection_file)
        numsamples = len(joker_samples)
        mcmc_check = 0

            # If an MCMC file exists, it replaces the rejection samples with the MCMC samples
        if os.path.exists(mcmc_file):
            joker_samples = tj.JokerSamples.read(mcmc_file)
            numsamples = len(joker_samples)
            mcmc_check = 1

        # Time between observation dates
        dt = t1[1:] - t1[:-1]

        # Check if unimodal
        if tj.is_P_unimodal(joker_samples, data):
            uni = 1
            bi = 0
            modes = [np.nan, np.nan]
            counts = [-1, -1]
        else:
            uni = 0

            try: # Checks if it is bimodal
                is_bi, modes, counts = tj.is_P_Kmodal(
                    joker_samples,
                    data,
                    n_clusters=2,
                )

                if is_bi:
                    bi = 1
                else:
                    bi = 0

            # If the bimodal check fails then record the failure
            except Exception as exc:
                failed_kmodal_ids.append(idnum)
            
                print(f"\n{idnum}: is_P_Kmodal failed")
                print(f"Error message: {exc}")
                traceback.print_exc()
                print()
            
                bi = -1
                modes = [np.nan, np.nan]
                counts = [-1, -1]
            
                if len(failed_kmodal_ids) >= max_failed_kmodal_ids:
                    print("Collected 10 is_P_Kmodal failures:")
                    print(failed_kmodal_ids)
                    raise SystemExit("Stopping after 10 IDs are collected")


        try: # Takes one sample and measures the largest gap in the phase coverage
            one_sample = joker_samples[:1]
            maxphase = tj.max_phase_gap(one_sample, data)
            maxphase = float(maxphase)
        except Exception as exc:
            print(f"{idnum}: max_phase_gap failed: {exc}")
            maxphase = np.nan

        try: # Meausres how well RV observations cover the orbital phase using 10 bins
            one_sample = joker_samples[:1]
            cov = tj.phase_coverage(one_sample, data, n_bins=10)
            cov = float(cov)
        except Exception as exc:
            print(f"{idnum}: phase_coverage failed: {exc}")
            cov = np.nan

        try: # First period mode and converts to a float in days if it has units
            mode1 = modes[0]
            if hasattr(mode1, "to_value"):
                mode1 = mode1.to_value(u.day)
            mode1 = float(mode1)
        except Exception:
            mode1 = np.nan

        try: # Same as above for second
            mode2 = modes[1]
            if hasattr(mode2, "to_value"):
                mode2 = mode2.to_value(u.day)
            mode2 = float(mode2)
        except Exception:
            mode2 = np.nan

        try: # Number of samples for each mode
            count1 = int(counts[0])
        except Exception:
            count1 = -1

        try:
            count2 = int(counts[1])
        except Exception:
            count2 = -1

        # Adds values to the lists for the CSV columns
        num_samples.append(numsamples)
        numRVs.append(RV)
        mcmc.append(mcmc_check)
        unimodal.append(uni)
        bimodal.append(bi)
        deltaT.append(dt.min().value)
        maxgap.append(maxphase)
        phasecov.append(cov)
        mode1_P.append(mode1)
        mode2_P.append(mode2)
        mode1_count.append(count1)
        mode2_count.append(count2)

        # Prints progress for the stars, if there isnt a rejection sample then it skips
        print(f"{idnum}: unimodal={uni}, bimodal={bi}, MCMC={mcmc_check}")

    else:
        print(f"{idnum}: skipping, no Joker samples")

# Turns the lists into columns for the output csv file
datatable["id"] = ids
datatable["dt"] = deltaT
datatable["num_RV"] = numRVs
datatable["num_samples"] = num_samples
datatable["MCMC"] = mcmc
datatable["unimodal"] = unimodal
datatable["bimodal"] = bimodal
datatable["mode1_P"] = mode1_P
datatable["mode2_P"] = mode2_P
datatable["mode1_count"] = mode1_count
datatable["mode2_count"] = mode2_count
datatable["max_phase_gap"] = maxgap
datatable["phase_coverage"] = phasecov

# Writes final csc
datatable.write("bimodalcheck_200M_phase.csv", format="csv", overwrite=True)
print("wrote bimodalcheck_200M_phase.csv")

