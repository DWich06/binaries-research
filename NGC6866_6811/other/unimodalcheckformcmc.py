import os
import astropy.units as u
import thejoker as tj
from astropy.table import QTable, vstack
from astropy.time import Time


workpath = "/data/labs/douglste-laf-lab/wichmand/stardata/200.0M_jitter"

star_ids = [
    name for name in os.listdir(workpath)
    if os.path.isdir(os.path.join(workpath, name)) and name.isdigit()
]

cat_6866 = QTable.read(
    "/data/labs/douglste-laf-lab/wichmand/stardata/rcat_ngc6866_v0.fits"
)

cat_6811 = QTable.read(
    "/data/labs/douglste-laf-lab/wichmand/stardata/rcat_ngc6811_v0.fits"
)

catalog = vstack([cat_6811, cat_6866])

mcmc_ids = []

for idnum in star_ids:

    rej_path = (
        f"{workpath}/{idnum}/"
        f"rejection_samples_200M_jitter_{idnum}.hdf5"
    )

    if not os.path.exists(rej_path):
        continue

    matched = catalog[catalog["GAIAEDR3_ID"] == int(idnum)]

    if len(matched) == 0:
        continue

    t = Time(matched["DATE-OBS"], format="fits", scale="tcb")

    data = tj.RVData(
        t=t,
        rv=matched["vrad"] * (u.km / u.s),
        rv_err=matched["vrad_err"] * (u.km / u.s),
    )

    joker_samples = tj.JokerSamples.read(rej_path)

    if tj.is_P_unimodal(joker_samples, data):
        mcmc_ids.append(idnum)

QTable({"GAIAEDR3_ID": mcmc_ids}).write(
    "mcmc_unimodal_ids.txt",
    format="ascii.no_header",
    overwrite=True,
)

print(f"{len(mcmc_ids)} stars selected for MCMC.")
