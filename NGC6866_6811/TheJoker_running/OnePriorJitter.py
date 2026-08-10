#imports
import astropy.table as at
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import h5py
import thejoker as tj
import thejoker.units as xu
from astropy.table import QTable, Table, Column, vstack, unique
from astropy.time import Time
from astropy.visualization.units import quantity_support
import astropy.coordinates as coord
import pymc as pm
import arviz as az
import argparse
import os
import schwimmbad

workpath = "/scratch/david_oneprior_jitter"
os.makedirs(workpath, exist_ok=True)

#random generator to ensure reproducibility
rnd = np.random.default_rng(seed=42)

def CreatePriors(num_priors):


    with pm.Model() as model:
        s = xu.with_unit(
            pm.Normal("s", 0.0, 0.5), u.km/u.s
    )

        mils = num_priors/1000000
        prior = tj.JokerPrior.default( #initializing the default prior
            P_min = 0.1 * u.day,
            P_max = 1e4 * u.day,
            sigma_K0 = 30 * u.km / u.s,
            sigma_v = 100 * u.km / u.s,
            model=model,
            pars={"s": s},
    )

    print('initialized prior with jitter')

    print('Creating new priors')
    prior_samples = prior.sample(size = num_priors, rng = rnd, return_logprobs = True) #generating prior samples
    print("New priors created")
    print(prior_samples["s"][:20])
    print(prior_samples["s"].min(), prior_samples["s"].max())
    #prior_samples.write("/data/labs/douglste-laf-lab/wichmand/stardata/prior_samples_1000_jitter_test.hdf5", overwrite = True)
    #prior_samples.write(f"{workpath}/stardata/prior_samples_{mils:.0f}M_jiter.hdf5", overwrite = True) #write out prior samples to research folder 
    outfile = f"{workpath}/prior_samples_{mils:.0f}M_jitter.hdf5"
    prior_samples.write(outfile, overwrite=True)
    print(f"Priors saved to {outfile}")
    return

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--prior', help = 'num of prior samp default 200000000', type = int, default = 200000000)
	args = parser.parse_args()
	print('args')

	CreatePriors(args.prior)









