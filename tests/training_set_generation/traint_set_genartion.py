import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.stats as stats

import pynbody as pb

import glob
import os
import re
from tqdm.notebook import tqdm

def extract_parameter_array(path='str'):
    """
    Extract the parameter from the path
    """
    sim = pb.load(path)
    sim.physical_units()
    h = sim.halos(write_fpos=False)
    try:
        regex = r'[^/]+$'
        name_file = re.search(regex, path).group()
        pb.analysis.angmom.faceon(h[1])
    except:
        np.savez(file='../../data/parameters/'+name_file+'_faceon_error.npz', emppty=np.array([0]))
        np.savez(file='../../data/observables/'+name_file+'_faceon_error.npz', emppty=np.array([0]))
    else:
        if len(sim.s['mass']) > 0:

            name_parameter_file = '../../data/parameters/' + name_file + '_parameters.npz'
            name_observable_file = '../../data/observables/' + name_file + '_observables.npz'

            star_mass = sim.s['mass'].sum()
            gas_mass = sim.g['mass'].sum()
            dm_mass = sim.dm['mass'].sum()
            infall_time = sim.properties['time'].in_units('Gyr')
            redshift = sim.properties['z']
            a = sim.properties['a'] 
            chemical_mean = np.array([sim.s['metals'].mean(), sim.s['FeMassFrac'].mean(), sim.s['OxMassFrac'].mean()])
            chemical_std = np.array([sim.s['metals'].std(), sim.s['FeMassFrac'].std(), sim.s['OxMassFrac'].std()])

            np.savez(file=name_parameter_file, star_mass=star_mass, gas_mass=gas_mass, dm_mass=dm_mass, infall_time=infall_time, redshift=redshift, a=a, chemical_mean=chemical_mean, chemical_std=chemical_std)


            feh = sim.s['feh']
            ofe = sim.s['ofe']
            np.savez(file=name_observable_file, feh=feh, ofe=ofe)

        else:
            print('Not formed stars yet')
        

all_paths = glob.glob('/mnt/storage/_data/nihao/nihao_classic/g*/0*/*.*.*')
# Regular expression to match files that end with 5 numbers and don't have a dot at the end

regex = r'^.*\d{5}$' #all the snapshots


# Filter the list of files
paths = [path for path in all_paths if re.match(regex, path)]


for path in tqdm(paths[210:]):
    print(path)
    extract_parameter_array(path)