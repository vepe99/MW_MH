import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.stats as stats
from multiprocessing import Pool


import pynbody as pb

import glob
import os
import re
from tqdm.notebook import tqdm


def extract_parameter_array(path='str', path_parameters='str', path_observables='str') -> None:
    """
    Extract the parameters and observables from the path. Checks all the possible errors and if one is found it is saved as an 'error_file'.  
    If no stars were formed in the snapshot, the function dosen't save any file.
    Two .npz files are returned, one with the parameters and another with the observables.
    In order to load the parameters values use the common way of accessing numpy array in .npz file, for example: np.load('file.npz')['star_mass'].
    The parameters that are extracted are: star_mass, gas_mass, dm_mass, infall_time, redshift, a, chemical_mean and chemical_std.
    The observables that are extracted are: [Fe/H], [O/Fe], refered to as 'feh' and 'ofe'.

    Parameters
    ----------
    path : str 
        Path to the simulation snapshot. The path should end with 'simulation_name.snapshot_number' and it is used to create the name of the .npz files.
    path_parameters : str
        Path to the folder where the parameters file will be saved
    path_observables : str
        Path to the folder where the observables file will be saved
    
    Returns
    -------
    parameters : .npz array
        The file is save in the folder '/path_parametrs/name_file_parameters.npz' and contains the following parameters:
    
        parameters['star_mass'] : float
            Total mass of the formed stars in the snapshot
        parameters['gas_mass'] : float
            Total mass of the gas in the snapshot
        parameters['dm_mass'] : float
            Total mass of the dark matter in the snapshot
        parameters['infall_time'] : float
            Time at which the snapshot was taken in Gyr
        parameters['redshift'] : float
            Redshift at which the snapshot was taken
        parameters['a'] : float
            Scale factor at which the snapshot was taken
        parameters['chemical_mean'] : np.array
            Array with the mean of metals, FeMassFrac and OxMassFrac in the snapshot
        parameters['chemical_std'] : np.array
            Array with the standard deviation of metals, FeMassFrac and OxMassFrac in the snapshot

    observables : .npz array
        The file is save in the folder '/path_observables/name_file_observables.npz' and contains the following parameters:

        observables['feh'] : np.array
            Array with the [Fe/H] of the formed stars in the snapshot
        observables['ofe'] : np.array
            Array with the [O/Fe] of the formed stars in the snapshot
    """
    

    #extract the name of the simulation+snapshot_number to create the name of the files to save
    regex = r'[^/]+$'
    name_file = re.search(regex, path).group()
    
    try:
        #check if the file can be loaded
        sim = pb.load(path)
        sim.physical_units()
    except:
        np.savez(file=path_parameters + name_file + '_load_error.npz', emppty=np.array([0]))
        np.savez(file=path_observables + name_file + '_load_error.npz', emppty=np.array([0]))
    else:
        try:
            #check if the halos can be loaded
            h = sim.halos()
            h_1 = h[1]
        except:
            print(f'Halo error {name_file}')
            np.savez(file=path_parameters + name_file + '_halos_error.npz', emppty=np.array([0]))
            np.savez(file=path_observables + name_file + '_halos_error.npz', emppty=np.array([0]))
        else:
            try: 
                mass = h_1.s['mass']
            except:
                print('Dummy halos')
                np.savez(file=path_parameters + name_file + '_dummy_error.npz', emppty=np.array([0]))
                np.savez(file=path_observables + name_file + '_dummy_error.npz', emppty=np.array([0]))
            
            else:
                #check if the simualtion has formed stars
                if len(h_1.s['mass']) > 0:
                    
                    name_parameter_file = path_parameters + name_file + '_parameters.npz'
                    name_observable_file = path_observables + name_file + '_observables.npz'

                    #PARAMETERS
                    star_mass = np.array(h_1.s['mass'].sum()) #in Msol
                    gas_mass = np.array(h_1.g['mass'].sum())  #in Msol
                    dm_mass = np.array(h_1.dm['mass'].sum())  #in Msol
                    infall_time = np.array(h_1.properties['time'].in_units('Gyr'))
                    redshift = np.array(h_1.properties['z'])
                    a = np.array(h_1.properties['a'])
                    try: 
                        #check if the metals, Iron mass fraction and Oxygen mass fraction mean and std can be extracted
                        chemical_mean = np.array([h_1.s['metals'].mean(), h_1.s['FeMassFrac'].mean(), h_1.s['OxMassFrac'].mean()])
                        chemical_std = np.array([h_1.s['metals'].std(), h_1.s['FeMassFrac'].std(), h_1.s['OxMassFrac'].std()])
                    except:
                        np.savez(file=path_parameters + name_file + '_ZMassFracc_error.npz', emppty=np.array([0]))
                        np.savez(file=path_observables + name_file + '_ZMassFracc_error.npz', emppty=np.array([0]))
                    else:
                        #OBSERVABLE
                        try:
                            #check if the [Fe/H] and [O/Fe] can be extracted
                            feh = h_1.s['feh']
                            ofe = h_1.s['ofe']
                        except:
                            np.savez(file=path_parameters + name_file + '_FeO_error.npz', emppty=np.array([0]))
                            np.savez(file=path_observables + name_file + '_FeO_error.npz', emppty=np.array([0]))
                        else:
                            np.savez(file=name_parameter_file, star_mass=star_mass, gas_mass=gas_mass, dm_mass=dm_mass, infall_time=infall_time, redshift=redshift, a=a, chemical_mean=chemical_mean, chemical_std=chemical_std)
                            np.savez(file=name_observable_file, feh=feh, ofe=ofe)
                else:
                    print('Not formed stars yet')        
                    
all_paths = glob.glob('/mnt/storage/_data/nihao/nihao_classic/g?.??e??/g?.??e??.0????')
for path in tqdm(all_paths):
    extract_parameter_array(path, path_parameters='../../data/parameters/', path_observables='../../data/observables/')
    
# def main():
#     all_paths = glob.glob('/mnt/storage/_data/nihao/nihao_classic/g?.??e??/g?.??e??.0????')
#     path_parameters = '../../data/parameters/'
#     path_observables = '../../data/observables/'
    
#     pool = Pool(processes=100)
#     items = zip(all_paths, [path_parameters]*len(all_paths), [path_observables]*len(all_paths))
#     pool.starmap(extract_parameter_array, items)
    
# if __name__ == '__main__':
#     main()