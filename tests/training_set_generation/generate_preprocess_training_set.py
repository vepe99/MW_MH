import numpy as np
import pandas as pd 
from multiprocessing import Pool

import os
import re
import sys
import glob



def normalize(df):
    return df.apply(lambda x: (x.to_numpy() - x.to_numpy().mean()) / x.to_numpy().std(), axis=0)

    
def load_data(observables_path, mass_cut=6*1e9, min_n_star=float, min_feh=float, min_ofe=float):
    
    name_columns_observables = [i for i in np.load(observables_path).keys()]
    name_columns_parameters = [i.replace('mass', 'log10mass') for i in np.load(observables_path.replace('observables', 'parameters')).keys()][:-2]
    name_columns_mean_std = ['mean_metallicity', 'mean_FeMassFrac', 'mean_OMassFrac', 'std_metallicity', 'std_FeMassFrac', 'std_OMassFrac']
    components = name_columns_observables + name_columns_parameters + name_columns_mean_std 

    
    parameter_path = observables_path.replace('observables', 'parameters')
    mass = np.load(parameter_path)['star_mass']
    if mass < mass_cut:
        observables = np.load(observables_path)
        parameters = np.load(parameter_path)
        if len(observables['feh']) > min_n_star:
            l = len(observables['feh'])
            data = np.zeros((l, len(components)))
            data[:, 0] = observables['feh']
            data[:, 1] = observables['ofe']
            ones = np.ones(l)
            data[:, 2] = np.log10(parameters['star_mass'])*ones
            data[:, 3] = np.log10(parameters['gas_mass'])*ones
            data[:, 4] = np.log10(parameters['dm_mass'])*ones
            data[:, 5] = parameters['infall_time']*ones
            data[:, 6] = parameters['redshift']*ones
            data[:, 7] = parameters['a']*ones
            data[:, 8] = parameters['chemical_mean'][0]*ones
            data[:, 9] = parameters['chemical_mean'][1]*ones
            data[:, 10] = parameters['chemical_mean'][2]*ones
            data[:, 11] = parameters['chemical_std'][0]*ones
            data[:, 12] = parameters['chemical_std'][1]*ones
            data[:, 13] = parameters['chemical_std'][2]*ones
            
            df_temp = pd.DataFrame(data, columns=components)
            df_temp = df_temp[(df_temp['feh'] > min_feh) & (df_temp['ofe'] > min_ofe)]
            df_temp['Galaxy_name'] = [observables_path.replace('../../data/observables/', '').replace('_observables.npz', '') for i in range(len(df_temp))]
            return df_temp
            
def main():
    min_n_star = np.percentile(np.load('../../data/preprocessing/Number_Star.npy'), 10)
    min_feh    = np.percentile(np.load('../../data/preprocessing/FeH.npy'), 0.1)
    min_ofe    = np.percentile(np.load('../../data/preprocessing/OFe.npy'), 0.1) 
    mass_cat = 6*1e9
     
    path_2observables = sorted(os.listdir('../../data/observables/'))
    # Filter the lists of paths to take only the ones that don't have the word 'error' in them and 1024 snapshots
    # regex = r'^(?!.*error).*1024.*$'
    regex = r'^(?!.*error)'
    path_2observables = ['../../data/observables/'+path for path in path_2observables if re.search(regex, path)]
    
    pool = Pool(processes=20)
    items = zip(path_2observables, [mass_cat]*len(path_2observables), [min_n_star]*len(path_2observables), [min_feh]*len(path_2observables), [min_ofe]*len(path_2observables))
    df_list = pool.starmap(load_data, items)
    df = pd.concat(df_list, ignore_index=True)
    
    bad_column = 'Galaxy_name'
    other_cols = df.columns.difference([bad_column])    
    df[other_cols] = normalize(df[other_cols]) #nomalization must be then reverted during inference to get the correct results
    df.to_parquet('../../data/preprocessing/preprocess_training_set_Galaxy_name.parquet')
    
if __name__ == '__main__':
    main()

