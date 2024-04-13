import numpy as np
import pandas as pd 

import os
import re
import sys
import glob

path_2parameters = sorted(os.listdir('../../data/parameters/'))
path_2observables = sorted(os.listdir('../../data/observables/'))

# Filter the lists of paths to take only the ones that don't have the word 'error' in them and 1024 snapshots
# regex = r'^(?!.*error).*1024.*$'
regex = r'^(?!.*error)'

path_2parameters = ['../../data/parameters/'+path for path in path_2parameters if re.search(regex, path)]
path_2observables = ['../../data/observables/'+path for path in path_2observables if re.search(regex, path)]

def normalize(df):
    return df.apply(lambda x: (x.to_numpy() - x.to_numpy().mean()) / x.to_numpy().std(), axis=0)


def load_data(observables_paths, mass_cut=6*1e9, n_star_percentile=10, feh_percentile=0.1, ofe_percentile=0.1):
    
    min_n_star = np.percentile(np.load('../../data/preprocessing/Number_Star.npy'), n_star_percentile)
    min_feh    = np.percentile(np.load('../../data/preprocessing/FeH.npy'), feh_percentile)
    min_ofe    = np.percentile(np.load('../../data/preprocessing/OFe.npy'), ofe_percentile)  
    
    len_array = 0
    for observables_path in observables_paths:
        feh = np.load(observables_path)['feh']
        ofe = np.load(observables_path)['ofe']
        feh = feh[feh > -5]
        len_array += len(feh)

    name_columns_observables = [i for i in np.load(observables_paths[0]).keys()]
    name_coumns_parameters = [i.replace('mass', 'log10mass') for i in np.load(observables_paths[0].replace('observables', 'parameters')).keys()][:-2]
    name_columns_mean_std = ['mean_metallicity', 'mean_FeMassFrac', 'mean_OMassFrac', 'std_metallicity', 'std_FeMassFrac', 'std_OMassFrac']
    components = name_columns_observables + name_coumns_parameters + name_columns_mean_std  

    df = pd.DataFrame(columns=components) 
    for i, observables_path in enumerate(observables_paths):
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
                df = pd.concat([df, df_temp], ignore_index=True)
                df.reset_index()
                df['Galaxy_name'] = observables_path.replace('../../data/observables/', '').replace('_observables.npz', '')
                print(i)
                
    for i in df.columns[:-1]:
        mean_i = df[i].mean()
        std_i = df[i].std()
        print(std_i)
        np.savez(file='../../data/preprocessing/mean_std_of_'+i, mean=mean_i, std=std_i)
    
    bad_column = 'Galaxy_name'
    other_cols = df.columns.difference([bad_column])    
    df[other_cols] = normalize(df[other_cols]) #nomalization must be then reverted during inference to get the correct results
    df.to_parquet('../../data/preprocessing/preprocess_training_set_Galaxy_name.parquet')   
    # df.to_parquet('../../data/preprocessing/TEST.parquet')
    
load_data(path_2observables)

