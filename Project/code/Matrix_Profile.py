"""
Auto Encoder method

It will output 250 residuals data in folder ../log/matrixprofile/
"""

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import matrixprofile as mp

path = '../data-sets/KDD-Cup/data/'
period_path = '../data-sets/KDD-Cup/period/period.csv'
files_name = [i for i in os.listdir(path) if 'Anomaly' in i] #remove irrelevant files
files_name.sort(key = lambda x : x.split('_')[0]) #sort by id

period = pd.read_csv(period_path) #load period file calculated by fourier transform

for name in files_name:
    split_index  = int(name.split('.')[0].split('_')[3]) #get split index

    data = pd.read_csv(path+name,header=None)#to handle non-uniform data format
    if data.shape == (1,1):
        tmp = [i for i in data[0][0].split(' ') if i!= '']
        data = pd.DataFrame({0:tmp}).astype('float')
    train= data[0:split_index]

    training_mean = train.mean()  #record mean
    training_std = train.std() #record std
    normalized_data = (data - training_mean) / training_std #noralize

    window_size = int(period[period['File_name'] == name]['Period']) #get window size according to period

    profile = mp.compute(normalized_data[0].values, window_size) #compute matrix profile

    profile_with_discords = mp.discover.discords(profile, k=5)
    mp_adjusted = np.append(profile_with_discords['mp'], np.zeros(window_size - 1) + np.nan) 

    np.savetxt('../log/matrix_profile/'+name,mp_adjusted) 

    print(name,' SUCCESS!')