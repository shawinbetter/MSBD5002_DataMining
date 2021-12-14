"""
STL Method

It will output 250 residuals data in folder ../log/autoencoder/stl
"""

from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

path = '../data-sets/KDD-Cup/data/'
period_path = '../data-sets/KDD-Cup/period/period.csv'
files_name = [i for i in os.listdir(path) if 'Anomaly' in i] #remove irrelevant files
files_name.sort(key = lambda x : x.split('_')[0]) #sort by id

period = pd.read_csv(period_path) #load period file calculated by fourier transform

for name in files_name:
    split_index  = int(name.split('.')[0].split('_')[3]) #get split index
    data = pd.read_csv(path+name,header=None)
    if data.shape == (1,1):
        tmp = [i for i in data[0][0].split(' ') if i!= '']
        data = pd.DataFrame({0:tmp}).astype('float')
    train,test = data[0:split_index],data[split_index::] #split
    test = test.reset_index().drop('index',axis=1)

    training_mean = train.mean()  #record mean
    training_std = train.std() #record std
    # normalized_train = (train - training_mean) / training_std
    # normalized_test = (test - training_mean) / training_std
    normalized_data = (data -  training_mean) / training_std

    TIME_STEPS = int(period[period['File_name'] == name]['Period'])

    result = seasonal_decompose(normalized_data, model='additive',period=TIME_STEPS)

    resid = abs(result.resid)

    np.savetxt('../log/stl/'+name+'.txt',resid[split_index:])

    print(name," SUCCESS!")
