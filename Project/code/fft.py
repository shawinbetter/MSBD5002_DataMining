"""
Fast Fourier Transform method

It will output 250 residuals data in folder ../log/fft/
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft,ifft,fftfreq
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from scipy import signal
import os

def ft_lowpass(y,TIME_STEPS,filter_rate = 0.01,order = 8):
    '''
    filter_rate: control filter how much signal(default = 1%)
    
    Threshold = n Hz = filter_rate * len
    len = len(y) Hz
    wn = 1 - 2*n/0.01*len(y), filter the highest 1% noise signal
    '''
    b,a = signal.butter(order,1-2*filter_rate*len(y)/len(y),'lowpass') # 8阶
    new = signal.filtfilt(b,a,y)
    point_sse = np.array(np.square(y-new))
    res = point_sse[split_index:-1-TIME_STEPS]
    
    # return the square error of all test data points
    # return type: np.array
    return res

def ft_highpass(y,TIME_STEPS,filter_rate = 0.01,order = 8):
    '''
    filter_rate: control filter how much signal(default = 1%)
    
    Threshold = n Hz = filter_rate * len
    len = len(y) Hz
    wn = 2*n/0.01*len(y)
    '''
    b,a = signal.butter(order,2*filter_rate*len(y)/len(y),'highpass') # 8阶
    new = signal.filtfilt(b,a,y)
    point_sse = np.array(np.square(y-new))
    res = point_sse[split_index:-1-TIME_STEPS]
    
    # return the square error of all test data points
    # return type: np.array
    return res

def ft_filter_output(y,TIME_STEPS,f_type='band',filter_rate = 0.01,order = 8):
    '''
    y: all data input,the whole data sequence.
    f_type: filter type. Choose 'band','highpass','lowpass', recommend 'band' and 'highpass'!
    filter_rate: filter how much signal. For band pass, it will filter 2*filter_rate signal.
    order: use for Butterwoth filter create(阶数)
    '''
    if f_type == 'highpass' or f_type == 'high':
        se = ft_highpass(y,TIME_STEPS,filter_rate,order)
    elif f_type == 'lowpass' or f_type == 'low':
        se = ft_lowpass(y,TIME_STEPS,filter_rate,order)
    elif f_type == 'band':
        se = ft_band(y,TIME_STEPS,filter_rate,order)
    else:
        print('Type is not correct, please check.')
    
    return se

def ft_band(y,TIME_STEPS,filter_rate = 0.01,order = 8):
    '''
    filter_rate: control filter how much signal(default = 1%)
    Cause this is bandpass filter, finally it will filter 2*filter_rate signal.
    
    Threshold = n Hz = filter_rate * len
    len = len(y) Hz
    wn_low = 2*n/0.01*len(y)
    wn_high = 1 - 2*n/0.01*len(y)
    '''
    b,a = signal.butter(order,[2*filter_rate*len(y)/len(y),1-2*filter_rate*len(y)/len(y)],'band') # 8阶
    new = signal.filtfilt(b,a,y)
    point_sse = np.array(np.square(y-new))
    res = point_sse[split_index:-1-TIME_STEPS]
    
    # return the square error of all test data points
    # return type: np.array
    return res

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
    
    # Normalization
    data_mean = train.mean() #record mean
    data_std = train.std() #record std
    normalized_data = (data - data_mean) / data_std

    TIME_STEPS = int(period[period['File_name'] == name]['Period'])
    
    res = ft_filter_output(normalized_data[0],TIME_STEPS,'band',0.01,8)

    np.savetxt('../log/fft/'+name+'.txt',res)

    print(name," SUCCESS!")