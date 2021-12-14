"""
Ensemble Model which aggregate residuals from all models

It will create the submission file 
"""
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn import preprocessing

path = '../data-sets/KDD-Cup/data/'
period_path = '../data-sets/KDD-Cup/period/period.csv'
files_name = [i for i in os.listdir(path) if 'Anomaly' in i] #remove irrelevant files
files_name.sort(key = lambda x : x.split('_')[0]) #sort by id

period = pd.read_csv(period_path) #load period file calculated by fourier transform

log_dir = '../log/'
res_dir = ['autoencoder','stl','matrix_profile','lstm','fft']

#locations of anomalies
Locations = []

for name in files_name:
    split_index  = int(name.split('.')[0].split('_')[3]) #get split index
    data = pd.read_csv(path+name,header=None)
    if data.shape == (1,1):
        tmp = [i for i in data[0][0].split(' ') if i!= '']
        data = pd.DataFrame({0:tmp}).astype('float')
    
    TIME_STEPS = int(period[period['File_name'] == name]['Period'])

    #load residual data
    a = np.loadtxt(log_dir+'autoencoder/'+name+'.txt')
    if '239_' in name or '240_' in name or '241_' in name:
        #for these three data, we failed to train LSTM model due to OOM
        #ignore the term of lstm residuals
        l = np.zeros(a.shape)
    else:
        l = np.loadtxt(log_dir+'lstm/'+name+'.txt')

    m = np.loadtxt(log_dir+'matrix_profile/'+name)
    s = np.loadtxt(log_dir+'stl/'+name+'.txt')
    f = np.loadtxt(log_dir+'fft/'+name+'.txt')

    #extract the residuals of test data
    m = m[split_index::]

    std_a,std_l,std_m,std_s,std_f = np.nanstd(a),np.nanstd(l),np.nanstd(m),np.nanstd(s),np.nanstd(f)

    w_a = (np.nanmax(a) - np.nanmean(a))/ np.nanstd(a)
    if '239_' not in name or '240_' not in name or '241_' not in name:
        w_l = (np.nanmax(l) - np.nanmean(l))/ np.nanstd(l)
    else:
        w_l = 0
    w_m = (np.nanmax(m) - np.nanmean(m))/ np.nanstd(m)
    w_s = (np.nanmax(s) - np.nanmean(s))/ np.nanstd(s)
    w_f = (np.nanmax(f) - np.nanmean(f))/ np.nanstd(f)

    summation = sum([w_a,w_l,w_m,w_s,w_f])
    w_a,w_l,w_m,w_s,w_f = w_a/summation,w_l/summation,w_m/summation,w_s/summation,w_f/summation

    print("The weight of autoencoder is {}, weight of lstm is {}, weight of matrix profile is {}, weight of stl is {}, weight of fft {}".format(w_a,w_l,w_m,w_s,w_f))

    #fill nan to zero
    a = np.nan_to_num(a, copy=True, nan=0.0)
    l = np.nan_to_num(l, copy=True, nan=0.0)
    m = np.nan_to_num(m, copy=True, nan=0.0)
    s = np.nan_to_num(s, copy=True, nan=0.0)
    f = np.nan_to_num(f, copy=True, nan=0.0)

    #normalize to [0,1], elimate the effect of scale
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(a.reshape(-1, 1))
    if '239_' not in name or '240_' not in name or '241_' not in name:
        l = min_max_scaler.fit_transform(l.reshape(-1, 1))
    m = min_max_scaler.fit_transform(m.reshape(-1, 1))
    s = min_max_scaler.fit_transform(s.reshape(-1, 1))
    f = min_max_scaler.fit_transform(f.reshape(-1, 1))

    #apppend 0 to the end of fft residuals, to aviod the problem of different shape
    f = np.append(f,[0 for i in range(TIME_STEPS+1)]).reshape(-1, 1)

    weight_res = w_a * a + w_l*l + w_m*m + w_s*s + w_f*f
    # weight_res = .25 * a + .25*l + .25*m + .25*s  + .25*f

    # weight_res =  w_m*m + w_s*s + w_f*f

    center = np.argmax(weight_res[TIME_STEPS:-1-TIME_STEPS]) + split_index  + TIME_STEPS #the center is the point with largest residual

    ### Also visualize the maximum residual point detected by ont algorithm
    center_a = np.argmax(a[TIME_STEPS:-1-TIME_STEPS]) + split_index + TIME_STEPS
    if '239_' not in name or '240_' not in name or '241_' not in name:
        center_l = np.argmax(l[TIME_STEPS:-1-TIME_STEPS]) + split_index + TIME_STEPS
    center_m = np.argmax(m[TIME_STEPS:-1-TIME_STEPS]) + split_index + TIME_STEPS
    center_s = np.argmax(s[TIME_STEPS:-1-TIME_STEPS]) + split_index + TIME_STEPS
    center_f =  np.argmax(f[TIME_STEPS:-1-TIME_STEPS]) + split_index + TIME_STEPS

    if '241_' not in name:
        plt.style.use(['science','ieee','std-colors'])
        fig = plt.figure(figsize=[16,6],dpi=200)
        fig.patch.set_facecolor('#FFFFFF')
        plt.grid(color='grey', linestyle='-.', linewidth=0.2)
        plt.plot(range(split_index,len(data)),data[split_index:],label='Test data')

        plt.plot(range(center_a-100,center_a+100),data[center_a-100:center_a+100],color='green',label = 'Region of detected anomaly located by autoencoder')
        if '239_' in name or '240_' in name or '241_' not in name:
            plt.plot(range(center_l-100,center_l+100),data[center_l-100:center_l+100],color='purple',label = 'Region of detected anomaly located by LSTM')
        plt.plot(range(center_m-100,min(center_m+100,len(data))),data[center_m-100:min(center_m+100,len(data))],color='orange',label = 'Region of detected anomaly located by Matrix Profile')
        plt.plot(range(center_s-100,center_s+100),data[center_s-100:center_s+100],color='pink',label = 'Region of detected anomaly located by STL')
        plt.plot(range(center_f-100,center_f+100),data[center_f-100:center_f+100],color='grey',label = 'Region of detected anomaly located by FFT')

        plt.plot(range(center-100,center+100),data[center-100:center+100],color='red',label = 'Region of detected anomaly located by weighted sum residuals')

        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig('../fig/'+name+'.png',dpi=200)
    Locations.append(center)

    print(name," SUCCESS!")

    del a,l,s,m,f,data,weight_res

pd.DataFrame({'No.':range(1,len(files_name)+1),'Location of Anomaly':Locations}).to_csv('submission.csv')