import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from statsmodels.tsa.stattools import acf
import datetime
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler

path = '../data-sets/KPI/'
train = pd.read_csv(path+'train-data.csv')
test = pd.read_hdf(path+"test-data.hdf")
test['KPI ID'] = test['KPI ID'].astype(str)

def get_week_of_month(year, month, day):
    begin = int(datetime.date(year, month, 1).strftime("%W"))
    end = int(datetime.date(year, month, day).strftime("%W"))

    return end - begin + 1
def extract_time_stamp(ts):
    string = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    year,month,day,hour = int(string[0:4]),int(string[5:7]),int(string[8:10]),int(string[11:13])
    
    week_of_month = get_week_of_month(year,month,day)
    
    if 0 <= hour <= 6:
        time_period = 'EarlyMorning'
    elif 6< hour <= 12:
        time_period = 'Morning'
    elif 12 < hour <= 18:
        time_period = 'Afternoon'
    else:
        time_period = 'Night'
    
    return week_of_month,time_period

def cal_period(data):
    fft_series = fft(data)
    power = np.abs(fft_series)
    sample_freq = fftfreq(fft_series.size)

    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    powers = power[pos_mask]

    top_k_seasons = 3

    # top K=5 index
    top_k_idxs = np.argpartition(powers, -top_k_seasons)[-top_k_seasons:]
    top_k_power = powers[top_k_idxs]
    fft_periods = (1 / freqs[top_k_idxs]).astype(int)
    
    # Expected time period
    scores = []
    for lag in fft_periods:
        # lag = fft_periods[np.abs(fft_periods - time_lag).argmin()]
        acf_score = acf(data, nlags=lag)[-1]
        scores.append(acf_score)
        # print(f"lag: {lag} fft acf: {acf_score}")
        
    period = fft_periods[scores.index(max(scores))] #candidated periods with highest acf score
    
    return period

# Generated training sequences for use in the model.
def create_sequences(df, time_steps):
    output = []
    for i in range(len(df) - time_steps + 1):
        x = np.array(df[i:i+time_steps])
        output.append(x)
    return np.stack(output)

def get_model(period,feature):
    model = keras.Sequential()
    model.add(layers.Conv1D(filters=32, kernel_size=6, activation='elu', padding='same',input_shape=(period,feature)))
    model.add(layers.Conv1D(filters=32, kernel_size=6, activation='elu',padding='same'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 10
    end_lr = 0.00001
    lrate = initial_lrate * np.power(drop,  
        np.floor((1+epoch)/epochs_drop))
    if lrate > end_lr:
        return lrate
    else:
        return end_lr

unique_kid = train['KPI ID'].unique().tolist()
TRUE = np.array([])
TRUE_PRED = np.array([])
for kid in unique_kid:
    partial_train = train[train['KPI ID'] == kid]
    partial_train = partial_train.reset_index().drop(['index'],axis=1)
    
    ##data engineering
    partial_train['week_of_month'],partial_train['time_period'] = zip(*partial_train['timestamp'].map(extract_time_stamp))
    
    ## normalization
    mean,std = partial_train['value'].mean(),partial_train['value'].std()
    
    partial_train['value'] = (partial_train['value']-mean)/std
    
    ##
    period = cal_period(np.array(partial_train['value']))
    
    ## define trainning data
    x_train = partial_train[[i for i in partial_train.columns if i not in ['timestamp','label','KPI ID']]]
    x_train = pd.get_dummies(x_train)
    num_of_features = x_train.shape[1]
    x_train = create_sequences(x_train, period)
    y_train = partial_train['label'][period-1:]
    percentile = 1 - (y_train.sum() / len(y_train))
    

    ##build model
    # clf = RandomForestClassifier()
    # clf.fit(x_train, y_train)
    model = get_model(period,num_of_features)
    epochs = 1
    batch_size = 2
    lr_scheduler = LearningRateScheduler(step_decay)
    optimizer = keras.optimizers.Adam(0.01)

    model.compile(optimizer=optimizer, loss="binary_crossentropy",metrics=['accuracy'])
    
    model.fit(x_train,y_train,epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.2,
                            callbacks=[lr_scheduler],
                            class_weight = {0:max(0.05,1-percentile), 1:min(0.95,percentile)},
                            verbose=1)
    
    del x_train,y_train,partial_train
    
    #predict on test data
     
    partial_test = test[test['KPI ID'] == kid]
    partial_test = partial_test.reset_index().drop(['index'],axis=1)
    
    partial_test['week_of_month'],partial_test['time_period'] = zip(*partial_test['timestamp'].map(extract_time_stamp))
    
    partial_test['value'] = (partial_test['value']-mean)/std
    
    x_test = partial_test[[i for i in partial_test.columns if i not in ['timestamp','label','KPI ID']]]
    x_test = pd.get_dummies(x_test)
    x_test = create_sequences(x_test, period)
    y_test = partial_test['label'][period-1:]
    
    pred = model.predict(x_test)
    threshold = 0.32
    pred = (pred >= threshold).astype(int)
    pred = np.squeeze(pred)
    
    # print(pred.shape,y_test.shape,x_test.shape)
    # print(classification_report(y_test, pred))
    
    pred = np.append(np.zeros(period-1),pred)
    
    TRUE_PRED = np.append(TRUE_PRED,pred)
    
    TRUE = np.append(TRUE,np.array(partial_test['label']))

    pd.DataFrame({'pred':pred,'true':partial_test['label']}).to_csv(str(kid)+'.csv',index=None)

print(classification_report(TRUE, TRUE_PRED))