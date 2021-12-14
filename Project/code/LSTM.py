"""
LSTM method

It will output 250 residuals data in folder ../log/lstm/
"""
import sys
from tensorflow.python.ops.variables import model_variables
sys.path = ['', '/usr/local/packages/python/modules/matplotlib-3.1.1/lib/python3.6/site-packages', '/usr/local/packages/python/modules/opencv-4.0.0/lib/python3.6/site-packages', '/usr/local/packages/python/modules/mxnet-1.2.0/lib/python3.6/site-packages', '/usr/local/packages/python/modules/pytorch-0.4.1/lib/python3.6/site-packages', '/usr/local/packages/python/modules/tensorflow-2.2/lib/python3.6/site-packages', '/usr/local/software/python3/lib/python36.zip', '/usr/local/software/python3/lib/python3.6', '/usr/local/software/python3/lib/python3.6/lib-dynload', '/data/yqiuau/qiuyaowen/lib/python3.6/site-packages']
import os
os.environ["CUDA_VISIBLE_DEVICES"] =  "0,1,2"
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler

path = '../data-sets/KDD-Cup/data/'
period_path = '../data-sets/KDD-Cup/period/period.csv'
files_name = [i for i in os.listdir(path) if 'Anomaly' in i] #remove irrelevant files
files_name.sort(key = lambda x : x.split('_')[0]) #sort by id

period = pd.read_csv(period_path) #load period file calculated by fourier transform

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def get_model(X_train):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(
        units=32,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(keras.layers.Dropout(rate=0.15))
    model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
    model.add(keras.layers.LSTM(units=32, return_sequences=True))
    model.add(keras.layers.Dropout(rate=0.15))
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Dense(units=X_train.shape[2])
        )
    )
    return model

def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.1
    epochs_drop = 20
    end_lr = 0.0001
    lrate = initial_lrate * np.power(drop,  
        np.floor((1+epoch)/epochs_drop))
    if lrate > end_lr:
        return lrate
    else:
        return end_lr

for name in files_name[238:241]:
    split_index  = int(name.split('.')[0].split('_')[3]) #get split index
    data = pd.read_csv(path+name,header=None)
    if data.shape == (1,1):
        tmp = [i for i in data[0][0].split(' ') if i!= '']
        data = pd.DataFrame({0:tmp}).astype('float')

    train,test = data[0:split_index],data[split_index::] #split
    test = test.reset_index().drop('index',axis=1)

    training_mean = train.mean()  #record mean
    training_std = train.std() #record std
    normalized_train = (train - training_mean) / training_std
    normalized_test = (test - training_mean) / training_std

    TIME_STEPS = int(period[period['File_name'] == name]['Period'])

    # reshape to [samples, time_steps, n_features]

    X_train, y_train = create_dataset(
        normalized_train[[0]],
        normalized_train[0],
        TIME_STEPS
    )

    X_test, y_test = create_dataset(
        normalized_test [[0]],
        normalized_test [0],
        TIME_STEPS
    )


    epochs = 50
    batch_size = 8
    lr_scheduler = LearningRateScheduler(step_decay)
    optimizer = keras.optimizers.Adam()

    ########Define Model#################
    strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1","GPU:2"])
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = get_model(X_train)
        model.compile(optimizer=optimizer, loss="mae")

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        shuffle=False
    )

    X_test_pred = model.predict(X_test)
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
    test_mae_loss_adjusted = np.append(np.zeros(TIME_STEPS),test_mae_loss)


    np.savetxt('../log/LSTM/'+name+'.txt',test_mae_loss_adjusted)
    print(name," SUCCESS!")

    del model,test_mae_loss,test_mae_loss_adjusted
    