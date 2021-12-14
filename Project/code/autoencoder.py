"""
Auto Encoder method

It will output 250 residuals data in folder ../log/autoencoder/
"""
import sys
from tensorflow.python.ops.variables import model_variables
sys.path = ['', '/usr/local/packages/python/modules/matplotlib-3.1.1/lib/python3.6/site-packages', '/usr/local/packages/python/modules/opencv-4.0.0/lib/python3.6/site-packages', '/usr/local/packages/python/modules/mxnet-1.2.0/lib/python3.6/site-packages', '/usr/local/packages/python/modules/pytorch-0.4.1/lib/python3.6/site-packages', '/usr/local/packages/python/modules/tensorflow-2.2/lib/python3.6/site-packages', '/usr/local/software/python3/lib/python36.zip', '/usr/local/software/python3/lib/python3.6', '/usr/local/software/python3/lib/python3.6/lib-dynload', '/data/yqiuau/qiuyaowen/lib/python3.6/site-packages']
import os
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

def get_train_test(name):
    split_index  = int(name.split('.')[0].split('_')[3]) #get split index
    data = pd.read_csv(path+name,header=None)
    if data.shape == (1,1):
        tmp = [i for i in data[0][0].split(' ') if i!= '']
        data = pd.DataFrame({0:tmp}).astype('float')
    train,test = data[0:split_index],data[split_index::] #split
    return train,test

def normalize(train):
    training_mean = train.mean()  #record mean
    training_std = train.std() #record std
    normalized_train = (train - training_mean) / training_std
    return training_mean,training_std ,normalized_train

def to_even(TIME_STEPS):
    """
    To ensure the period is even and it is multipleir of 4
    """
    if TIME_STEPS % 2 != 0:
        TIME_STEPS -= 1
    if TIME_STEPS % 4 != 0:
        TIME_STEPS -= 2 
    return TIME_STEPS

# Generated training sequences for use in the model.
def create_sequences(values, time_steps):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

def create_model(x_train):
    model = keras.Sequential(
        [
            layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            layers.Conv1D(
                filters=32, kernel_size=5, padding="same", strides=2, activation="relu"
            ),      
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16, kernel_size=5, padding="same", strides=2, activation="relu"
            ),       
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=16, kernel_size=5, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=32, kernel_size=5, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ]
    )
    return model

def train_network(model,x_train,name,epochs,batch_size):
    
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
        

    epochs = 100
    batch_size = 32
    lr_scheduler = LearningRateScheduler(step_decay)
    optimizer = keras.optimizers.Adam(0.001)

    model.compile(optimizer=optimizer, loss="mse")
    
    history = model.fit(
        x_train,
        x_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            lr_scheduler
        ],
        verbose=0
    )
    # model.save('../Model/Autoencoder/'+name+'.h5')
    return history,model

def predict(model,test,training_mean,training_std,TIME_STEPS):
    normalized_test = (test- training_mean) / training_std

    # Create sequences from test values.
    x_test = create_sequences(normalized_test,TIME_STEPS)

    # Get test MAE loss.
    x_test_pred = model.predict(x_test)
    test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
    test_mae_loss = test_mae_loss.reshape((-1))
    return test_mae_loss

def predict(model,test,training_mean,training_std,TIME_STEPS):
    normalized_test = (test- training_mean) / training_std

    # Create sequences from test values.
    x_test = create_sequences(normalized_test,TIME_STEPS)

    # Get test MAE loss.
    x_test_pred = model.predict(x_test)
    test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
    test_mae_loss = test_mae_loss.reshape((-1))
    return test_mae_loss

def compute_acc_residuals(test_mae_loss,n,TIME_STEPS):
    acc_residuals = [0]*n

    start_index = 0
    for loss in test_mae_loss:
        for t in range(TIME_STEPS):
            acc_residuals[start_index+t] += loss
        start_index += 1 
        
    return acc_residuals

def get_suspicious_index(TIME_STEPS,acc_residuals):
    min_index = acc_residuals.index(min(acc_residuals[TIME_STEPS:-1-TIME_STEPS]))
    max_index = acc_residuals.index(max(acc_residuals[TIME_STEPS:-1-TIME_STEPS]))
    return min_index,max_index


for name in files_name[226::]:

    train,test = get_train_test(name)
    test = test.reset_index().drop('index',axis=1)
    n = len(test)

    training_mean,training_std,normalized_train = normalize(train)
    
    TIME_STEPS = int(period[period['File_name'] == name]['Period'])
    TIME_STEPS = to_even(TIME_STEPS)
    
    normalized_train = create_sequences(normalized_train,TIME_STEPS)
    
    
    
    ########Define Model#################
    model = create_model(normalized_train)

    epochs = 100
    batch_size = 32
    
    history,model = train_network(model,normalized_train,name,epochs,batch_size)
    
    test_mae_loss = predict(model,test,training_mean,training_std,TIME_STEPS)

    acc_residuals = compute_acc_residuals(test_mae_loss,n,TIME_STEPS)

    min_index,max_index = get_suspicious_index(TIME_STEPS,acc_residuals)

    # plt.style.use(['science','ieee','std-colors'])
    # fig = plt.figure(figsize=[16,4],dpi=200)
    # fig.patch.set_facecolor('#FFFFFF')
    # plt.grid(color='grey', linestyle='-.', linewidth=0.2)
    # plt.plot(test,color='blue',label='Normal Data')
    # plt.plot(range(max_index-100,min(max_index+100,n)),test[max_index-100:min(max_index+100,n)],color='red',label='Anomaly detected by maximum residual')
    # plt.legend()
    # plt.savefig('../fig/autoencoder/'+name+'.png',dpi=200)
    # plt.close()
    

    np.savetxt('../log/autoencoder/'+name+'.txt',acc_residuals)
    print(name," SUCCESS!")

    del model,test_mae_loss,acc_residuals
