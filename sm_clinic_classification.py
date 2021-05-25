# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
plt.rcParams['font.family'] = 'SimHei'

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, concatenate, GlobalAveragePooling2D, Reshape, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# training set
pos = pd.read_excel('positive.xlsx')
neg = pd.read_excel('negative.xlsx')

# preprocessing for the attribute "sex"
# 
#   male - 1
#   female - 0
#
neg['Sex'][neg['Sex'] == '女'] = 0
neg['Sex'][neg['Sex'] == '男'] = 1
pos['Sex'][pos['Sex'] == '女'] = 0
pos['Sex'][pos['Sex'] == '男'] = 1

# data format ready for modeling
x = np.concatenate((pos.values.astype(np.float32), neg.values.astype(np.float32)))
y = np.concatenate( (np.ones(len(pos)), np.zeros(len(neg))) )
print("total number of instances in ORIGINAL training set", len(x))

# preprocessing - normalization
scaler = StandardScaler().fit(x)
print('mean value of attributes', scaler.mean_, 'standard deviation of attributes', np.sqrt(scaler.var_))

# split of training set for model training and validation 
#x_train, x_valid, y_train, y_valid = train_test_split(scaler.transform(x),y,test_size=0.2,random_state=2021,stratify=y)
x_train = scaler.transform(x)
y_train = y
print("size of model training set as", x_train.shape)
#print("size of model validation set as", x_valid.shape)

# creation of DNN 
def create_mlp(input_dim, nlogits=1536, lr=0.0003, show_summary=True):
    model = Sequential()
    model.add(Dense(32, input_dim=input_dim, activation="relu"))
    model.add(Dense(nlogits, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=['acc'])

    if show_summary:
       model.summary()

    return model

# 保留权重
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'sm_clinic.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(filepath=filepath,
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True)
callbacks = [checkpoint]

# DNN initialization and training
clf = create_mlp(x_train.shape[1])
h = clf.fit(x_train,y_train,epochs=1000,batch_size=32,validation_split=0.15,shuffle=True,verbose=1,callbacks=callbacks)
print('performance of DNN in training set', clf.evaluate(x_train, y_train)[1])
