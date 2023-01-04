#!/usr/bin/env python

import pandas as pd
import numpy as np
import pickle
import sys
import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization
from myClasses import WeatherDataGenerator

year = '2022'

#######################
## define architecture
#######################
model = Sequential()
model.add(Conv1D(filters = 124, kernel_size = 6, activation = 'relu', input_shape = (304, 11))) # 365 days with 16 features
model.add(Conv1D(filters = 124, kernel_size = 6, activation = 'relu')) # 365 days with 16 features
model.add(Dropout(0.3))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(30, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

##########################
## load data for training
##########################
## inputs are present in the Weather file
with open(f'pickles/loyo{year}_weather_train.pickle', 'rb') as f:
    train_features = pickle.load(f)

with open(f'pickles/loyo{year}_weather_test.pickle', 'rb') as f:
    test_features = pickle.load(f)

## labels are present in the Traits file
df = pd.read_csv('../Training_Data/1_Training_Trait_Data_2014_2021.csv', header=0)
df = df.dropna(subset=['Yield_Mg_ha'])

## split into ~20/80 validation set by environment
random.seed(42)
envs_20 = random.sample(train_features.keys(), 44)  # list of 20 random environments
envs_80 = list(train_features.keys() - envs_20)     # list of remaining

train_labels = df.loc[df.Env.isin(envs_80)]
val_labels = df.loc[df.Env.isin(envs_20)]

batch_size_train = 64
batch_size_test = 64

train_generator = WeatherDataGenerator(train_labels, train_features, batch_size_train)
val_generator = WeatherDataGenerator(val_labels, train_features, batch_size_test)

##########################
## training
##########################
EPOCHS = 250
checkpoint_filepath = f'models_{year}/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_mse',
    mode='min',
    save_best_only=True)

model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, verbose=2, callbacks=[model_checkpoint_callback])
