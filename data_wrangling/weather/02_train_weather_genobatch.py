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
from myClasses_genobatch import GenoBatchWeatherDataGenerator

year = str(sys.argv[1])
df = pd.read_csv('../../Training_Data/1_Training_Trait_Data_2014_2021.csv', header=0)

#######################
## define architecture
#######################
model = Sequential()
model.add(Conv1D(filters = 124, kernel_size = 3, activation = 'relu', input_shape = (304, 11))) # 365 days with 16 features
model.add(Conv1D(filters = 124, kernel_size = 3, activation = 'relu'))
model.add(Dropout(0.3))
model.add(MaxPooling1D(3))
#model.add(Conv1D(filters = 124, kernel_size = 3, activation = 'relu'))
#model.add(Conv1D(filters = 124, kernel_size = 3, activation = 'relu'))
#model.add(Dropout(0.3))
#model.add(MaxPooling1D(3))
#model.add(Conv1D(filters = 124, kernel_size = 3, activation = 'relu'))
#model.add(Conv1D(filters = 124, kernel_size = 3, activation = 'relu'))
#model.add(Dropout(0.3))
#model.add(MaxPooling1D(3))
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

with open(f'pickles/loyo{year}_weather_val.pickle', 'rb') as f:
    val_features = pickle.load(f)

## labels are present in the Traits file
df = df.dropna(subset=['Yield_Mg_ha'])

envs = train_features.keys()
train_labels = df.loc[df.Env.isin(envs)]

envs = val_features.keys()
val_labels = df.loc[df.Env.isin(envs)]

hybrids = train_labels.Hybrid.unique()
batch_size = 64

train_generator = GenoBatchWeatherDataGenerator(train_labels, train_features, hybrids)
val_generator = WeatherDataGenerator(val_labels, val_features, batch_size)

##########################
## training
##########################
EPOCHS = 300
checkpoint_filepath = f'models_{year}_geno/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_mse',
    mode='min',
    save_best_only=True)

model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, verbose=2, callbacks=[model_checkpoint_callback])
