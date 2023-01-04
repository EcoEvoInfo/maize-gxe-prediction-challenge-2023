#!/usr/bin/env python

import pandas as pd
import numpy as np
import sys
import pickle
import tensorflow as tf

loyo = sys.argv[1]

## load trained model and inputs
model = tf.keras.models.load_model(f'models_{loyo}')

## function to extract output of penultimate dense layer
intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                       outputs=model.layers[5].output)

with open(f'pickles/loyo{loyo}_weather_train.pickle', 'rb') as f:
    train_features = pickle.load(f)

with open(f'pickles/loyo{loyo}_weather_test.pickle', 'rb') as f:
    test_features = pickle.load(f)

## 
f = open(f'loyo{loyo}_weather_reduced.txt', 'w')

for env in train_features.keys():
    input = np.expand_dims(train_features[env], axis = 0)
    feature = (intermediate_layer_model(input)).numpy()
    f.write(f'{env}\t')
    for i in range(0, feature.shape[1]):
        f.write(f'{feature[0, i]:.2f}\t')
    f.write("\n")


for env in test_features.keys():
    input = np.expand_dims(test_features[env], axis = 0)
    feature = (intermediate_layer_model(input)).numpy()
    f.write(f'{env}\t')
    for i in range(0, feature.shape[1]):
        f.write(f'{feature[0, i]:.2f}\t')
    f.write("\n")

f.close()
