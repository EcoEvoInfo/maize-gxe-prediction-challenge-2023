#!/usr/bin/env python

import pandas as pd
import numpy as np
import sys
import pickle
import tensorflow as tf

loyo = sys.argv[1]

#######################################################
## write_features
##
## dict:      dictionary of inputs for each environment 
## layer_out: function to extract penultimate layer output
## filename:  file handle for where to write
#######################################################
def write_features(dict, layer_out, filename):
    for env in dict.keys():
        input = np.expand_dims(dict[env], axis = 0)
        feature = (layer_out(input)).numpy()
        filename.write(f'{env}\t')
        for i in range(0, feature.shape[1]):
            filename.write(f'{feature[0, i]:.2f}\t')
        filename.write("\n")

## load trained model and inputs
model = tf.keras.models.load_model(f'best_models/models_{loyo}_geno')

## function to extract output of penultimate dense layer
intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                       outputs=model.layers[-2].output)

with open(f'pickles/loyo{loyo}_weather_train.pickle', 'rb') as f:
    train_features = pickle.load(f)

with open(f'pickles/loyo{loyo}_weather_val.pickle', 'rb') as f:
    val_features = pickle.load(f)

with open(f'pickles/loyo{loyo}_weather_test.pickle', 'rb') as f:
    test_features = pickle.load(f)

## write to file (train, val, test all in one file)
f = open(f'loyo{loyo}_weather_reduced.txt', 'w')

write_features(train_features, intermediate_layer_model, f)
write_features(val_features, intermediate_layer_model, f)
write_features(test_features, intermediate_layer_model, f)

f.close()

