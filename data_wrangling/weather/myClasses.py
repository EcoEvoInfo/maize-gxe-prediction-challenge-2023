import numpy as np
import pandas as pd
import random
import glob
import tensorflow.keras as keras
from pandas import read_csv

class WeatherDataGenerator(keras.utils.Sequence):
    def __init__(self, labels, inputs, batch_size, shuffle=True):
        'Initialization'
        self.labels = labels        # a data frame containing labels
        self.inputs = inputs        # a dictionary of features for each env
        self.batch_size = batch_size
        self.num = self.labels.shape[0]
        self.idxs = list(range(0,self.num))
        self.shuffle = shuffle
        super().__init__()
        self.on_epoch_end()         # This line must be last (after all initialization takes place)

    def __len__(self):
        'Denotes the number of batches per epoch'
        bpe = int(np.floor(len(self.idxs)/self.batch_size))
        return bpe

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        bidxs = [i for i in range(index*self.batch_size, (index+1)*self.batch_size)]

        # Find list of IDs
        list_IDs_temp = [self.idxs[k] for k in bidxs] # these are the indices from the original, unshuffled list

        # Generate data
        X, y = self.data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.num)
        if self.shuffle is True:
            random.shuffle(self.idxs)

    def data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Generate array of images
        X_vals = list()
        for sampidx in list_IDs_temp:
            tmp = self.inputs[self.labels.Env.iloc[sampidx]]
            X_vals.append(tmp)

        X_vals = np.stack(X_vals)

        # Generate labels
        y_vals=list()
        for sampidx in list_IDs_temp:
            yld = self.labels.Yield_Mg_ha.iloc[sampidx]
            y_vals.append(yld)

        y_vals = np.stack(y_vals)

        return X_vals, y_vals
