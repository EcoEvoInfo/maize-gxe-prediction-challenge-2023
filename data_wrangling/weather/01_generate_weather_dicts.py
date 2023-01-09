#!/usr/bin/env python

# this script generates dictionaries storing weather input arrays for each LOYO training, validation, and test set
# running this on AHPCC, on comp06, using my-tensorflow-gpu-3.8 conda environment
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

### edit filepaths if needed
set = str(sys.argv[1])
df = pd.read_csv('../../Training_Data/4_Training_Weather_Data_2014_2021.csv', header=0)
test_df = pd.read_csv('../../Testing_Data/4_Testing_Weather_Data_2022.csv', header=0)

print(set)

######################################################
## get_one_env_array
##
## df:       input data frame
## features: list of feature names
## env:      specific environment
##
## returns: array (304 days x 11 features)
######################################################
def get_one_env_array(df, features, env):
    chanstack = list()
    df_long = df.loc[df.Env == env]
    for feature in features:
        df_wide = pd.pivot(df_long, index = ['Env'], columns = 'Date')[feature]   # pivot to 1 feature w/365 cols
        chanstack.append(df_wide.squeeze())
    chanstack = np.stack(chanstack, axis = -1)
    return chanstack

#######################################################
## make_dictionary
##
## df:       input data frame
## features: list of feature names
##
## returns: dictionary of arrays for each environment
#######################################################
def make_dictionary(df, features):
    dict = {}
    envs = df.Env.unique()
    for env in envs:
        arr = get_one_env_array(df, features, env)
        dict[env] = arr
    return dict

### split and process training, val, and test sets
df['Date'] = df['Date'].astype(str)
train_df = df[~df.Date.str.contains('0229')]                                       # remove extra day on leap years
train_df['Date'] = pd.to_datetime(train_df.Date)
train_df = train_df[(train_df.Date.dt.month != 12)&(train_df.Date.dt.month != 11)] # dec. and nov. missing in 2022
train_df = train_df.drop(['GWETTOP','GWETROOT','GWETPROF','ALLSKY_SFC_PAR_TOT','ALLSKY_SFC_SW_DNI'], axis = 1) # missing in the majority of 2022 data

val_df = train_df[train_df.Date.dt.year == int(set)]
train_df = train_df[train_df.Date.dt.year != int(set)]

test_df = test_df.drop(['GWETTOP','GWETROOT','GWETPROF','ALLSKY_SFC_PAR_TOT','ALLSKY_SFC_SW_DNI'], axis = 1) 
test_df.ffill(inplace = True)                                                   # fill missing w/value from previous day
test_df['Date'] = test_df['Date'].astype(str)
test_df['Date'] = pd.to_datetime(test_df.Date)
test_df = test_df[test_df.Date.dt.month != 11]                                  # dec. and nov. missing in 2022

### scale based on training set
features = list(train_df.keys())   
features = features[2:len(features)]                                            # remove id cols 

scaler = StandardScaler()
train_df.loc[:, features] = scaler.fit_transform(train_df.loc[:, features])     # scale training set
val_df.loc[:, features] = scaler.transform(val_df.loc[:, features])             # scale val set using sd and mean from train set
test_df.loc[:, features] = scaler.transform(test_df.loc[:, features])           # scale test set using sd and mean from train set

### create dictionary of arrays for each set
train = make_dictionary(train_df, features)
val = make_dictionary(val_df, features)
test = make_dictionary(test_df, features)

### save to pickle file
with open(f'pickles/loyo{set}_weather_train.pickle', 'wb') as f:
    pickle.dump(train, f)

with open(f'pickles/loyo{set}_weather_val.pickle', 'wb') as f:
    pickle.dump(val, f)

with open(f'pickles/loyo{set}_weather_test.pickle', 'wb') as f:
    pickle.dump(test, f)
