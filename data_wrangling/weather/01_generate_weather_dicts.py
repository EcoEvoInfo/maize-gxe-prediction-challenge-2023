# this script generates dictionaries storing weather input arrays for each LOYO training fold and test set
# running this on AHPCC, on comp06, using my-tensorflow-gpu-3.8 conda environment
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

############################################
## get_one_env_array
##
## df:       input data frame
## features: list of feature names
## env:      specific environment
##
## returns: array (365 days x 11 features)
############################################
def get_one_env_array(df, features, env):
    chanstack = list()
    df_long = df.loc[df.Env == env]
    for feature in features:
        df_wide = pd.pivot(df_long, index = ['Env'], columns = 'Date')[feature] # pivot to 1 feature w/365 cols
        chanstack.append(df_wide.squeeze())
    chanstack = np.stack(chanstack, axis = -1)
    return chanstack

### get input training data, and scale
df = pd.read_csv('../Training_Data/4_Training_Weather_Data_2014_2021.csv', header=0)
df['Date'] = df['Date'].astype(str)
train_df = df[~df.Date.astype(str).str.contains('0229')]                           # remove extra day on leap years
train_df = train_df.drop(['GWETTOP','GWETROOT','GWETPROF','ALLSKY_SFC_PAR_TOT','ALLSKY_SFC_SW_DNI'], axis = 1) # missing in the majority of 2022 data
train_df['Date'] = pd.to_datetime(train_df.Date)
train_df = train_df[(train_df.Date.dt.month != 12)&(train_df.Date.dt.month != 11)] # dec. and nov. missing in 2022

features = list(train_df.keys())   
features = features[2:len(features)]                                            # remove first two element 
scaler = StandardScaler()
train_df.loc[:, features] = scaler.fit_transform(train_df.loc[:, features])     # scale training set

test_df = pd.read_csv('../Testing_Data/4_Testing_Weather_Data_2022.csv', header=0)
test_df = test_df.drop(['GWETTOP','GWETROOT','GWETPROF','ALLSKY_SFC_PAR_TOT','ALLSKY_SFC_SW_DNI'], axis = 1) 
test_df.ffill(inplace = True)                                                   # fill missing w/value from previous day
test_df['Date'] = test_df['Date'].astype(str)
test_df['Date'] = pd.to_datetime(test_df.Date)
test_df = test_df[test_df.Date.dt.month != 11]                                  # dec. and nov. missing in 2022
test_df.loc[:, features] = scaler.transform(test_df.loc[:, features])           # scale test set using sd and mean from train set

### create dictionary of arrays for train set
train = {}
envs = train_df.Env.unique()
for env in envs:
    arr = get_one_env_array(train_df, features, env)
    train[env] = arr

### create dictionary of arrays for test set
test = {}
envs = test_df.Env.unique()
for env in envs:
    arr = get_one_env_array(test_df, features, env)
    test[env] = arr

### save to pickle file
with open(f'loyo2022_weather_train.pickle', 'wb') as f:
    pickle.dump(train, f)

with open(f'loyo2022_weather_test.pickle', 'wb') as f:
    pickle.dump(test, f)


