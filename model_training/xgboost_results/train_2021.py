#!/usr/bin/env python

import pandas as pd
import xgboost as xgb
import sklearn as sk
import hyperopt
import math
import sys
import random
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

set = '2021'

max_depth = 8
gamma = 5.87
reg_alpha = 161.0
reg_lambda = 106.0
min_child_weight = 6.0
col_sample_by_tree = 0.66

df = pd.read_csv(f'../../combo_dfs/combo_loyo_{set}.csv')

## split into train and val sets
df_val = df[df.Env.str.endswith(set)]
df_train = df[~df.Env.isin(df_val.Env)]
df_train = df_train[~df_train.Env.str.endswith('2022')]
df_test = df[df.Env.str.endswith('2022')]

X_train = df_train.iloc[:,3:]
Y_train = df_train.iloc[:,2]
X_val = df_val.iloc[:,3:]
Y_val = df_val.iloc[:,2]
X_test = df_test.iloc[:,3:]
Y_test = df_test.iloc[:,2]

evaluation = [(X_train, Y_train), (X_val, Y_val)]

model = xgb.XGBRegressor(n_estimators = 300, max_depth = max_depth, gamma = gamma, reg_alpha = reg_alpha, reg_lambda = reg_lambda, min_child_weight = min_child_weight, colsample_bytree = col_sample_by_tree)
model.fit(X_train, Y_train, eval_set = evaluation, eval_metric="rmse", early_stopping_rounds=10, verbose=True)
y_pred = model.predict(X_val)

print(f'Overall RMSE for {set}: {math.sqrt(sk.metrics.mean_squared_error(Y_val, y_pred)):.2f}')

## calculate mean RMSE as average of environments
envs = list(df_val.Env.unique())
envs_rmse = []

for env in envs:
    env_X = df_val[df_val.Env == env].iloc[:,3:]
    env_Y = df_val[df_val.Env == env].iloc[:,2]
    y_pred = model.predict(env_X)
    val = math.sqrt(sk.metrics.mean_squared_error(env_Y, y_pred))
    envs_rmse.extend([val])

print(f'Environment Average RMSE for {set}: {(sum(envs_rmse)/len(envs_rmse)):.2f}')

model.save_model(f'xgb_loyo{set}_mod.txt')

## generate predictions for 2022
env_X = df.iloc[:,3:]
df[f'prediction_{set}'] = model.predict(env_X)
df[['Env','Hybrid','Yield_Mg_ha',f'prediction_{set}']].to_csv(f'predictions_loyo{set}.csv')

feature_imp = pd.DataFrame({'Value':model.feature_importances_,'Feature':X_train.columns})
feature_imp.sort_values(by=['Value'])
feature_imp.sort_values(by=['Value']).to_csv(f'feature_importance_loyo{set}.csv')

