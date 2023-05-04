#!/usr/bin/env python

import pandas as pd
import lightgbm as lgb
import sklearn as sk
import hyperopt
import math
import sys
import random
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

set = '2015'
max_depth = 3
reg_alpha = 27.0
reg_lambda = 93.0
min_child_weight = 4.0
col_sample_by_tree = 0.88
bagging_fraction = 1.0
learning_rate = 0.124
num_leaves = 85

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

model = lgb.LGBMRegressor(n_estimators = 300, bagging_fraction = bagging_fraction, max_depth = max_depth, reg_alpha = reg_alpha, reg_lambda = reg_lambda, min_child_weight = min_child_weight, colsample_bytree = col_sample_by_tree, num_leaves = num_leaves, learning_rate = learning_rate)
model.fit(X_train, Y_train, eval_set = evaluation, eval_metric="rmse", early_stopping_rounds=30, verbose=True)
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

model.booster_.save_model(f'lgb_loyo{set}_mod.txt')

## generate predictions for 2022
env_X = df.iloc[:,3:]
df[f'prediction_{set}'] = model.predict(env_X)
df[['Env','Hybrid','Yield_Mg_ha',f'prediction_{set}']].to_csv(f'predictions_loyo{set}.csv')

feature_imp = pd.DataFrame({'Value':model.feature_importances_,'Feature':X_train.columns})
feature_imp.sort_values(by=['Value'])
feature_imp.sort_values(by=['Value']).to_csv(f'feature_importance_loyo{set}.csv')

