#!/usr/bin/env python

import pandas as pd
import lightgbm as lgb
import sklearn as sk
import hyperopt
import sys
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

set = sys.argv[1]

df = pd.read_csv(f'../../combo_dfs/combo_loyo_{set}.csv')

## split into test and train sets
df_val = df[df.Env.str.endswith(set)]
df_train = df[~df.Env.isin(df_val.Env)]
df_train = df_train[~df_train.Env.str.endswith('2022')]

X_train = df_train.iloc[:,3:]
Y_train = df_train.iloc[:,2]
X_val = df_val.iloc[:,3:]
Y_val = df_val.iloc[:,2]

## initialize a search space of hyperparameters
space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
    'learning_rate': hp.loguniform('learning_rate', -4, -1),
    'num_leaves': hp.quniform('num_leaves', 30, 100, 1),
    'reg_alpha' : hp.quniform('reg_alpha', 0, 180, 1),
    'reg_lambda' : hp.quniform('reg_lambda', 0, 180, 1),
    'colsample_bytree' : hp.uniform('colsample_bytree', 0.1,0.9),
    'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
    'bagging_fraction' : hp.quniform('bagging_fraction', 0.5, 1.0, 0.1),
    'n_estimators': 300
}

## define objective function
def hyperparameter_tuning(space):
    model=lgb.LGBMRegressor(n_estimators =space['n_estimators'], bagging_fraction = space['bagging_fraction'], max_depth = int(space['max_depth']), learning_rate=space['learning_rate'], num_leaves = int(space['num_leaves']), reg_alpha = space['reg_alpha'], reg_lambda = space['reg_lambda'], min_child_weight=space['min_child_weight'], colsample_bytree=space['colsample_bytree'])
    evaluation = [(X_train, Y_train), (X_val, Y_val)]
    model.fit(X_train, Y_train, eval_set=evaluation, eval_metric="mse", early_stopping_rounds=30, verbose=False)
    pred = model.predict(X_val)
    mse = sk.metrics.mean_squared_error(Y_val, pred)
    print ("SCORE:", mse)
    return {'loss':mse, 'status': STATUS_OK, 'model': model}

## train model
trials = Trials()
best = fmin(fn=hyperparameter_tuning, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

print(best)
