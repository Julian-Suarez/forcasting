#
# ------------------------------------------
# Author: Julian Suarez
# Date: Sep 16, 2020
#
# Descrpiton: This script makes used of the
# output of dataExploration.py - modelData
# to train an optimize three XGBRegressor models
# then output the trianed models into json files.
#
# ------------------------------------------
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import auProjectFunctions as au 
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# Load training data and drop Australia data
covidData = pd.read_csv('modelData/modelData.csv')
auCovidData = covidData[covidData.Country == 'Australia']
covidData.drop(covidData[covidData.Country == 'Australia'].index,inplace=True)
au.updateIndex(covidData)
# Define target and predictor metrics, as well as output files for all models (before and after parameter tuning)
target = 'Value'
predictors = [x for x in covidData.columns if x != target and x != 'Country' and x != 'Date']
outputFiles = [('modelData/sleModel_v1.json','modelData/sleModel.json'),('modelData/seModel_v1.json','modelData/seModel.json'),('modelData/hubereModel_v1.json','modelData/hubereModel.json')]

# Define XGBRegressor with objective function squaredlogerror
xgb1 = XGBRegressor(
        learning_rate =0,
        n_estimators=1000,
        max_depth=5,
        booster='gbtree',
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'reg:squaredlogerror',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
print('Report for squared log error model')
au.optimizeParams(xgb1,covidData,predictors,target,saveModel=outputFiles[0])
print('End of Report\n')

# Define XGBRegressor with objective function squarederror
xgb2 = XGBRegressor(
        learning_rate =0,
        n_estimators=1000,
        max_depth=5,
        booster='gbtree',
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'reg:squarederror',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
print('Report for squared error model')
au.optimizeParams(xgb2,covidData,predictors,target,saveModel=outputFiles[1])
print('End of Report\n')

# Define XGBRegressor with objective function pseudohubererror
xgb3 = XGBRegressor(
        learning_rate =0,
        n_estimators=1000,
        max_depth=5,
        booster='gbtree',
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'reg:pseudohubererror',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
print('Report for pseudo huber error model')
au.optimizeParams(xgb3,covidData,predictors,target,saveModel=outputFiles[2])
print('End of Report\n')

