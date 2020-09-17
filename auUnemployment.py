#
# ------------------------------------------
# Author: Julian Suarez
# Date: Sep 16, 2020
#
# Descrpiton: This script loads the 
# output models from boost.py and creates the 
# final predicitve model, as well as simulating 
# a second wave of lockdown measures.
#
# ------------------------------------------
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import auProjectFunctions as au 
import copy
from xgboost import XGBRegressor

# Load the training data and isolate Australias data
# Defines target and predictor metrics
covidData = pd.read_csv('modelData/modelData.csv')
auCovidData = covidData[covidData.Country == 'Australia']
target = 'Value'
predictors = [x for x in covidData.columns if x != target and x != 'Country' and x != 'Date']

# Define a dictionary to store the input models
keyModels = ['sle','se','hubere']
models = {}
for i in keyModels:
    models[i] = XGBRegressor()

# Load the input models into the models dictionary
inputModels = ['modelData/sleModel_v1.json','modelData/seModel.json','modelData/hubereModel.json']
i = 0
for model in models:
    models[model].load_model(inputModels[i])
    i += 1

# Poduce and store each models predictions into
# predictions dictionary.
# Graph predictions with au.graph.
predictions = copy.deepcopy(models)
for model in models:
    predictions[model] = models[model].predict(auCovidData[predictors])
    au.graph(auCovidData,predictions[model],countries=['Australia'])

# Define a Pandas DataFrame with the country, date, value 
# fron original Australian unemployment data and with the
# predicitons made by the input models.
# Also re-define predictors metrics.
df = pd.DataFrame(list(zip(auCovidData.Country,
                            auCovidData.Date,
                            auCovidData.Value,
                            predictions['sle'],
                            predictions['se'],
                            predictions['hubere'])),
                            columns = ['Country','Date','Value','sle','se','hubere'])
predictors = [x for x in df.columns if x != target and x != 'Country' and x != 'Date']

# Define final model and train with ~70% - ~30% split
# and use optimizeParams from auProjectFunctions.py.
# Produce final predictions and graph with au.graph.
final_model =  XGBRegressor(
            learning_rate =0,
            n_estimators=1000,
            max_depth=5,
            booster='gbtree',
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            nthread=4,
            scale_pos_weight=1,
            seed=27)
au.optimizeParams(final_model,df.iloc[:5],predictors,target,countries=['Australia'])
finalPredictions = final_model.predict(df[predictors])
au.graph(df,finalPredictions,countries=['Australia'])

# Produce a plot wirh all countries data
fig,ax = plt.subplots()
for country in au.countries.values():
            ax.plot(covidData.Date[covidData.Country == country],covidData.Value[covidData.Country == country],label=country)
ax.set_title('Unemployment over Time')
ax.legend(loc='upper right')
plt.xlabel('Date')
plt.ylabel('Unemployment')
plt.show()

