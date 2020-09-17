#
# ------------------------------------------
# Author: Julian Suarez
# Date: Sep 16, 2020
#
# Descrpiton: This script holds functions used
# for Australian Unemployment Forcasting Machine
# Learning project
#
# ------------------------------------------
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

#
# ------------------------------------------
# Description: This function produces a pyplot
# graph from a pandas df and an array of predictions.
#
# Parameters:   df,                 Pandas DataFrame, the DataFrame
#                                   needs a Country, Value, and Date
#                                   column.
#               predictions,        A numpy array with predicitons 
#                                   outputed by trained models.
#                                   This array has to be the same size 
#                                   as df.
#               countries,          Array containing strings with the 
#                                   countries interested on plotting.
#               graphAll,           Boolean determeaning if you want to 
#                                   graph all countries at once.
#                                   (Requires proper implementation)
# ------------------------------------------
#
def graph(df,predictions=[],countries=None,graphAll=False): 
    plt.style.use('seaborn')
    fig,ax = plt.subplots() 
    if graphAll:
        for country in au.countries.values():
            ax.plot(df.Date[df.Country == country],df.Value[df.Country == country],label=country)
            if predictions.size:
                countryIdx = np.array(df[df.Country == country].index)
                ax.plot(df.Date[df.Country == country],predictions[countryIdx],label=country+' Prediction')
    else:
        if countries == None:
            countries=['United States','Japan','Germany']
        for country in countries:
            ax.plot(df.Date[df.Country == country],df.Value[df.Country == country],'o-',label=country)
            if predictions.size:
                countryIdx = np.array(df[df.Country == country].index)
                ax.plot(df.Date[df.Country == country],predictions[countryIdx],'^--',label=country+' Prediction')
    ax.set_title('Unemplment Rate over time')
    ax.legend(loc='upper right')
    plt.xlabel('Date [Month]')
    plt.ylabel('Unemployment Rate [%]')
    plt.show()
    return

#
# ------------------------------------------
# Description: This is a helper function to train
# a XGBRegressor algorithm. It has an option 
# to train the model making use of cross validation
# technique. This function trains the model with
# root mean square error metric.
#
# Parameters:   alg,                        Algorithm to be trained.
#               dtrain,                     A pandas DataFrame with the 
#                                           data set to train the algorithm.
#               predictors,                 An array with the names of the 
#                                           predictors metrics.
#               target,                     A string with the name of 
#                                           the target metric.
#               cv_folds,                   Optional parameter to set the 
#                                           number of folds used for cross validation.
#               early_stopping_rounds,      Parameter to set the omptimal 
#                                           number of iterations. 
#               countries (optional),       Parameter used for the graph fucntion.
#               verbose (optional),         Parameter to print into terminal the
#                                           rmse result of the training and to graph
#                                           the prediction and the original results.
#               useTrainCV (optional),      Parameter to toggle cross validation training
#               saveModel (optional),       String with the path to save the trained algorithm.
#
# Returning:    Returns the root mean square error results of the model.
# ------------------------------------------
#
def modelfit(alg, dtrain, predictors, target, cv_folds=5, early_stopping_rounds=50, countries= None, verbose=False, useTrainCV=True, saveModel=None):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresults = xgb.cv(xgb_param,xgtrain,num_boost_round=alg.get_params()['n_estimators'],nfold=cv_folds,
                            metrics=['rmse'],early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresults.shape[0])
        
    #Fit algorithm to data
    alg.fit(dtrain[predictors],dtrain[target],eval_metric='rmse')

    #Predict training set
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Print Model Report
    if verbose:
        print('Model Report')
        print('root_mean_squared_error: {:.4f}'.format(np.sqrt(mean_squared_error(dtrain[target],dtrain_predictions))))
        graph(dtrain,dtrain_predictions,countries=countries)

    #Save model
    if saveModel:
        print('Saving model on: {}'.format(saveModel))
        alg.save_model(saveModel)

    return np.sqrt(mean_squared_error(dtrain[target],dtrain_predictions))

#
# ------------------------------------------
# Description:  This function optimizes an algorith by iterative methods
#               and making use of the GridSearchCV function. The fucntion
#               begins by iterativley finding the best learing rate. 
#               Then does four GridSearchCV parameter seting after which 
#               it prints the rmse result from the chosen paramter.
#
# Parameters:   xgb,                        Algorithm to be trained.
#               dtrain,                     A pandas DataFrame with the 
#                                           data set to train the algorithm.
#               predictors,                 An array with the names of the 
#                                           predictors metrics.
#               target,                     A string with the name of 
#                                           the target metric.
#               countries (optional),       Parameter used for the graph fucntion.
#               saveModel (optional),       String with the path to save the trained algorithm.
# ------------------------------------------
#
def optimizeParams(xgb,dtrain,predictors,target,countries=None,saveModel=None):
    # Find optimal Learning Rate, and n_estimator
    bestLR = 0
    bestMSE = 10
    for i in np.arange(0.01,0.3,0.01):
        xgb.set_params(learning_rate=i)
        runMSE = modelfit(xgb, dtrain, predictors, target, verbose=False)
        if runMSE < bestMSE:
            bestMSE = runMSE
            bestLR = i
    xgb.set_params(learning_rate=bestLR)
    if saveModel:
        modelfit(xgb,dtrain,predictors,target,countries=countries,verbose=True,saveModel=saveModel[0])
    else:
        modelfit(xgb,dtrain,predictors,target,countries=countries,verbose=True)

    #Tune max_depth and min_child_weigth
    param_test1 = {
            'max_depth':range(0,10,2),
            'min_child_weight':range(1,6,2)
    }

    gsearch1 = GridSearchCV(estimator = xgb,
                            param_grid = param_test1,
                            scoring = 'neg_mean_squared_error',
                            n_jobs = 4,
                            cv = 5)
    gsearch1.fit(dtrain[predictors],dtrain[target])
    print('max depth and min_child_weight root mean squared error')
    print(np.sqrt(-1*gsearch1.best_score_))
    xgb.set_params(max_depth=gsearch1.best_params_['max_depth'],min_child_weight=gsearch1.best_params_['min_child_weight'])

    #Tune Gamma Parameter
    param_test2 = {
            'gamma':np.arange(0.1,1.1,0.1)
    }
    gsearch2 = GridSearchCV(estimator = xgb,
                            param_grid = param_test2,
                            scoring = 'neg_mean_squared_error',
                            n_jobs = 4,
                            cv = 5)
    gsearch2.fit(dtrain[predictors],dtrain[target])
    print('Gamma Parameter root mean squared error')
    print(np.sqrt(-1*gsearch1.best_score_))
    xgb.set_params(gamma=gsearch2.best_params_['gamma'])

    #Tune subsample and colsample_bytree 
    param_test3 = {
            'subsample':np.arange(0.6,0.8,0.01),
            'colsample_bytree':np.arange(0.6,0.8,0.01)
    }

    gsearch3 = GridSearchCV(estimator = xgb,
                            param_grid = param_test3,
                            scoring = 'neg_mean_squared_error',
                            n_jobs = 4,
                            cv = 5)
    gsearch3.fit(dtrain[predictors],dtrain[target])
    print('Subsample and colsample_bytree root mean squared error')
    print(np.sqrt(-1*gsearch1.best_score_))
    xgb.set_params(subsample=gsearch3.best_params_['subsample'],colsample_bytree=gsearch3.best_params_['colsample_bytree'])

    #Tuning regularization parameters
    param_test4 = {
            'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
    }

    gsearch4 = GridSearchCV(estimator = xgb,
                            param_grid = param_test4,
                            scoring = 'neg_mean_squared_error',
                            n_jobs = 4,
                            cv = 5)
    gsearch4.fit(dtrain[predictors],dtrain[target])
    print('Regularization root mean squared error')
    print(np.sqrt(-1*gsearch1.best_score_))
    xgb.set_params(reg_alpha=gsearch4.best_params_['reg_alpha'])
    modelfit(xgb,dtrain,predictors,target,countries=countries,verbose=True)

    #Readjust the learning rate for the optimize model and save final traind algorithm
    bestLR = 0
    bestMSE = 10
    for i in np.arange(0.01,0.3,0.01):
        xgb.set_params(learning_rate=i)
        runMSE = modelfit(xgb, dtrain, predictors, target)
        if runMSE < bestMSE:
            bestMSE = runMSE
            bestLR = i
    xgb.set_params(learning_rate=bestLR)
    if saveModel:
        modelfit(xgb,dtrain,predictors,target,countries=countries,verbose=True,saveModel=saveModel[1])
    else:
        modelfit(xgb,dtrain,predictors,target,countries=countries,verbose=True)
    return

#
# ------------------------------------------
# Description:  Helper function to update the index
#               of a pandas DataFrame
#
# Parameters:   dtrain,         A pandas DataFrame whos 
#                               index is to be updated.
# ------------------------------------------
#
def updateIndex(df):
    df.set_index(np.arange(len(df)),inplace=True)
    return

#
# ------------------------------------------
# Description:  Python dictionary with all of the countries
#               used by the training data.
# ------------------------------------------
#
countries = {'AUS':'Australia', 'AUT':'Austria', 'BEL':'Belgium', 'CAN':'Canada', 'CHL':'Chile',
        'COL':'Colombia', 'CZE':'Czech Republic', 'DEU':'Germany', 'DNK':'Denmark',
        'ESP':'Spain', 'EST':'Estonia', 'FIN':'Finland', 'FRA':'France',
        'GBR':'United Kingdom', 'GRC':'Greece', 'HUN':'Hungary', 'IRL':'Ireland', 
        'ISL':'Iceland', 'ISR':'Israel', 'ITA':'Italy', 'JPN':'Japan',
        'KOR':'South Korea', 'LTU':'Lithuania', 'LUX':'Luxembourg', 'LVA':'Latvia',
        'MEX':'Mexico', 'NLD':'Netherlands', 'NOR':'Norway', 'POL':'Poland', 
        'PRT':'Portugal', 'SVK':'Slovakia', 'SVN':'Slovenia', 'SWE':'Sweden', 'TUR':'Turkey',
        'USA':'United States'}
