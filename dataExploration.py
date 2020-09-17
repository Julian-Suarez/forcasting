#
# ------------------------------------------
# Author: Julian Suarez
# Date: Aug 20, 2020
#
# Descrpiton: This script had the purpose of exploring 
# and formatting the raw data extracted from 
# the Australian Bureau of Statistics for the Australian 
# employment data and Our World in Data for the Gov. Response
# data. 
#
# ------------------------------------------
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import auProjectFunctions as au
import warnings 
warnings.filterwarnings('ignore')

#Format Employment Data
uneployData = pd.read_csv('covid-19-data/worldUnemploymentData.py')
dropCols = ['INDICATOR', 'SUBJECT', 'MEASURE', 'FREQUENCY', 'Flag Codes']
uneployData.drop(dropCols,axis=1,inplace=True)
drop2019 = uneployData[uneployData.TIME == '2019-12'].index
uneployData.drop(drop2019,inplace=True)
dropConutryColection = ['EA19','EU27_2020','EU28','G-7','OECD']
for country in dropConutryColection:
    drop = uneployData[uneployData.LOCATION == country].index
    uneployData.drop(drop,inplace=True)
for country in au.countries.keys():
    countryIdx = np.array(uneployData[uneployData.LOCATION == country].index)
    uneployData.LOCATION[countryIdx] = au.countries[country]
outputFile = 'modelData/formattedUnemploymentData.csv'
if not os.path.isfile(outputFile):
    uneployData.to_csv(outputFile,index=False)
    print('Formatted Unemplyment Data was save at: {}'.format(outputFile))
else: 
    print('Formatted Unemployment Data was allready saved')

#Format Covid-19 Data
pd.set_option("display.max_rows", None, "display.max_columns", 4)
covidData = pd.read_csv('covid-19-data/covidGovResponse.csv')
formatedCovidData = pd.DataFrame()
for country in au.countries.values():
    tmpDf = covidData[covidData.Country == country]
    newColumnNames = ['Country', 'Year', 'School_closures',
                    'Workplace_Closures','Cancel_public_events',
                    'Restrictions_on_gatherings','Close_public_transport',
                    'Stay_at_home_requirements',
                    'Restrictions_on_internal_movement',
                    'International_travel_controls','Income_support',
                    'Debt/contract_relief','Fiscal_measures',
                    'International_support','Public_information_campaigns',
                    'Testing_policy','Contact_tracing',
                    'Emergency_investment_in_health_care',
                    'Investment_in_Vaccines','Stringency_Index']
    tmpDf.columns = newColumnNames
    tmpDf.insert(2,'Date',pd.date_range(start='1/1/2020',periods=len(tmpDf)),True)
    tmpDf.drop('Year',axis=1,inplace=True)
    tmpDf['Date'] = pd.to_datetime(tmpDf.Date) 
    #Group the data by month to join with emplyment data
    tmpDf = tmpDf.groupby(by=tmpDf.Date.dt.month).mean()
    tmpDf.insert(0,'Date',pd.date_range(start='1/1/2020',periods=len(tmpDf),freq='M'),True)
    tmpDf['Date'] = tmpDf.Date.dt.strftime('%Y-%m')
    tmpDf.insert(0,'Country',country,True)
    formatedCovidData = formatedCovidData.append(tmpDf,ignore_index=True)
outputFile = 'modelData/formattedCovidData.csv'
if not os.path.isfile(outputFile):
    formatedCovidData.to_csv(outputFile,index=False)
    print('Formatted Gov. Response Data was save at: {}'.format(outputFile))
else: 
    print('Formatted Gov. Response Data was allready saved')

formatedFinalData = pd.DataFrame()
for country in au.countries.values():
    tmp1 = uneployData[uneployData.LOCATION == country]
    tmp2 = formatedCovidData[formatedCovidData.Country == country]
    if len(tmp1) != len(tmp2):
        tmp2.drop(index=tmp2.index[len(tmp1):],inplace=True)
    tmp2.insert(2,'Value',np.array(tmp1.Value),True)
    formatedFinalData = formatedFinalData.append(tmp2,ignore_index=True)
outputFile = 'modelData/modelData.csv'
if not os.path.isfile(outputFile):
    formatedFinalData.to_csv(outputFile,index=False)
    print('Model Data was save at: {}'.format(outputFile))
else: 
    print('Model Data was allready saved')
