import pandas as pd
import numpy as np

from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import matplotlib as mp
import os
pd.options.display.float_format = '{:.2f}'.format
# from prophet import Prophet
from croston import croston
import statsmodels.api as sm
import plotly
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,mean_absolute_error,r2_score
from pmdarima.arima.utils import ndiffs
from NBEATS import NeuralBeats
from pandas import DataFrame
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor
import calendar
from datetime import timedelta
import datetime as dt
#from autots import AutoTS
#import time


def data_preparation(df,n_split):
    df=df[df.iloc[:,0].notnull()] ## selecting all rows that has date values only
    df.columns=['date','value'] ##renaming the columns to date and value
    df.index=pd.to_datetime(df.iloc[:,0])
    df=df.iloc[:,1:]   
#     df=df.fillna(0) 
    df.index = pd.DatetimeIndex(df.index.values,
                                   freq=df.index.inferred_freq)
    ##drop na rows to get first and last values in data
    df=df.dropna()
    ## splitting data from last only
    df_test=df.tail(n_split)
    df_train=df[0:len(df)-n_split]
    print("pre processing done!")
    return(df,df_train,df_test)

# Accuracy metrics
def forecast_accuracy(forecast, actual):
    forecast,actual= np.array(forecast), np.array(actual) 
    try:
        mape = np.mean(np.abs((actual - forecast) / actual))*100
    except:
        mape = 200 # MAPE
    mae = mean_absolute_error(actual,forecast)    # MAE
    rmse = (mean_squared_error(actual,forecast))**.5  # RMSE
#     MASE=mase(actual,forecast)
    r2=r2_score(actual,forecast)                   # r-squared
    rtae=(mean_absolute_error(actual,forecast)/np.mean(abs(actual)))
    return({'mape':mape, 'mae': mae, 
            'rmse':rmse,
             'r2':r2,
            'rtae':rtae
           })
###Models below
def sarimax(df_train,n_split,df,h,df_test):
    model_test=pm.auto_arima(df_train,start_p=1,start_q=1, max_p=5, max_q=5, start_P=0, start_Q=0,
                        max_P=5, max_Q=5,out_of_sample_size=5,scoring='mse',
                        error_action='warn'
    )
    #univariate so exogenous variable as of now is none
    #######prediction for n_split steps
    forecast_sarimax_test=model_test.predict(n_periods=n_split,X=None,return_conf_int=False,m=12)
    ##model for entire data
    model=pm.auto_arima(df,start_p=1,start_q=1, max_p=5, max_q=5, start_P=0, start_Q=0,
                        max_P=5, max_Q=5,out_of_sample_size=5,scoring='mse',
                        error_action='warn')
    forecast_sarimax=model.predict(n_periods=h,X=None,return_conf_int=False,m=12)
    ##accuracy
    forecast_sarimax_test[forecast_sarimax_test<0]=0
    forecast_sarimax[forecast_sarimax<0]=0
    acc_arima=forecast_accuracy(forecast_sarimax_test[0:n_split], df_test)
    print("sarimax done!")
    return(forecast_sarimax,acc_arima)

# def prophet(df_train,n_split,df,h,df_test):
#     df_pr_test = df_train.copy()
#     df_pr=df.copy()
#     df_pr_test=df_pr_test.reset_index()
#     df_pr=df_pr.reset_index()
#     df_pr_test.columns = ['ds','y']
#     df_pr.columns = ['ds','y']
#     m = Prophet()
#     m.fit(df_pr_test)
#     future_test = m.make_future_dataframe(periods=n_split,freq='M',include_history=False)
#     forecast_prophet_test = m.predict(future_test)
#     m = Prophet()
#     m.fit(df_pr)
#     future= m.make_future_dataframe(periods=h,freq='M',include_history=False)
#     forecast_prophet = m.predict(future)
#     forecast_prophet[forecast_prophet['yhat']<0]=0
#     forecast_prophet_test[forecast_prophet_test['yhat']<0]=0
#     acc_prophet=forecast_accuracy(forecast_prophet_test['yhat'], df_test)
#     print("prophet done!")
#     return(forecast_prophet['yhat'],acc_prophet)

def etssp_best(df_train,n_split,df,h,df_test):
    # Simple exponential smoothing, denoted (A,N,N)
    mod_ses = sm.tsa.statespace.ExponentialSmoothing(df_train)
    ses=mod_ses.fit()
    forecast_ses=ses.forecast(n_split)
    # Holt's linear method, denoted (A,A,N)
    mod_hwl = sm.tsa.statespace.ExponentialSmoothing(df_train, trend=True)
    hwl = mod_hwl.fit()
    forecast_hwl=hwl.forecast(n_split)
    # Damped trend model, denoted (A,Ad,N)
    mod_hwdt = sm.tsa.statespace.ExponentialSmoothing(df_train, trend=True,damped_trend=True)                                                
    hwdt = mod_hwdt.fit()
    forecast_hwdt=hwdt.forecast(n_split)
    # Holt-Winters' trend and seasonality method, denoted (A,A,A)
    mod_hws = sm.tsa.statespace.ExponentialSmoothing(df_train, trend=True,seasonal=12)                                               
    hws = mod_hws.fit()
    forecast_hws=hws.forecast(n_split)
    a={"ses":forecast_accuracy(forecast_ses, df_test)['rmse'],
                   "hwl":forecast_accuracy(forecast_hwl, df_test)['rmse'],
                   "hwdt":forecast_accuracy(forecast_hwdt, df_test)['rmse'],
                   "hws":forecast_accuracy(forecast_hws, df_test)['rmse']
      }
    if min(a, key=a.get)=="ses":
        mod_ses = sm.tsa.statespace.ExponentialSmoothing(df)
        ses=mod_ses.fit()
        forecast_ets=ses.forecast(h)
        forecast_ets[forecast_ets<0]=0
        forecast_ses[forecast_ses<0]=0
        acc_ets=forecast_accuracy(forecast_ses, df_test)
    elif min(a, key=a.get)=="hwl":
        mod_hwl = sm.tsa.statespace.ExponentialSmoothing(df, trend=True)
        hwl = mod_hwl.fit()
        forecast_ets=hwl.forecast(h)
        forecast_ets[forecast_ets<0]=0
        forecast_hwl[forecast_hwl<0]=0
        acc_ets=forecast_accuracy(forecast_hwl, df_test)
    elif min(a, key=a.get)=="hwdt":
        mod_hwdt = sm.tsa.statespace.ExponentialSmoothing(df, trend=True,damped_trend=True)                            
        hwdt = mod_hwdt.fit()
        forecast_ets=hwdt.forecast(h)
        forecast_ets[forecast_ets<0]=0
        forecast_hwdt[forecast_hwdt<0]=0
        acc_ets=forecast_accuracy(forecast_hwdt, df_test)
    else: 
        mod_hws = sm.tsa.statespace.ExponentialSmoothing(df, trend=True,seasonal=12)                                               
        hws = mod_hws.fit()
        forecast_ets=hws.forecast(h)
        forecast_ets[forecast_ets<0]=0
        forecast_hws[forecast_hws<0]=0
        acc_ets=forecast_accuracy(forecast_hws, df_test)
    return(forecast_ets,acc_ets)
    
def crostn(df_train,n_split,df,h,df_test):
    fit_croston_test = croston.fit_croston(df_train, n_split,'original')
    fit_croston = croston.fit_croston(df, h,'original')
    forecast_croston_test=fit_croston_test['croston_forecast']
    forecast_croston=fit_croston['croston_forecast']
    forecast_croston[forecast_croston<0]=0
    forecast_croston_test[forecast_croston_test<0]=0
    acc_croston=forecast_accuracy(forecast_croston_test, df_test)
    forecast_croston=forecast_croston.flatten()
    print("croston done!")
    return(forecast_croston,acc_croston)

def nbeat(df_train,n_split,df,h,df_test):
    model = NeuralBeats(forecast_length=n_split,data=df_train.values,backcast_length=2*n_split)
    model.fit(plot=False,verbose=False)
    forecast_nbeats_test=model.predict(df_train.tail(n_split*2).values)
    forecast_nbeats_test[forecast_nbeats_test<0]=0
    acc_nbeats=forecast_accuracy(forecast_nbeats_test,df_test)
    model_nbeats=NeuralBeats(forecast_length=h,data=df.values,backcast_length=2*h)
    model_nbeats.fit(plot=False,verbose=False)
    forecast_nbeats=model_nbeats.predict(df.tail(h*2).values)
    forecast_nbeats[forecast_nbeats<0]=0
    return(forecast_nbeats.flatten(),acc_nbeats)



def xgboost(df_train,n_split,df,h,df_test):
    def add_month(df, forecast_length, forecast_period):
        dyn_date_df=pd.DataFrame({'date':pd.date_range(start=max(dat.date), periods=forecast_length, freq='M'),
                  'value':0})
        new_df=pd.concat([df,dyn_date_df], ignore_index=True)
        new_df['month']=pd.to_datetime(new_df['date'], format='%Y-%m-%d').dt.month
        new_df = new_df.drop(['date'], axis=1)
        return new_df
    def create_lag(df3):
        dataframe = DataFrame()
        for i in range(12, 0, -1):
            dataframe['t-' + str(i)] = df3.value.shift(i)
        df4 = pd.concat([df3, dataframe], axis=1)
        df4.dropna(inplace=True)
        return df4
    def randomForest(df1, forecast_length, forecast_period):
        df3 = df1[['value', 'date']]
        df3 = add_month(df3, forecast_length, forecast_period)
        finaldf = create_lag(df3)
        finaldf = finaldf.reset_index(drop=True)
        n = forecast_length
        end_point = len(finaldf)
        x = end_point - n
        finaldf_train = finaldf.loc[:x - 1, :]
        finaldf_train_x = finaldf_train.loc[:, finaldf_train.columns != 'value']
        finaldf_train_y = finaldf_train['value']
        print("Starting model train..")
        rfe = RFE(estimator=GradientBoostingRegressor(random_state=9),n_features_to_select=4)
        fit = rfe.fit(finaldf_train_x, finaldf_train_y)
        print("Model train completed..")
        print("Creating forecasted set..")
        yhat = []
        end_point = len(finaldf)
        n = forecast_length
        df3_end = len(df3)
        for i in range(n, 0, -1):
            y = end_point - i
            inputfile = finaldf.loc[y:end_point, :]
            inputfile_x = inputfile.loc[:, inputfile.columns != 'value']
            pred_set = inputfile_x.head(1)
            pred = fit.predict(pred_set)
            df3.at[df3.index[df3_end - i], 'value'] = pred[0]
            finaldf = create_lag(df3)
            finaldf = finaldf.reset_index(drop=True)
            yhat.append(pred)
        yhat = np.array(yhat)
        print("Forecast complete..")
        return yhat
    dat=df.reset_index()
    dat.columns=['date','value']
    dat_train=df_train.reset_index()
    dat_train.columns=['date','value']
    pred_test=randomForest(dat_train, n_split, 'Month')
    predicted_value=randomForest(dat, h, 'Month')
    pred_test[pred_test<0]=0
    predicted_value[predicted_value<0]=0
    acc_xgboost=forecast_accuracy(pred_test,df_test)
    return(predicted_value.flatten(),acc_xgboost)

def add_dates(df, forecast_length):
    dat=df.reset_index()
    dat.columns=['date','value']
    index=pd.date_range(start=max(dat.date), periods=forecast_length, freq='M')
    return index

def combine_forecasts(forecast_sarimax,forecast_croston,forecast_ets,forecast_nbeats,forecast_xgboost,value_name,df,h):
    strs_objnames=["sarimax","croston","ets","nbeats","xgboost"]
    obj_list=[forecast_sarimax,forecast_croston,forecast_ets,forecast_nbeats,forecast_xgboost]
    touse_outputs=[]
    touse_col_names=[]
    for i,obj in enumerate(obj_list):
        if obj is not None:
            touse_outputs.append(obj)
            touse_col_names.append(strs_objnames[i])
    comb_frcst=pd.DataFrame(data=touse_outputs).T
    print(comb_frcst)
    comb_frcst.columns=touse_col_names
    comb_frcst.index=add_dates(df,h)
    forecasts=comb_frcst.reset_index().melt(id_vars=['index'],
                             value_vars=touse_col_names,
                                            value_name=value_name)
    print("combining done!")
    return(forecasts)

def best_model_rmse_mape(acc_sarimax,acc_croston,acc_ets,acc_nbeats,acc_xgboost):
    # acc_arima,acc_prpt,acc_croston,acc_ets
    obj_list=[acc_sarimax,acc_croston,acc_ets,acc_nbeats,acc_xgboost]
    strs_objnames=["sarimax","croston","ets","nbeats","xgboost"]
    acc_rmse=dict()
    acc_mape=dict()
    for i,obj in enumerate(obj_list):
        if obj is not None:
            acc_rmse_temp={strs_objnames[i]:obj_list[i]['rmse']}
            acc_rmse.update(acc_rmse_temp)
            acc_mape_temp={strs_objnames[i]:obj_list[i]['mape']}
            acc_mape.update(acc_mape_temp)
    rmse_best=min(acc_rmse, key=acc_rmse.get)
    min_rmse=min(acc_rmse.values())
    mape_best=min(acc_mape, key=acc_mape.get)
    min_mape=min(acc_mape.values())
    return(rmse_best,min_rmse,mape_best,min_mape)
