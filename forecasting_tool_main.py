#### created by : N sai Lalitha
#### creation date: 01/01/2022
#### last modified: 17/01/2022
#### objective: code runs univariate forecasting methods for a given dataset

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.2f}'.format

import importlib
import myfunction
# from myfunction import data_preparation,forecast_accuracy,sarimax,croston,etssp_best,prophet
importlib.reload(myfunction)
from myfunction import data_preparation,forecast_accuracy,sarimax,crostn,etssp_best,prophet,combine_forecasts,best_model_rmse_mape,nbeat

##parameters
h=12 #number of dat apoints to predict
n_split=5
#input data
data = pd.read_csv('sample data.csv')
print('data read into df')

def overall_function(data,n_split,h,value_name):
    ## preprocessing data
    df,df_train,df_test=data_preparation(data,n_split)
    forecast_sarimax,acc_sarimax=sarimax(df_train,n_split,df,h,df_test)
    forecast_croston,acc_croston=crostn(df_train,n_split,df,h,df_test)
    forecast_ets,acc_ets=etssp_best(df_train,n_split,df,h,df_test)
    forecast_prophet,acc_prophet=prophet(df_train,n_split,df,h,df_test)
    forecast_nbeats,acc_nbeats=nbeat(df_train,n_split,df,h,df_test)
    forecasts=combine_forecasts(forecast_sarimax,forecast_prophet,forecast_croston,forecast_ets,forecast_nbeats,value_name)
    best_models=best_model_rmse_mape(acc_sarimax,acc_croston,acc_ets,acc_prophet,acc_nbeats)
    return(forecasts,best_models)

start = time.time()
forecasts_final=list()
bm=[]
for i in range(1,data.shape[1]):
    print(i)
    dat=data.iloc[:,[0,i]]
    value_name=data.columns[i]
    forecasts,best_models=overall_function(dat,n_split,h,value_name)
    forecasts_final.append(forecasts)
    bm.append([value_name,best_models[0],best_models[1],best_models[2],best_models[3]])
end=time.time()
iter_time=end-start
print('iteration done!')
print(iter_time)

## export forecast fter concatinating results
forecast_op=pd.concat(forecasts_final,axis=1)
forecast_op.rename(columns={'index':'Date','variable':'Model'},inplace=True)
forecast_op=forecast_op.loc[:,~forecast_op.columns.duplicated()]
## replace negative forecasts by zero
forecast_op.to_csv("forecast_test.csv")

bm_df=pd.DataFrame(bm, columns = ['model', 'rmse_best','min_rmse','mape_best','min_mape'])
bm_df.to_csv("bestmodels.csv")
