# from Prophet import
# AQI - 02-03-2024 - first iter 
from prophet import *
from prophet.plot import *
from prophet.serialize import model_to_json, model_from_json

import pandas as pd
from prophet import Prophet


def prophet_AQI(city):

  data = pd.read_csv('{}.csv'.format(city))


  data['time'] = pd.to_datetime(data['dt'], dayfirst=True)
  data['aqi'] = data['main.aqi'].interpolate(method='linear')
  data_prophet = data[['time', 'aqi']].rename(columns={'time': 'ds', 'aqi': 'y'})

  split_date = '2022-01-01'
  train = data_prophet[data_prophet['ds'] < split_date]
  test = data_prophet[data_prophet['ds'] >= split_date]

  model_optimized = Prophet(
      changepoint_prior_scale=0.01,
      seasonality_prior_scale=1.0,
  )
  model_optimized.fit(data_prophet)



  future = model_optimized.make_future_dataframe(periods=365) # horizon
  future['yearly'] = future['ds'].apply(lambda x: x.year - 1) # addition
  forecast = model_optimized.predict(future)
  with open('./{}.json'.format(city), 'w') as fout:
      fout.write(model_to_json(model_optimized))  # Save model
  print("model saved")
  forecast.to_csv('./{}_.csv'.format(city), index=False)
  print("prediction file saved")


prophet_AQI('bangalore_aqi_csv')

def prophet_temp(city):

  data = pd.read_csv('{}.csv'.format(city))
  data['time'] = pd.to_datetime(data['time'], dayfirst=True)
  data['tmax'] = data['tmax'].interpolate(method='linear')
  data_prophet = data[['time', 'tmax']].rename(columns={'time': 'ds', 'tmax': 'y'})


  split_date = '2022-01-01'
  train = data_prophet[data_prophet['ds'] < split_date]
  test = data_prophet[data_prophet['ds'] >= split_date]


  model_optimized = Prophet(
    changepoint_prior_scale = 0.5,
      seasonality_prior_scale = 0.1
      
  )
  model_optimized.fit(train)

  future = model_optimized.make_future_dataframe(periods=365) # horizon
  future['yearly'] = future['ds'].apply(lambda x: x.year - 1) # addition
  forecast = model_optimized.predict(future)

  with open('./{}.json'.format(city), 'w') as fout:
      fout.write(model_to_json(model_optimized))  # Save model
  print("model saved")
  forecast.to_csv('./{}_.csv'.format(city), index=False)
  print("prediction file saved")


prophet_temp('bangalore_temp_csv')