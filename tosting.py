import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.layers import Bidirectional
import tensorflow as tf
import pandas as pd
import quandl
from alpha_vantage.timeseries import TimeSeries
from tensorflow import keras
import pickle
import os
import datetime as dt
import streamlit
comp = 'IBM'




ts = TimeSeries(key='496xBxfk3x-WYLS1vjmz', output_format='pandas')
df = ts.get_daily(symbol=comp, outputsize='full')[0]

df = df.sort_index(ascending=True, axis=0)
df = df[-365:]
y = df['4. close']
#print(y)
y = y.values.reshape(-1, 1)

    # scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(y)
y = scaler.transform(y)

n = int(input())
n_lookback = 150
#n_lookback = 100  # length of input sequences (lookback period)
n_forecast = n  # length of output sequences (forecast period)

X = []
Y = []

for i in range(n_lookback, len(y) - n_forecast + 1):
    X.append(y[i - n_lookback: i])
    Y.append(y[i: i + n_forecast])

X = np.array(X)
Y = np.array(Y)


# fit the model
# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
# model.add(LSTM(units=50))
# model.add(Dense(n_forecast))
#
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(X, Y, epochs=100, batch_size=32, verbose=2)

# model.save('model1')
#new_model.fit(X, Y, epochs=200, batch_size=32, verbose=0)
# generate the forecasts
#if not os.path.isdir('LSTM3_model'):
    #fit_model(X, Y, n_forecast)
model = tf.keras.models.load_model(f'my model{n}')
X_ = y[- n_lookback:]  # last available input sequence
X_ = X_.reshape(1, n_lookback, 1)
Y_ = model.predict(X_).reshape(-1, 1)
Y_ = scaler.inverse_transform(Y_)

df_past = df[['4. close']].reset_index()
df_past.rename(columns={'index': 'date', '4. close': 'Actual'}, inplace=True)
df_past['date'] = pd.to_datetime(df_past['date'])
df_past['Forecast'] = np.nan
df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

df_future = pd.DataFrame(columns=['date', 'Actual', 'Forecast'])
df_future['date'] = pd.date_range(start=df_past['date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
df_future['Forecast'] = Y_.flatten()
df_future['Actual'] = np.nan

results = df_past.append(df_future).set_index('date')

# plot the results
plt.plot(results)
plt.show()
#results.plot(title='AAPL')
