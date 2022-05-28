import streamlit as st
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import quandl
import pandas as pd
import pipu_upd as pu
import numpy as np
import tensorflow as tf
import pickle
import os
import datetime as dt

quandl.ApiConfig.api_key = "496xBxfk3x-WYLS1vjmz"
ts = TimeSeries(key='0E73MDNQ65Q6HIDP', output_format='pandas')

sec_list = pd.read_csv('cik_ticker.csv', sep='|',
                       names=['CIK', 'Ticker', 'Name', 'Exchange', 'SIC', 'Business', 'Incorporated', 'IRS'])
name_options = ['International Business Machines Corp']
name_hint = st.sidebar.text_input(label='Название содержит',value='International Business Machines Corp')
if name_hint is not None:
    name_options = sec_list[sec_list['Name'].str.contains(name_hint, case=False)]['Name'].tolist()
if not name_options:
    name_options = ['International Business Machines Corp']

company_name = st.sidebar.selectbox('Компании', name_options)
ticker = sec_list.loc[sec_list['Name'] == company_name, 'Ticker'].iloc[0]
#end_date = st.sidebar.date_input('Дата окочания', value=datetime.now()).strftime("%Y-%m-%d")
#start_date = st.sidebar.date_input('Дата начала', value=datetime(2010, 5, 31)).strftime("%Y-%m-%d")
#tick = ts.get_daily(symbol=ticker, outputsize='full')[0]

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_ticker_daily(ticker_input):
    ticker_data, ticker_metadata = ts.get_daily(symbol=ticker_input, outputsize='full')
    return ticker_data, ticker_metadata
#print(ticker)

try:
    price_data, price_meta_data = get_ticker_daily(ticker)
    #market_data, market_meta_data = get_ticker_daily(ticker)
    md_chart_1 = f"Цена **{ticker}** (**{company_name}**) "
    md_chart_2 = f"Измение цена акции за день **{ticker}** (**{company_name}**) "
    #md_chart_3 = f"Прогноз цен акций **{ticker}** (**{company_name}**) на **{n}** дней вперёд"
except:
    price_data, price_meta_data = get_ticker_daily('IBM')
    #market_data, market_meta_data = get_ticker_daily('IBM')
    md_chart_1 = f"Цена **{ticker}** (**{company_name}**) "
    md_chart_2 = f"Измение цена акции за день **{ticker}** (**{company_name}**) "
    #md_chart_3 = f"Прогноз цен акций **{ticker}** (**{company_name}**) на **{n}** дней вперёд"



def apr_change(pandas_series_input):
     return ((pandas_series_input - pandas_series_input.shift(periods=-1,
                                                              fill_value=0)) / pandas_series_input) * 100 * 252


price_data['change'] = apr_change(price_data['4. close'])
#market_data['change'] = apr_change(market_data['4. close'])


st.markdown(md_chart_1)
st.line_chart(price_data['4. close'])
#st.dataframe(price_data['4. close'])
st.markdown(md_chart_2)
st.line_chart(price_data['change'])
st.caption('1. Open - цена в момент открытия торгов', unsafe_allow_html=False)
st.caption('2. High - максимальная цена в момент торгов', unsafe_allow_html=False)
st.caption('3. Low - минимальная цена в момент торгов', unsafe_allow_html=False)
st.caption('4. Close - цена на момент закрытия торгов', unsafe_allow_html=False)
st.caption('5. Volume - объём продаж', unsafe_allow_html=False)
price_data.drop('change',axis=1, inplace=True)
st.dataframe(price_data)

n = option = st.selectbox(
     'Выбор количества дней для прогноза',
     (int(7), int(14), int(30), int(50)))
with st.spinner('Подождите. Выполняются вычисления'):
  df = price_data
  #df = ts.get_daily(symbol=ticker, outputsize='full')[0]
  df = df.sort_index(ascending=True, axis=0)
  df = df[-365:]
  y = df['4. close']
    #print(y)
  y = y.values.reshape(-1, 1)

        # scale the data
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaler = scaler.fit(y)
  y = scaler.transform(y)

  n_lookback = 150  # length of input sequences (lookback period)
  n_forecast = n  # length of output sequences (forecast period)

  X = []
  Y = []

  for i in range(n_lookback, len(y) - n_forecast + 1):
      X.append(y[i - n_lookback: i])
      Y.append(y[i: i + n_forecast])

  X = np.array(X)
  Y = np.array(Y)

  @st.cache(suppress_st_warning=True, allow_output_mutation=True)
  def model_load():
    return tf.keras.models.load_model(f'my model{n}')
  model = model_load()
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
  md_chart_3 = f"Прогноз цен акций **{ticker}** (**{company_name}**) на **{n}** дней вперёд"

    # plot the results
  st.markdown(md_chart_3)
  st.line_chart(results)
  st.dataframe(results)
st.success('Готово!')






