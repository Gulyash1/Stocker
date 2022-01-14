import streamlit as st
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
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
name_options = ['Microsoft Corp']
name_hint = st.sidebar.text_input(label='Название содержит')
if name_hint is not None:
    name_options = sec_list[sec_list['Name'].str.contains(name_hint, case=False)]['Name'].tolist()
if not name_options:
    name_options = ['Microsoft Corp']

company_name = st.sidebar.selectbox('Компании', name_options)
ticker = sec_list.loc[sec_list['Name'] == company_name, 'Ticker'].iloc[0]
end_date = st.sidebar.date_input('Дата окочания', value=datetime.now()).strftime("%Y-%m-%d")
start_date = st.sidebar.date_input('Дата начала', value=datetime(2010, 5, 31)).strftime("%Y-%m-%d")
#tick = ts.get_daily(symbol=ticker, outputsize='full')[0]

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_ticker_daily(ticker_input):
    ticker_data, ticker_metadata = ts.get_daily(symbol=ticker_input, outputsize='full')
    return ticker_data, ticker_metadata

print(ticker)

try:
    price_data, price_meta_data = get_ticker_daily(ticker)
    market_data, market_meta_data = get_ticker_daily('SPY')
    md_chart_1 = f"Цена **{ticker}** (**{company_name}**) "
    md_chart_2 = f"Измение цена акции за день **{ticker}** (**{company_name}**) "
    md_chart_3 = f"Прогноз цен акций **{ticker}** (**{company_name}**)"
except:
    price_data, price_meta_data = get_ticker_daily('MSFT')
    market_data, market_meta_data = get_ticker_daily('SPY')
    md_chart_1 = f"Invalid ticker **{ticker}** showing **MSFT** price"
    md_chart_2 = f"Invalid ticker **{ticker}** showing **MSFT** APR daily change of"




def apr_change(pandas_series_input):
     return ((pandas_series_input - pandas_series_input.shift(periods=-1,
                                                              fill_value=0)) / pandas_series_input) * 100 * 252


price_data['change'] = apr_change(price_data['4. close'])
market_data['change'] = apr_change(market_data['4. close'])


st.markdown(md_chart_1)
st.line_chart(price_data['4. close'])
st.dataframe(price_data['4. close'])
st.markdown(md_chart_2)
st.line_chart(price_data['change'])
st.dataframe(price_data)



with st.spinner('Подождите. Выполняются вычисления'):
    df = ts.get_daily(symbol=ticker, outputsize='full')[0]
    data = pu.data_preparation(df)

    # разбиение на тестовую и обучающую выборки
    count_step_for_learn = 350
    X_train, X_test, y_train, y_test, count_step_for_learn = pu.scaled(data, 0.7, count_step_for_learn)

        # обучение модели
    if not os.path.isdir('LSTM2_model') or count_step_for_learn != 350:
        pu.fit_model(X_train, y_train)

    # загрузка модели
    model_load = tf.keras.models.load_model('LSTM2_model')

        # предсказание для тестовой выборки
    closing_price = model_load.predict(X_test)

        # инверсия нормализации
    with open("scaler.pickle", "rb") as input_file:
        scaler = pickle.load(input_file)
    closing_price = scaler.inverse_transform(closing_price)

        # график прогнозов
    train_data = data[:-len(closing_price)]
    valid_data = data[-len(closing_price):]
    valid_data = valid_data.assign(Predictions=closing_price)
    pu.prediction_plot(train_data['4. close'], valid_data['4. close'], valid_data['Predictions'])

        # создаем новый фрейм с прошлыми прогнозами
    pred_data = pd.DataFrame(columns=['Forecast'])
    pred_data.loc[len(data)] = data.values[-1][0]
    old_index = len(data)

    # n = 50
    n = st.slider('Количество дней прогноза:', 0, 350, 50)
    print('Прогноз выбранного количества значений: ', n)
    forecast_n = []
    vect_pred = valid_data['4. close'].values

    for i in range(n):
        vect_pred = vect_pred[-350:]
        # подгонка подпространств
        v_pred = []
        for v in vect_pred:
            v_pred.append([v])
        scaled_pred = scaler.fit_transform(v_pred)
        scaled_pred = np.array([scaled_pred])

            # прогноз одного элемента
        one_step_pred = model_load.predict(scaled_pred)

            # инверсия нормализации
        with open("scaler.pickle", "rb") as input_file:
            scaler = pickle.load(input_file)
        one_step_pred = scaler.inverse_transform(one_step_pred)[0][0]

        print('Прогноз ' + str(i + 1) + ': ' + str(one_step_pred))
        forecast_n.append(one_step_pred)
        index = old_index + i + 1
        pred_data.loc[index] = one_step_pred

         # добавляем прогноз в вектор
        vect_pred = list(vect_pred)
        vect_pred.append(one_step_pred)
        vect_pred = np.array(vect_pred)


    plotdata = pd.DataFrame({"Close": data['4. close'],
                             "Forecast": pred_data['Forecast']})

    #st.line_chart(plotdata)

    plot_1 = plotdata
    date_list = list(df['date'])
    end_date = dt.date.today()
    rp = list([end_date + dt.timedelta(days=x) for x in range(n)])
    date_list = date_list + rp
    date_list = pd.DataFrame({'Date': date_list})
    date_list = date_list.sort_values(by='Date')
    date_list = date_list.reset_index(drop=True)
    plot_1['Date'] = date_list['Date']
    plot_1.reset_index(level=0, drop=True)
    plot_1 = plot_1.set_index('Date')




    #st.line_chart(data['4. close'])
    #st.line_chart(ha)
    st.markdown(md_chart_3)
    st.line_chart(plot_1)
    st.dataframe(plot_1)
    time.sleep(5)
st.success('Готово!')
time.sleep(5)





