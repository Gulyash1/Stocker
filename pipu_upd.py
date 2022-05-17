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

def get_df(quandl_api_key, times_key,comp):
    quandl.ApiConfig.api_key = quandl_api_key
    ts = TimeSeries(key=times_key, output_format='pandas')
    df = ts.get_daily(symbol=comp, outputsize='full')[0]
    return df


def data_preparation(df):
    df['Year'] = df.index.year
    df.reset_index(level=0, inplace=True)
    data = df.sort_index(ascending=True, axis=0)
    df['date'] = pd.to_datetime(df['date'])
    df.index = df['date']
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['date', '4. close'])
    for i in range(0, len(data)):
        new_data['date'][i] = data['date'][i]
        new_data['4. close'][i] = data['4. close'][i]
    new_data.drop('date', axis=1, inplace=True)
    return new_data


def scaled(data, size_train_sample, count_step_for_learn):
    data = data.values
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)

    train = data[0:int(size_train_sample * len(data)), :]
    test = data[int(size_train_sample * len(data)):, :]

    print('Размер тренировочных данных: ', len(train))
    print('Размер проверочных данных: ', len(test))
    print('Число элементов для прогноза следующего шага: ', count_step_for_learn)


    if count_step_for_learn > len(test):
        count_step_for_learn = int(len(test) - 2)
        print('UPD: Число шагов изменено по причине маленького размера проверочных данных.')
        print('Требуется переобучить модель!')
        print('Число элементов для прогноза следующего шага: ', count_step_for_learn)

    X_train, X_test, y_train, y_test = [], [], [], []
    for i in range(count_step_for_learn, len(train)):
        X_train.append(scaled_data[i - count_step_for_learn:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    for i in range(count_step_for_learn + len(train), len(data)):
        X_test.append(scaled_data[i - count_step_for_learn:i, 0])
        y_test.append(scaled_data[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # сохранение нормализации
    with open("scaler.pickle", "wb") as output_file:
        pickle.dump(scaler, output_file)

    return X_train, X_test, y_train, y_test, count_step_for_learn


def fit_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=2)

    tf.keras.models.save_model(model, 'LSTM1_model')

def prediction_plot(tr_data, v_data, pr_data):
    plt.figure(figsize=(8, 6))
    plt.title('Метод LTSM')
    plt.plot(tr_data, label='Тренировочные данные')
    plt.plot(v_data, label='Проверочные данные')
    plt.plot(pr_data, label='Прогноз')

    plt.legend()
    plt.savefig('plot_lstm.png')
    return plt.show()


def forecast_plot(data, forecast_data):
    plt.figure(figsize=(8, 6))
    plt.title(f'Прогноз на {n} значений методом LTSM')
    plt.plot(data, label='Данные')
    plt.plot(forecast_data, label='Прогноз')
    plt.legend()
    plt.savefig('forecast_lstm.png')
    return plt.show()


if __name__ == '__main__':
    # загрузка данных
    df = get_df("496xBxfk3x-WYLS1vjmz", "0E73MDNQ65Q6HIDP",comp)


    # подготовка данных
    data = data_preparation(df)
    #print(df)
    #print(data)
    # разбиение на тестовую и обучающую выборки
    count_step_for_learn = 800
    X_train, X_test, y_train, y_test, count_step_for_learn = scaled(data, 0.7, count_step_for_learn)
    #print(X_test)
    #print(X_train)

    # обучение модели
    if not os.path.isdir('LSTM1_model') or count_step_for_learn != 800:
        fit_model(X_train, y_train)
    #print(fit_model(X_train,y_train))

    # загрузка модели
    model_load = tf.keras.models.load_model('LSTM1_model')
    #print(model_load)
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
    prediction_plot(train_data['4. close'], valid_data['4. close'], valid_data['Predictions'])

    # создаем новый фрейм с прошлыми прогнозами
    pred_data = pd.DataFrame(columns=['Forecast'])
    pred_data.loc[len(data)] = data.values[-1][0]
    old_index = len(data)

    n = 50
    print('Прогноз выбранного количества значений: ', n)
    forecast_n = []
    vect_pred = valid_data['4. close'].values

    for i in range(n):
        vect_pred = vect_pred[-count_step_for_learn:]
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

    forecast_plot(data['4. close'], pred_data['Forecast'])
