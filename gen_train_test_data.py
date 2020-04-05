import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import keras
import pickle
import os

data_dir="./data/"
ticker="TSLA"

date_pool=pd.date_range("1/1/2019","1/31/2019",freq="B").strftime("%Y%m%d")
date_pool=[d for d in date_pool if os.path.exists(data_dir+"trades_{}_{}.csv".format(d,ticker))]
train_days=10
train_date_list=date_pool[:train_days]
test_date_list=date_pool[train_days+1:]

nforward=10

def load_data(ticker, date):
    df = pd.read_csv(data_dir + 'trades_{}_{}.csv'.format(date, ticker), index_col=[0], parse_dates=[0])

    # Feature Engineering
    df["direction"] = (df["trade_px"] - df["trade_px"].shift(1)).apply(np.sign)
    df["pct_change"] = df["trade_px"].pct_change()

    mysign = lambda x: 0 if abs(x) < 1e-5 else (1 if x > 0 else -1)
    df["label"] = (df["trade_px"].rolling(nforward).mean().shift(-nforward) - df["trade_px"]).apply(mysign)
    # df["label"]=(df["trade_px"].shift(-1)-df["trade_px"]).apply(np.sign) # last version

    df.fillna(method="ffill", inplace=True)
    df.dropna(axis=0, inplace=True)
    # print(df.head(10),df.shape)
    # print("NaN number: ",df.isna().sum().sum())

    return df[["trade_px", "trade_size", "pct_change", "direction", "label"]].values


def create_dataset(ticker, dates, time_steps, input_scaler=None):
    for i, d in enumerate(dates):
        datanew = load_data(ticker, d)
        if i == 0:
            data = datanew
        else:
            data = np.vstack((data, datanew))

    label = data[:, -1]
    data = data[:, :-1]

    if input_scaler is None:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    else:
        data = input_scaler.transform(data)
        scaler = input_scaler

    x = [data[0: time_steps]]
    y = [label[time_steps - 1]]
    N = len(data) // time_steps

    print(N)
    for i in range(1, N):
        t = data[i * time_steps: (i + 1) * time_steps]
        x = np.vstack((x, [t]))
        y.append(label[(i + 1) * time_steps - 1])

    y = pd.get_dummies(y)
    # print(y)

    return x, y.values, scaler


def load_test_data(ticker, date):
    df = pd.read_csv(data_dir + 'trades_{}_{}.csv'.format(date, ticker), index_col=[0], parse_dates=[0])

    # Feature Engineering
    df["direction"] = (df["trade_px"] - df["trade_px"].shift(1)).apply(np.sign)
    df["pct_change"] = df["trade_px"].pct_change()

    df.fillna(method="ffill", inplace=True)
    df.dropna(axis=0, inplace=True)

    return df[["trade_px", "trade_size", "pct_change", "direction"]]

def rolling_window(a, window, axis=0):
    if axis == 0:
        shape = (a.shape[0] - window +1, window, a.shape[-1])
        strides = (a.strides[0],) + a.strides
        a_rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    elif axis==1:
        shape = (a.shape[-1] - window +1,) + (a.shape[0], window)
        strides = (a.strides[-1],) + a.strides
        a_rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return a_rolling


def create_test_dataset(ticker, time_steps, input_scaler=None, date=test_date_list[0]):
    data = load_test_data(ticker, date)
    N = data.shape[0]
    print(N)

    idx = data.index[time_steps-1:]
    data = data.values

    if input_scaler is None:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    else:
        data = input_scaler.transform(data)

    return rolling_window(data, window=time_steps, axis=0), idx