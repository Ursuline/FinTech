#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:34:24 2020

@author: charly
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yahoo_io as io
print(tf.__version__)

def clean_data(df):
    '''Drop index column & add a unix time column as "ndays" '''
    # Remove rows with empty cells (***impute later***)
    df = df.dropna()
    try:
        df = df.drop(['Unnamed: 0'], axis=1)
    except :
        pass

    # Add a unix days ie: since 1 January 1970 column:
    df['ndays'] = (pd.to_datetime(df['Date'], format='%Y-%m-%d') - pd.Timestamp("1970-01-01")) // pd.Timedelta('1d')
    return df


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)

  return dataset

def plot_series(time, series, series_name, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel(series_name)
    plt.grid(True)


def run_linear_regression(X_series, x_train, window_size, batch_size, shuffle_buffer_size, split_time):

    print(f'run_linear_regression parameters:window_size={window_size}\nbatch_size={batch_size}\nshuffle_buffer_size={shuffle_buffer_size}\nsplit_time={split_time}\n')
    dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
    print(f'dataset:\n{dataset}')
    l0 = tf.keras.layers.Dense(1, input_shape=[window_size])
    model = tf.keras.models.Sequential([l0])

    model.compile(loss="mse",
                  optimizer=tf.keras.optimizers.SGD(lr=1e-6,
                                                    momentum=0.9))
    model.fit(dataset, epochs=100,verbose=1)

    print("Layer weights {}".format(l0.get_weights()))

    forecast = []

    for time in range(len(X_series) - window_size):
      forecast.append(model.predict(X_series[time:time + window_size][np.newaxis]))

    forecast = forecast[split_time-window_size:]
    results = np.array(forecast)[:, 0, 0]

    plt.figure(figsize=(10, 6))

    plot_series(t_valid, x_valid, target)
    plot_series(t_valid, results, target)


if __name__ == '__main__':
    file_prefix = 'merged_raw'
    target      = 'Close_BTC_USD'

    split_time  = 1000
    epochs      = 250
    window_size = 20
    batch_size  = 32
    shuffle_buffer_size = 1000

    df = io.load_csv(file_prefix)
    df = clean_data(df)

    # Convert df columns into 2 numpy array for tensorflow
    X_series =  df[target].values
    t_series = df['ndays'].values

    print(f'min-max: {t_series.min()}-{t_series.max()}={t_series.max()-t_series.min()}')

    plot_series(t_series, X_series, target)

    # split the data in training/validation sets
    t_train = t_series[:split_time]
    x_train = X_series[:split_time]
    t_valid = t_series[split_time:]
    x_valid = X_series[split_time:]

    print(f'{t_train}')
    print(f'{x_train}')

    # Linear regression:
    run_linear_regression(X_series, x_train, window_size, batch_size, shuffle_buffer_size, split_time)

    # # Build first model:
    # dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"),
    #     tf.keras.layers.Dense(10, activation="relu"),
    #     tf.keras.layers.Dense(1)
    #     ])

    # model.compile(loss="mse",
    #               optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
    # model.fit(dataset, epochs=100, verbose=1)

    # forecast = []
    # for time in range(len(X_series) - window_size):
    #   forecast.append(model.predict(X_series[time:time + window_size][np.newaxis]))

    # forecast = forecast[split_time-window_size:]
    # results = np.array(forecast)[:, 0, 0]

    # plt.figure(figsize=(10, 6))
    # plot_series(t_valid, x_valid, 'validation set')
    # plot_series(t_valid, results, 'results set')

    # tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()

    # # Build second model:

    # dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"),
    #     tf.keras.layers.Dense(10, activation="relu"),
    #     tf.keras.layers.Dense(1)
    # ])

    # lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    #     lambda epoch: 1e-8 * 10**(epoch / 20))
    # optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
    # model.compile(loss="mse", optimizer=optimizer)
    # history = model.fit(dataset, epochs=100, callbacks=[lr_schedule], verbose=1)

    # lrs = 1e-8 * (10 ** (np.arange(100) / 20))
    # plt.semilogx(lrs, history.history["loss"])
    # plt.axis([1e-8, 1e-3, 0, 300])


    # # Build third model:
    # window_size = 30
    # dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

    # model = tf.keras.models.Sequential([
    #   tf.keras.layers.Dense(10, activation="relu", input_shape=[window_size]),
    #   tf.keras.layers.Dense(10, activation="relu"),
    #   tf.keras.layers.Dense(1)
    # ])

    # optimizer = tf.keras.optimizers.SGD(lr=8e-6, momentum=0.9)
    # model.compile(loss="mse", optimizer=optimizer)
    # history = model.fit(dataset, epochs=500, verbose=0)

    # loss = history.history['loss']
    # epochs = range(len(loss))
    # plt.plot(epochs, loss, 'b', label='Training Loss')
    # plt.show()

    # # Plot all but the first 10
    # loss = history.history['loss']
    # epochs = range(10, len(loss))
    # plot_loss = loss[10:]
    # print(plot_loss)
    # plt.plot(epochs, plot_loss, 'b', label='Training Loss')
    # plt.show()

    # forecast = []
    # for time in range(len(X_series) - window_size):
    #   forecast.append(model.predict(X_series[time:time + window_size][np.newaxis]))

    # forecast = forecast[split_time-window_size:]
    # results = np.array(forecast)[:, 0, 0]


    # plt.figure(figsize=(10, 6))

    # plot_series(t_valid, x_valid)
    # plot_series(t_valid, results)

    # tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()