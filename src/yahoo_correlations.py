#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 11:48:27 2020

Generates various correlation-related plots

@author: charly
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yahoo_io as io

pd.set_option('max_columns', 10)

FILE_PREFIX = 'merged_raw'

TARGET = 'Close_BTC_USD'
#labels = ['BTC', 'CAC40', 'Crude', 'Gold', 'DJI', 'S&P', 'NASDAQ', '$/CNY', '$/€', '$/¥']


def corr_plots(dfc):# Build correlation maps
    '''Builds correlation maps by dispatching to reg_plots() kde_plots() & corr_map()'''
    volume_cols = []
    close_cols  = []
    for col in dfc.columns:
        if col.startswith( 'Vol_' ):
            volume_cols.append(col)
        elif col.startswith('Close_'):
            close_cols.append(col)
    volume_cols.append(TARGET)

    # Linear regression plots
    reg_plots(dfc, close_cols)
    reg_plots(dfc, volume_cols)

    # KDE plots
    kde_plots(dfc, [TARGET, 'Close_DJI_USD','Close_GC=F_USD','Close_EUR=X'])
    kde_plots(dfc, [TARGET, 'Vol_BTC_USD','Vol_GSPC_USD','Vol_DJI_USD'])

    # Correlation heatmaps
    # correlation heatmap with quotes
    corr_map(dfc[close_cols], 'Close Heatmap ')
    # correlation heatmap with volume
    corr_map(dfc[volume_cols], 'Volume Heatmap ')
    #correlation among all predictors
    corr_map(dfc, 'Predictor Heatmap ')


def corr_map(dfc, title):
    '''Builds a red->blue correlation map for a given dataframe'''
    size = 18
    linewidths = 1.5
    f,ax = plt.subplots(figsize=(size, size))
    ax.set(title  = title,
           xlabel = "security",
           ylabel = "security")
    sns.heatmap(dfc.corr(),
                annot      = True,
                linewidths = linewidths,
                fmt        = '.2f',
                ax         = ax,
                cmap       = "coolwarm",
                center     = 0.0)
    plt.show()


def reg_plots(dfc, column_list):
    ''' linear regression fit & univariate KDE curves'''
    color = '#d11928'
    for col in column_list:
        sns.jointplot(dfc[TARGET],
                      dfc[col],
                      kind="reg",
                      color=color)
    plt.show()


def kde_plots(dfc, column_list):
    '''Build joint pdfs'''
    sns.set(style="white")
    g = sns.PairGrid(dfc[column_list], diag_sharey=False)
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map_diag(sns.kdeplot, lw=3)
    plt.show()


def time_derivative(data_frame):
    '''computes time derivative delta = [X_0 -  X_(-1)]/Dt for each column
       output dataframe is Date, Delta_date, Delta_columns
    '''
    delta_df = pd.DataFrame()

    data_frame['Date']  = pd.to_datetime(data_frame['Date'])
    # Keep date column
    delta_df['Date'] = pd.to_datetime(data_frame['Date'])

    for col in data_frame.columns:
        if col == 'Date':
            # Convert to # of days
            delta_df['D_Date'] = (data_frame[col] - data_frame[col].shift(1))/data_frame.timedelta(days=1)
        else:
            # Divide by number of days
            delta_df[col] = (data_frame[col] - data_frame[col].shift(1))/delta_df['D_Date']

    # Remove first row (all NaNs)
    delta_df = data_frame.iloc[1:]

    return delta_df


if __name__ == '__main__':
    df = io.load_csv(FILE_PREFIX)

    print(df.describe().transpose())

    # Remove rows with empty cells (***impute later***)
    df = df.dropna()
    try:
        df = df.drop(['Unnamed: 0'], axis=1)
    except :
        pass

    corr_plots(df)

    # Time derivative: [X_i - X_(i-1)] / Delta time
    Ddf = time_derivative(df)
    print(Ddf.describe().transpose())
    corr_plots(Ddf)
