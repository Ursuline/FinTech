#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 13:10:57 2020

Loads list of ticker symbols from ist of securities
List of securities downloaded from:
ftp://ftp.nasdaqtrader.com/symboldirectory

@author: charly
"""
import os
import pandas as pd
import scipy.stats
import yfinance as yf

BTC_TICKER = 'BTC-USD'

NDATA_CUTOFF = 500 # skip if less than skip rows
CORR_CUTOFF = .7   # skip if |correlation| < cutoff
PVAL_CUTOFF = .01  # skip if p-value above cutoff

DIRECTORY = './data'
FILENAME  = 'nasdaqtraded.txt' #Nasdaq
OUT_FILE   = 'correlations'

START_DATE = '2015-01-01'
END_DATE   = '2020-10-31'

def write_csv(data_frame, prefix):
    '''Write dataframe to csv'''
    filename = os.path.join(DIRECTORY, prefix + ".csv")
    data_frame.to_csv(filename, sep=',', encoding='utf-8')
    print(f'\ndataframe saved as {filename}\n')


def extract_ticker_symbol(line):
    '''Extract ticker symbol between the first "|" pair from a character
    string of the type:
    Y|AAAU|Perth Mint Physical Gold ETF|P| |Y|100|N||AAAU|AAAU|N
    returns 'AAAU'
    '''
    # remove "Y|" line header
    line = line.replace('Y|', '', 1)
    #truncate rest of line starting with the second '|'
    return line[:line.find('|')]


def extract_ticker_name(line):
    '''Extract ticker symbol between the first "|" pair from a character
    string of the type:
    Y|AAAU|Perth Mint Physical Gold ETF|P| |Y|100|N||AAAU|AAAU|N
    returns 'Perth Mint Physical Gold ETF'
    '''
    line = line.replace('Y|', '', 1)
    line = line[line.find('|')+1:]
    return line[:line.find('|')]


def clean_columns(data_frame):
    '''remove unused columns specified in drop_columns[]
    Use Adj Close coumn as Close if it exists'''
    drop_columns = ['Open', 'High', 'Low', 'Dividends', 'Stock Splits']

    data_frame = data_frame.drop(drop_columns, axis = 1)
    # Keep Adjusted close price adjusted for dividends and splits
    if 'Adj Close' in data_frame.columns:
        del data_frame['Close']
        data_frame.rename(columns = {'Adj Close': 'Close'}, inplace = True)
    return data_frame


def get_ticker_df(symbol, start, end):
    '''Download ticker data from Yahoo Finance'''
    security = yf.Ticker(symbol)
    data_frame =  security.history(start=start, end=end)
    data_frame = data_frame.reset_index() # set Date as a regular column
    if data_frame.shape[0] == 0 :
        return data_frame

    return clean_columns(data_frame)


if __name__ == '__main__':
    file = os.path.join(DIRECTORY, FILENAME)
    # Get bitcoin data
    df_btc = get_ticker_df(BTC_TICKER, START_DATE, END_DATE)

    with open (file, 'rt') as myfile:
        # accumulate results in df_out dataframe
        df_out = pd.DataFrame(columns = ['ticker', 'corr', 'p_value', 'ndata', 'name'], index=None)

        for i, line in enumerate(myfile): # loop over file rows / securities
            if i != 0: # Skip header line
                ticker_name   = extract_ticker_name(line)
                ticker_symbol = extract_ticker_symbol(line)
                print(f'\n{i}:{ticker_name} / {ticker_name}')
                # Get historical data for security
                df = get_ticker_df(ticker_symbol, START_DATE, END_DATE)
                if df.shape[0] > NDATA_CUTOFF: # minimum number of data points met
                    # join the two datasets
                    table = pd.merge(df, df_btc, how='inner', on='Date')

                    try: # compute Pearson correlation & p-value for Close
                        corr, p_value = scipy.stats.pearsonr(table['Close_x'], table['Close_y'])
                    except:
                        #raise Exception(f'Failed to compute correlation with {ticker_symbol}')
                        corr = 0
                    finally:
                        pass

                    # If correlation & p-value criteria met
                    if abs(corr) > CORR_CUTOFF and p_value < PVAL_CUTOFF:
                        df_out = df_out.append(pd.Series([ticker_symbol,
                                                          corr,
                                                          p_value,
                                                          table.shape[0],
                                                          ticker_name],
                                                         index = df_out.columns),
                                               ignore_index = True)
                #if i == 50: break
        write_csv(df_out, OUT_FILE)
