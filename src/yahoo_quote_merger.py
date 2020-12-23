#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:44:43 2020

Loads quote & volume historical data from csv yahoo finance, joins on date,
and saves merged data to csv

@author: charles mégnin
"""
import pandas as pd
import yahoo_io as io

# yahoo historical data file prefix
prefixes = ['BTC_USD',
            'FCHI_EUR',
            'CL=F_USD',
            'GC=F_USD',
            'DJI_USD',
            'GSPC_USD',
            'IXIC_USD',
            'CNY=X',
            'EUR=X',
            'JPY=X']
# Remove these columns if they exist
unused_columns = ['High', 'Low', 'Open']
# Remove empty volume currency columns
cy_vol_columns = ['Vol_CNY=X', 'Vol_EUR=X', 'Vol_JPY=X']


def delete_columns(data_frame, column_list):
    '''Removes data_frame columns included in the column_list'''
    for column in column_list:
        try:
            del data_frame[column]
        except:
            raise Exception(f'failed to remove {column}')
        finally:
            pass

    return data_frame


def clean_columns(data_frame):
    '''Consistently rename columns'''

    # Remove trailing spaces in column names
    data_frame.columns = data_frame.columns.str.strip(' ')

    try:
        data_frame.rename(columns = {'Close/Last': 'Close'}, inplace = True)
    except:
        pass

    # Keep Adjusted close price adjusted for dividends and splits
    if 'Adj Close' in data_frame.columns:
        del data_frame['Close']
        data_frame.rename(columns = {'Adj Close': 'Close'}, inplace = True)

    try:
        data_frame.rename(columns = {'Volume': 'Vol'}, inplace = True)
    except:
        pass

    return data_frame


def format_date(data_frame):
    '''Give all Date columns the timestamp format'''

    data_frame.insert(loc = 0,
              column = 'Timestamp',
              value = data_frame['Date'].apply(lambda x: pd.Timestamp(x)))
    data_frame = data_frame.sort_values(by='Timestamp')
    # Replace Date column with Timestamp
    del data_frame['Date']
    data_frame.rename(columns = {'Timestamp': 'Date'}, inplace = True)

    return data_frame


def concat_column_names(data_frame, filename):
    '''Add filename to each column name'''
    data_frame = data_frame.rename(columns = lambda x: x + '_' + filename)
    data_frame = data_frame.rename(columns = {"Date_"+filename:"Date"})

    return data_frame


# if __name__ == '__main__':
    pd.set_option('max_columns', 10)

    # Create a {file_number:file_name} dictionary
    files = {}
    [files.setdefault(prefix, i) for i, prefix in enumerate(prefixes)]

    # Initialize list of data_frames -> 1 dataframe per file contents
    data_frame = ['' for i in range(len(files))]

    # Load and clean up data
    for prefix in prefixes:
        id = files[prefix]
        # Load data
        data_frame[id] = io.load_csv(prefix)
        # Remove unnecessary columns
        data_frame[id] = clean_columns(data_frame[id])
        # Remove empty currency volume columns
        data_frame[id] = delete_columns(data_frame[id], unused_columns)
        # Convert date columns to unique timestamp format
        data_frame[id] = format_date(data_frame[id])
        # Add filename to column names
        data_frame[id] = concat_column_names(data_frame[id], prefix)

    # Join all dataframes on Date column
    merged = pd.merge(data_frame[0], data_frame[1], how='inner', on='Date')
    for i in range(2, len(files)):
        merged = pd.merge(merged, data_frame[i], how='inner', on='Date')

    # Remove empty currency volume columns
    merged = delete_columns(merged, cy_vol_columns)

    print(merged.columns)
    print(merged.describe().transpose())

    io.write_csv(merged, 'merged_raw')
