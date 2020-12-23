#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 11:53:49 2020

@author: charly
"""
import os
import pandas as pd


DIR_NAME = '../data/'

def load_csv(prefix):
    '''Load csv data & return as a dataframe'''
    filename = os.path.join(DIR_NAME, prefix + ".csv")
    return pd.read_csv(filename)


def write_csv(data_frame, prefix):
    '''Write dataframe to csv'''
    filename = os.path.join(DIR_NAME, prefix + ".csv")
    data_frame.to_csv(filename, sep=',', encoding='utf-8')
    print(f'\ndataframe saved as {filename}\n')
