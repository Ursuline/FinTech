#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:03:17 2020

@author: charly
"""
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, stride=1, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
# for i, window in enumerate(dataset):
#   print(i, window.numpy())
#print(f'window -> {type(window)}')
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)
for x, y in dataset:
  print(f'x={x.numpy()}, y={y.numpy()}')