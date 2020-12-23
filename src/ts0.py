#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:20:59 2020

scientific python 2016

@author: charly
"""

def _pi(b):
    lam = 30.0
    mu = 1.0 / 3.0
    x = 1.0
    s = x
    for i in range(1, b+1):
        x *= lam / (mu*i)
        s += x

    pi = 1.0 / s
    for i in range(1, b+1):
        pi *= lam / (mu*i)
        return pi

for b in range(1, 111):
    print(f'pi({b}) = {_pi(b)}')
