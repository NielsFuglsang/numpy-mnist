#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 07:48:09 2020

@author: abda
"""
import numpy as np
import matplotlib.pyplot as plt

#%% Read file and labels
def read_mnist():
    f = open(r'mnist/t10k-images-idx3-ubyte', 'r')
    a = np.fromfile(f, dtype='>i4', count=4) # data type is signed integer big-endian
    images = np.fromfile(f, dtype=np.uint8)


    f = open(r'mnist/t10k-labels-idx1-ubyte', 'r')
    t = np.fromfile(f, count = 2, dtype='>i4') # data type is signed integer big-endian
    labels = np.fromfile(f, dtype=np.uint8)

    #%% Show random images
    ims = images.reshape(a[1:])

    return ims, labels, a
