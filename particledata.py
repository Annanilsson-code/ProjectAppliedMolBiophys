#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:49:52 2020

@author: ylvjan
"""
 
import numpy as np
from skimage import io 
import mrcfile 


# import cryo em data set for cc analysis


with mrcfile.open('FoilHole_3164980_Data_3165536_3165537_20190104_1142-60848.mrcs') as mrc:
   mrc.data[]


n = 10 #len(cryodata)
exim = io.imread('ellipse.jpg', 'L')
dataset = np.zeros((n, exim.shape[0], exim.shape[1]))
for i in range (n):
    new = ('ellipse{}.jpg').format(i)
    dataset[i] = io.imread(new)
    
    
reshaped_dataset = dataset.reshape((n,-1))
cc_matrix = np.corrcoef(reshaped_dataset)


file = open("cc_file.txt", "w")
for i in range(0, cc_matrix.shape[1]):
    for j in range(0, i+1):
        file = open("cc_file.txt", "a")
        file.write("%s % s %.2f" % (i+1, j+1, cc_matrix[i,j]) + '\n')
