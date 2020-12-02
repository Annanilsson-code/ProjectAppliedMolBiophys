#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:49:52 2020

@author: ylvjan
"""
 
import numpy as np
import mrcfile 
from matplotlib import pyplot as plt

# import electron density map from pdb model  
with mrcfile.open('6yhs3.mrc', 'r+') as mrc: 
    arr_6yhs = np.array(mrc.data) 

# simulate projections by summing along an axis
size = np.zeros((200,200))
z_view = np.sum(arr_6yhs, 0)
z_view = z_view[tuple(slice(0,n) for n in size.shape)]
y_view = np.sum(arr_6yhs, 1)
y_view = y_view[tuple(slice(0,n) for n in size.shape)]
x_view = np.sum(arr_6yhs, 2)
x_view = x_view[tuple(slice(0,n) for n in size.shape)]

n = 1000
nclasses = 3
proj = [x_view, y_view, z_view]
noisy_proj = np.zeros((n, size.shape[0], size.shape[1]))
classes = np.zeros((n))
for i in range(n):
    c = np.random.randint(0, nclasses)
    noisy_proj[i] = np.random.normal(proj[c]*0.01)
    classes[i] = c
np.save('classes.npy',classes)

#import experimental data for statistical noise distrubution 
# with mrcfile.open('FoilHole_3164980_Data_3165536_3165537_20190104_1142-60848.mrcs', 'r+') as mrc:
#     arr_exp = np.array(mrc.data[1])
# reshape_exp = arr_exp.reshape((-1,1))

# compare noise distrubution 
# oneprojnoise = noisy_proj[60]
# reshape_proj = oneprojnoise.reshape(-1,1)
# plt.hist([reshape_exp, reshape_proj], bins='auto', stacked='true')
# plt.figure()
# plt.imshow(oneprojnoise)
# plt.figure()
# plt.imshow(arr_exp)
# plt.figure()
# plt.imshow(x_view)
# plt.show()

# create correlation coeffisients and infile 
cc_matrix = np.corrcoef(noisy_proj.reshape(n,-1))
file = open("cc_file.txt", "w")
for i in range(0, cc_matrix.shape[1]):
    for j in range(0, i+1):
        file = open("cc_file.txt", "a")
        file.write("%s % s %.5f" % (i+1, j+1, cc_matrix[i,j]) + '\n')
     
