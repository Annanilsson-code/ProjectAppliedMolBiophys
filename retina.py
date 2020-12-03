#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:25:51 2020

@author: ylvjan
"""
 
import numpy as np
import mrcfile 
from matplotlib import pyplot as plt

# import electron density map from pdb model  
with mrcfile.open('6G7H.mrc', 'r+') as mrc: 
    arr_6G7H = np.array(mrc.data) 
    del mrc
    
with mrcfile.open('6G7I.mrc', 'r+') as mrc: 
    arr_6G7I = np.array(mrc.data) 
    del mrc
    
with mrcfile.open('6G7J.mrc', 'r+') as mrc: 
    arr_6G7J = np.array(mrc.data) 
    del mrc
    
with mrcfile.open('6G7K.mrc', 'r+') as mrc: 
    arr_6G7K = np.array(mrc.data) 
    del mrc
    
#flatten to projections
flat_6G7H = np.sum(arr_6G7H, 0)
flat_6G7H = flat_6G7H[0:97,0:78]
flat_6G7I = np.sum(arr_6G7I, 0)
flat_6G7I = flat_6G7I[0:97,0:78]
flat_6G7J = np.sum(arr_6G7J, 0)
flat_6G7J = flat_6G7J[0:97,0:78]
flat_6G7K = np.sum(arr_6G7K, 0)
flat_6G7K = flat_6G7K[0:97,0:78]

# plt.figure()
# plt.imshow(flat_6G7H)
# plt.figure()
# plt.imshow(flat_6G7I)
# plt.figure()
# plt.imshow(flat_6G7J)
# plt.figure()
# plt.imshow(flat_6G7K)
# plt.show()

n = 1000
nclasses = 4
mean = 0
std = 0.1
proj = [flat_6G7H, flat_6G7I, flat_6G7J, flat_6G7K]
noisy_proj = np.zeros((n, flat_6G7H.shape[0], flat_6G7I.shape[1]))
classes = np.zeros((n))
for i in range(n):
    c = np.random.randint(0, nclasses)
    gaussian = np.random.normal(mean, std, proj[c].shape)*10
    noisy_proj[i] = (proj[c]*0.1) + gaussian
    classes[i] = c
np.save('classes.npy',classes)

#import experimental data for statistical noise distrubution 
with mrcfile.open('FoilHole_3164980_Data_3165536_3165537_20190104_1142-60848.mrcs', 'r+') as mrc:
    arr_exp = np.array(mrc.data[1])
exp_selct = arr_exp[0:97,0:78]
reshape_exp = exp_selct.reshape((-1,1))

# compare noise distrubution 
oneprojnoise = noisy_proj[1]
reshape_proj = oneprojnoise.reshape(-1,1)
plt.figure()
plt.hist(reshape_exp, bins='auto', alpha=0.5, label="Experimental data")
plt.hist(reshape_proj, bins='auto', alpha=0.5, label="Generated projections", stacked='true')
plt.title('Pixel intensities of projected and experimental data')
plt.legend(loc='upper right')
plt.figure()
plt.imshow(oneprojnoise)
plt.figure()
plt.imshow(exp_selct)
plt.show()

# create correlation coeffisients and infile 
cc_matrix = np.corrcoef(noisy_proj.reshape(n,-1))
file = open("cc_file.txt", "w")
for i in range(0, cc_matrix.shape[1]):
    for j in range(0, i+1):
        file = open("cc_file.txt", "a")
        file.write("%s % s %.5f" % (i+1, j+1, cc_matrix[i,j]) + '\n')