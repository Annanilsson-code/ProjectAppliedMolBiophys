#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:49:52 2020

@author: ylvjan
"""
import numpy as np
import mrcfile 
from sklearn.decomposition import PCA
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
    
# flatten electron density to 2D projections 
# axises are numbered (0, 1, 2)
# choose x to sum along an axis
# maximal projection size from electron density are (119, 97, 78) 
# change 2D size according to which axis where flattened 
x = 1
flat_6G7H = np.sum(arr_6G7H, x)
flat_6G7H = flat_6G7H[0:119,0:78]
flat_6G7I = np.sum(arr_6G7I, x)
flat_6G7I = flat_6G7I[0:119,0:78]
flat_6G7J = np.sum(arr_6G7J, x)
flat_6G7J = flat_6G7J[0:119,0:78]
flat_6G7K = np.sum(arr_6G7K, x)
flat_6G7K = flat_6G7K[0:119,0:78]

# view images of flattened projections befor adding noise 
# plt.figure()
# plt.imshow(flat_6G7H)
# plt.figure()
# plt.imshow(flat_6G7I)
# plt.figure()
# plt.imshow(flat_6G7J)
# plt.figure()
# plt.imshow(flat_6G7K)
# plt.show()


# Create copies of projection and add generated noise
# n is total amoun of numbers 
# nclasses are for labeling noisy projections and correspond to proj
# sf is the scaling factor to decrease intensity of original pixels
n = 1000
nclasses = 4
sf = 0.01
mean = 0
std = 0.1
proj = [flat_6G7H, flat_6G7I, flat_6G7J, flat_6G7K]
noisy_proj = np.zeros((n, flat_6G7H.shape[0], flat_6G7I.shape[1]))
classes = np.zeros((n))
for i in range(n):
    c = np.random.randint(0, nclasses)
    gaussian = np.random.normal(mean, std, proj[c].shape)*10
    noisy_proj[i] = (proj[c]*sf) + gaussian
    classes[i] = c
np.save('classes.npy',classes)

# import experimental data for statistical noise distrubution 
# size of experimental noise is (416,416)
# adapt to projection size to compare same number of pixels
with mrcfile.open('FoilHole_3164980_Data_3165536_3165537_20190104_1142-60848.mrcs', 'r+') as mrc:
    arr_exp = np.array(mrc.data[1]) #0-81 images in mrc file 
exp_selct = arr_exp[0:119,0:78]  #change according to size of projection 
reshape_exp = exp_selct.reshape((-1,1))

# compare pixel intensity distrubution of noisy projection and experimental data 
# oneprojnoise = noisy_proj[1]
# reshape_proj = oneprojnoise.reshape(-1,1)
# plt.figure()
# plt.hist(reshape_exp, bins='auto', alpha=0.5, label="Experimental data")
# plt.hist(reshape_proj, bins='auto', alpha=0.5, label="Generated projections", stacked='true')
# plt.title('Pixel intensities of projected and experimental data')
# plt.legend(loc='upper left')
# plt.figure()
# plt.imshow(oneprojnoise)
# plt.figure()
# plt.imshow(exp_selct)
# plt.show()

# calculate correlation coefficients and sort to infile 
cc_matrix = np.corrcoef(noisy_proj.reshape(n,-1))
file = open("cc_file.txt", "w")
for i in range(0, cc_matrix.shape[1]):
    for j in range(0, i+1):
        file = open("cc_file.txt", "a")
        file.write("%s % s %f" % (i+1, j+1, cc_matrix[i,j]) + '\n')
# run the MDS with calculated cc_file and plot result from that 
        
        
        
        
# For PCA 
# data = noisy_proj.reshape(n,-1)   # (n,40000)
# def pca_components(data, pc_count = None):
#     return PCA(n_components = 3).fit_transform(data)
# vectors = pca_components(data)
# pca1= vectors[:,0]
# pca2 = vectors[:,1]

# Plot PCA in scatter plot
# fig, ax = plt.subplots()
# scatter = ax.scatter(pca1, pca2, c=classes)
# legend1 = ax.legend(*scatter.legend_elements(), loc="best", title="Classes")
# ax.add_artist(legend1)
# plt.title('6G7H, 6G7I, 6G7J, 6G7K, 1000 replicas, *0.05')
# plt.show()
        
     
