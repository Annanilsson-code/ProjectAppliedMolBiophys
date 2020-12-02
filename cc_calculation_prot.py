# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 09:42:00 2020

@author: Anna
"""

import matplotlib.pyplot as plt
import numpy as np 
import mrcfile 

mrc1 = mrcfile.mmap('FoilHole_3164980_Data_3165536_3165537_20190104_1142-60848.mrcs', mode='r+')

n = 1       #n = nr of datasets

# Calculate projections
prot1_mrc = mrcfile.mmap('6yhs.mrc', mode='r+')
u = np.array(prot1_mrc.data)    # shape = (231,200,214)

sum_x = u.sum(axis=0, keepdims=True)    # shape = (1,200,214)
sum_y = u.sum(axis=1, keepdims=True)    # shape = (231,1,214)
sum_z = u.sum(axis=2, keepdims=True)    # shape = (231,200,1)


# Make a histogram of the experimental data. 
exp_data = np.array(mrc1.data)
exp_data_reshaped = exp_data.reshape(n,-1)

    # With numpy
np.histogram(exp_data_reshaped, bins = 'auto')
hist, bins = np.histogram(exp_data_reshaped, bins = 'auto')
print(hist)
print(bins)

    # With matplotlib: Only works in Spyder's console. 
#plt.hist(exp_data_reshaped.T, bins='auto')
#plt.title('Histogram of experimental data')
#plt.show()

# Add noise to the projections.
mean = 0
std = 0.1

noise_x = np.random.normal(mean, std, sum_x.shape)     # (1,200,214)
noise_y = np.random.normal(mean, std, sum_y.shape)     # (231,1,214)
noise_z = np.random.normal(mean, std, sum_z.shape)     # (231,200,1)

noise_x_reshaped = noise_x.reshape(n,-1)    # (1, 42800)
noise_y_reshaped = noise_y.reshape(n,-1) 
noise_z_reshaped = noise_z.reshape(n,-1) 

# Make histogram of the noise. Could do the same for y and z axes.
np.histogram(noise_x_reshaped, bins = 'auto')
hist, bins = np.histogram(noise_x_reshaped, bins = 'auto')
print(hist)
print(bins)

    # With matplotlib: Only works in Spyder's console.
#plt.hist(noise_x_reshaped.T, bins='auto')
#plt.title('Histogram of noise data: projection on x-axis')
#plt.show()

# Calc cc matrix and prepare infile
cc_matrix = np.corrcoef(noise_x_reshaped, noise_y_reshaped[:, 0:42800])

f = open("infile.txt", "w")

for i in range(0, cc_matrix.shape[1]):
    for j in range(0, i+1):
        print(i+1,'\t', j+1, '\t', cc_matrix[i,j], file=f)

