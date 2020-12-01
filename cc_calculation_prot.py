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

    # With matplotlib
plt.hist(exp_data_reshaped, bins='auto')
plt.title('Histogram of exp data')
plt.show()
