# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 18:52:12 2020

@author: Anna
"""

# This script performs PCA on the projections of 6yhs.pdb

import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import mrcfile 
import pandas as pd

# Import electron density map (6yhs)
with mrcfile.open('6yhs.mrc', 'r+') as mrc: 
    prot1 = np.array(mrc.data)     # (231,200,214)
    del mrc


# Matrix to add noise and protein data
prot_noise = np.zeros((200,200)) 

# Calculate projections
sum_x = prot1.sum(axis=0)    # shape = (1,200,214)
sum_x = sum_x[tuple(slice(0,n) for n in prot_noise.shape)] 
sum_y = prot1.sum(axis=1)    # shape = (231,1,214)
sum_y = sum_y[tuple(slice(0,n) for n in prot_noise.shape)]
sum_z = prot1.sum(axis=2)    # shape = (231,200,1)
sum_z = sum_z[tuple(slice(0,n) for n in prot_noise.shape)]


# Calculate the noise
n=100 
nclasses=3

proj = [sum_x, sum_y, sum_z]
noisy_proj = np.zeros((n, prot_noise.shape[0], prot_noise.shape[1]))

classes = np.zeros((n))
mean = 0
std = 0.1

for i in range(n):
    c = np.random.randint(0, nclasses)
    gaussian = np.random.normal(mean, std, proj[c].shape)*12
    noisy_proj[i] = proj[c]*0.008 + gaussian    
    classes[i] = c
np.save('classes.npy',classes)


# Perform PCA on the data
data = noisy_proj.reshape(n,-1)     # (100,40000)
from sklearn.decomposition import PCA

def pca_components(data, pc_count = None):
    return PCA(n_components = 2).fit_transform(data)

vectors = pca_components(data)

pca1= vectors[:,0]
pca2 = vectors[:,1]

# Plot in scatter plot
fig, ax = plt.subplots()
scatter = ax.scatter(pca1, pca2, c=classes)

legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper left", title="Classes")
ax.add_artist(legend1)
plt.title('Correlation separation in images')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
