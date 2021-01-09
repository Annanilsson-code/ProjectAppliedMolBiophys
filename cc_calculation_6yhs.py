# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 09:42:00 2020
@author: Anna
"""


# Projections of 6yhs.pdb

import matplotlib.pyplot as plt
import numpy as np 
import mrcfile 
# from PIL import Image, ImageDraw   

# Import electron density map (6yhs)
with mrcfile.open('6yhs_aligned.mrc', 'r+') as mrc: 
    prot1 = np.array(mrc.data)
    del mrc

# Initiate matrix to add noise and protein data
prot_noise = np.zeros((200,200)) 

# Calculate projections
sum_x = prot1.sum(axis=0)
sum_x = sum_x[tuple(slice(0,n) for n in prot_noise.shape)] 
sum_y = prot1.sum(axis=1)
sum_y = sum_y[tuple(slice(0,n) for n in prot_noise.shape)]
sum_z = prot1.sum(axis=2)
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
    gaussian = np.random.normal(mean, std, proj[c].shape)
    noisy_proj[i] = proj[c]*0.0079 + gaussian*11
    classes[i] = c
np.save('classes.npy',classes)


# Import experimental data
with mrcfile.open('FoilHole_3164980_Data_3165536_3165537_20190104_1142-60848.mrcs', 'r+') as mrc:
        exp_data = np.array(mrc.data[1])

exp_data = exp_data[0:105]

# Plot histograms. 
oneprojnoise = noisy_proj[99]
proj_reshaped = oneprojnoise.reshape(-1,1)
exp_data_reshaped = exp_data.reshape((-1,1))


# Plot histograms
plt.figure(num=1)
plt.hist(exp_data_reshaped, bins='auto', alpha=0.5, label="Exp")
plt.hist(proj_reshaped, bins='auto', alpha=0.5, label="Noise", stacked='true')
plt.title('Histograms: experimental noise & simulated noise')
plt.ylabel('#matches in each bin')
plt.legend(loc='upper right')


# Show the noisy images. 
noise = np.random.normal(mean, std, sum_x.shape)
noisy_x = sum_x*0.008 + noise*11

plt.figure(num=2);
plt.imshow(noisy_x);
plt.title('Breakpoint for 6yhs, x-axis')

# Calculate cc_matrix and print to infile
cc_matrix = np.corrcoef(noisy_proj.reshape(n,-1))
file = open("infile.txt", "w")
            
for i in range(0, cc_matrix.shape[1]):
    for j in range(0, i+1):
        file = open("infile.txt", "a")
        file.write("%s % s %.5f" % (i+1, j+1, cc_matrix[i,j]) + '\n')
