# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 09:42:00 2020

@author: Anna
"""
# Comparing two different proteins (4o0p.pdb and 4o01.pdb)

import matplotlib.pyplot as plt
import numpy as np 
import mrcfile 

# Import electron density maps (4o0p and 4o01)
with mrcfile.open('4o0p_aligned.mrc', 'r+') as mrc: 
    prot1 = np.array(mrc.data)     # (103,107,97)
    del mrc
    
with mrcfile.open('4o01_aligned.mrc', 'r+') as mrc1: 
    prot2 = np.array(mrc1.data)     # (156,124,125)
    del mrc1

# Matrix to add noise and protein data
prot_noise = np.zeros((97,97))

# Calculate projections
sum_x = prot1.sum(axis=0)    
sum_x = sum_x[tuple(slice(0,n) for n in prot_noise.shape)]   # (103,97)
sum_y = prot1.sum(axis=1)    
sum_y = sum_y[tuple(slice(0,n) for n in prot_noise.shape)]   # (103,97)
sum_z = prot1.sum(axis=2)    
sum_z = sum_z[tuple(slice(0,n) for n in prot_noise.shape)]   # (103,107)

sum_x2 = prot2.sum(axis=0)    
sum_x2 = sum_x2[tuple(slice(0,n) for n in prot_noise.shape)] # (103,107)
sum_y2 = prot2.sum(axis=1)    
sum_y2 = sum_y2[tuple(slice(0,n) for n in prot_noise.shape)] # (103,107)
sum_z2 = prot2.sum(axis=2)    
sum_z2 = sum_z2[tuple(slice(0,n) for n in prot_noise.shape)] # (103,107)

# Calculate the noise
n=100 
nclasses=2


proj = [sum_x, sum_y, sum_z, sum_x2, sum_y2, sum_z2]
noisy_proj = np.zeros((n, prot_noise.shape[0], prot_noise.shape[1]))

classes = np.zeros((n))
mean = 0
std = 0.1

for i in range(n):
    c = np.random.randint(0, nclasses)
    gaussian = np.random.normal(mean, std, proj[c].shape)
    noisy_proj[i] = proj[c]*0.005 + gaussian 
    classes[i] = c
    
np.save('classes.npy',classes)

# Calculate cc_matrix and print to infile
cc_matrix = np.corrcoef(noisy_proj.reshape(n,-1))    #(200,200)


file = open("infile.txt", "w")
            
for i in range(0, cc_matrix.shape[1]):
    for j in range(0, i+1):
        file = open("infile.txt", "a")
        file.write("%s % s %.5f" % (i+1, j+1, cc_matrix[i,j]) + '\n')


plt.figure(num=1);
plt.imshow(sum_x);
plt.title('Prot1: projection on x axis without noise')

plt.figure(num=2);
plt.imshow(sum_x2);
plt.title('Prot2: projection on x axis without noise')

plt.figure(num=3);
plt.imshow(sum_y);
plt.title('Prot1: projection on y axis without noise')

plt.figure(num=4);
plt.imshow(sum_y2);
plt.title('Prot2: projection on y axis without noise')

plt.figure(num=5);
plt.imshow(sum_z);
plt.title('Prot1: projection on z axis without noise')

plt.figure(num=6);
plt.imshow(sum_z2);
plt.title('Prot2: projection on z axis without noise')





### The old code

# import matplotlib.pyplot as plt
# import numpy as np 
# import mrcfile 

# mrc1 = mrcfile.mmap('FoilHole_3164980_Data_3165536_3165537_20190104_1142-60848.mrcs', mode='r+')

# n = 1       #n = nr of datasets

# # Calculate projections
# prot1_mrc = mrcfile.mmap('6yhs.mrc', mode='r+')
# u = np.array(prot1_mrc.data)    # shape = (231,200,214)

# sum_x = u.sum(axis=0, keepdims=True)    # shape = (1,200,214)
# sum_y = u.sum(axis=1, keepdims=True)    # shape = (231,1,214)
# sum_z = u.sum(axis=2, keepdims=True)    # shape = (231,200,1)


# # Create the noise
# mean = 0
# std = 0.1

# noise_x = np.random.normal(mean, std, sum_x.shape)   # * number to match the exp data histogram       # (1,200,214)
# noise_y = np.random.normal(mean, std, sum_y.shape)     # (231,1,214)
# noise_z = np.random.normal(mean, std, sum_z.shape)     # (231,200,1)

# noise_x_reshaped = noise_x.reshape(n,-1)    # (1, 42800)
# noise_y_reshaped = noise_y.reshape(n,-1) 
# noise_z_reshaped = noise_z.reshape(n,-1) 


# # Add the noise to the data
# final_x = noise_x + u
# final_y = noise_y + u


# # Create the histograms and superimpose them (only works in Spyder's console)
# exp_data = np.array(mrc1.data)
# exp_data_reshaped = exp_data.reshape(n,-1)
# final_x_reshaped = final_x.reshape(n,-1)
# final_y_reshaped = final_y.reshape(n,-1)

# plt.figure()
# plt.hist(exp_data_reshaped.T, bins='auto', alpha=0.5, label="data1")
# plt.hist(final_x_reshaped.T, bins='auto', alpha=0.5, label="data2")
# plt.title('Histogram of noise and exp data')
# plt.legend(loc='upper right')


# # Calc cc matrix and prepare infile
# cc_matrix = np.corrcoef(noise_x_reshaped, noise_y_reshaped[:, 0:42800])

# f = open("infile.txt", "w")

# for i in range(0, cc_matrix.shape[1]):
#     for j in range(0, i+1):
#         print(i+1,'\t', j+1, '\t', cc_matrix[i,j], file=f)

