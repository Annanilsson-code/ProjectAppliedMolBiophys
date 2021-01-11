#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 11:53:25 2020

@author: ylvjan
"""
import numpy as np
import matplotlib.pyplot as plt

rep_mat = np.load('rep_mat.npy') # resulting vectors from MDS analysis 
classes = np.load('classes.npy') #saved labels for projections corresponding to rep_mat

a = rep_mat[0:len(rep_mat),0]
b = rep_mat[0:len(rep_mat),1]
    
scale = 50
fig, ax = plt.subplots()
scatter = ax.scatter(a, b, scale, classes)
legend1 = ax.legend(*scatter.legend_elements(), loc="best",)
ax.add_artist(legend1)
plt.title('6G7H, 6G7I, 6G7J, 6G7K, 1000 replicas', *0.01)
plt.show()
