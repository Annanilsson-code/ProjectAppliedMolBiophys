# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:52:08 2020

@author: Anna
"""
import numpy as np
import matplotlib.pyplot as plt

# This script evaluates the result of cc_analysis.py
# Vectors are saved in results.txt

d = np.loadtxt('results.txt', delimiter="\t")

x = d[0:len(d),0]
y = d[0:len(d),1]

plt.scatter(x,y, c='cyan')
plt.title('Scatter plot, correlation separation of images in dataset')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
