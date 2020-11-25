# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:52:08 2020

@author: Anna
"""
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt

# This script evaluates the result of cc_analysis.py
# Vectors are saved in results.txt

d = np.loadtxt('results.txt', delimiter="\t")

x = d[0:len(d),0]  
y = d[0:len(d),1]

n = 5    # n has to be changed so it has half the length of x and y
classes = [] 

for i in range(n):
    classes.append(0)

for i in range(n):
    classes.append(1)
    
scale = 50
color = np.array(["black", "green"])
plt.scatter(x,y, c=color[classes], s=scale)

plt.title('Correlation separation in images')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
