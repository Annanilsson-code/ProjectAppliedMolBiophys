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

a = d[0:len(d),0]
b = d[0:len(d),1]

fig, ax = plt.subplots()

for color in ['tab:blue', 'tab:orange']:
    n = len(d)
    scale = 200.0 * np.random.rand(n)
    ax.scatter(a, b, c=color, s=scale, label=color)

ax.legend()
ax.grid(True)

plt.title('Correlation separation in images')
plt.show()
