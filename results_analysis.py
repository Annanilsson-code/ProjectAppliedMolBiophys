# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:52:08 2020

@author: Anna
"""
import numpy as np
import matplotlib.pyplot as plt

d = np.loadtxt('results.txt', delimiter="\t")   # d = (n,2)

a = d[0:len(d),0]
b = d[0:len(d),1]

from cc_calculation import classes
    
scale = 50

fig, ax = plt.subplots()
scatter = ax.scatter(a, b, c=classes, s=scale)

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper left", title="Classes")
ax.add_artist(legend1)
plt.title('Correlation separation in images')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
