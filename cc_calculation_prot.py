# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 09:42:00 2020

@author: Anna
"""

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import mrcfile 



mrc1 = mrcfile.mmap('FoilHole_3164980_Data_3165536_3165537_20190104_1142-60848.mrcs', mode='r+')

#Put the mrc files in np arrays. 
array1 = np.array(mrc1.data)

# Calculate the mean, std, var of the images

# Simulate noise
