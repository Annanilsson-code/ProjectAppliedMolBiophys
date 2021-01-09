# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 20:09:50 2020

@author: Anna
"""

# This script creates noised copies of squares and rectangles.

import numpy
from PIL import Image, ImageDraw    
import numpy as np
np.random.seed(0) 
import matplotlib.pyplot as plt
from skimage import io


# Create an image of a rectangle.
im2 = Image.new('RGB', (500, 300), (128, 128, 128))
draw2 = ImageDraw.Draw(im2)
draw2.rectangle((190, 100, 300, 200), fill=(255, 0, 0), outline=(0, 0, 0))
im2.save('rectangle_notconverted.png', quality=95)
data2 = 'rectangle_notconverted.png'
im2_convert = Image.open('rectangle_notconverted.png', 'r').convert('LA')
im2_convert.save('rectangle.png')
 
# Create an image of a square.
im3 = Image.new('RGB', (500, 300), (128, 128, 128))
draw3 = ImageDraw.Draw(im3)
draw3.rectangle((200, 100, 300, 200), fill=(255, 0, 0), outline=(0, 0, 0))
im3.save('square_notconverted.png', quality=95)
data3 = 'square_notconverted.png'
im3_convert = Image.open('square_notconverted.png', 'r').convert('LA')
im3_convert.save('square.png')

# Create n noisy images of each geometric figure
n = 100
nclasses = 2

# Store the images in a matrix
images = [io.imread('square.png',pilmode="L"), io.imread('rectangle.png',pilmode="L")]

# Store the noise in a 3D matrix
noisy_images = np.zeros((n,images[0].shape[0], images[0].shape[1]))

# Initiate a class vector
classes = np.zeros((n))

# Calculate noise 
for i in range(n):
  c = np.random.randint(0,nclasses)
  noisy_images[i] = np.random.poisson(images[c]*0.01)
  classes[i] = c

np.save('classes.npy',classes)

# Calculate correlation coefficient matrix
cc_matrix = numpy.corrcoef(noisy_images.reshape(n,-1))

# Write cc’s to a text file
f = open("infile.txt", "w")         

for i in range(0, cc_matrix.shape[1]):
    for j in range(0, i+1):
        file = open("infile.txt", "a")
        file.write("%s % s %.5f" % (i+1, j+1, cc_matrix[i,j]) + '\n')

