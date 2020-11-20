# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 20:09:50 2020

@author: Anna
"""

# This script creates noised copies of an ellipse and a rectangle.

import numpy
from PIL import Image, ImageDraw    
from scipy import signal
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
from skimage import io
import random

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


# Create an image of an ellipse.

im1 = Image.new('RGB', (500, 300), (128, 128, 128))
draw1 = ImageDraw.Draw(im1)
draw1.ellipse((200, 100, 300, 200), fill=(255, 0, 0), outline=(0, 0, 0))
im1.save('ellipse_notconverted.png', quality=95)
data1 = 'ellipse_notconverted.png'
im1_convert = Image.open('ellipse_notconverted.png', 'r').convert('LA')
im1_convert.save('ellipse.png')

# Create an image of a rectangle.
im2 = Image.new('RGB', (500, 300), (128, 128, 128))
draw2 = ImageDraw.Draw(im2)
draw2.rectangle((200, 100, 300, 200), fill=(255, 0, 0), outline=(0, 0, 0))
im2.save('rectangle_notconverted.png', quality=95)
data2 = 'rectangle_notconverted.png'
im2_convert = Image.open('rectangle_notconverted.png', 'r').convert('LA')
im2_convert.save('rectangle.png')



pix_val_ellipse = []

 # Create n copies of the images, add noise, extract and store pixel values
for n in range(10):
    image1 = io.imread('ellipse.png',pilmode="L")
    random_nr = numpy.random.poisson(lam=1.0, size=None)
    
    # Ask about this line: how to make the images "noisier"?
    noise_ellipse = image1 + random_nr*image1.std()*np.random.random(image1.shape)
    noisy_img_ellipse = Image.fromarray(noise_ellipse)
    
    if noisy_img_ellipse.mode != 'RGB':
        noisy_img_ellipse = noisy_img_ellipse.convert('RGB')
        noisy_img_ellipse.save('noisy_img_ellipse%000d.png' % n)
        # noisy_img_ellipse.show()

    
    # In each iteration, pixel values from the noisy images are added to the list.
    pix_val_ellipse += list(noisy_img_ellipse.getdata())


# Flatten the list to facilitate further data interpretation.
pix_val_ellipse_flat = [x for sets in pix_val_ellipse for x in sets]
    
chunks_ellipse = [pix_val_ellipse_flat[x:x+450000] for x in range(0, len(pix_val_ellipse_flat), 450000)]


pix_val_rectangle = []

# n = nr of images to produce


for n in range(10): 
    image2 = io.imread('rectangle.png',pilmode="L")
    random_nr = numpy.random.poisson(lam=1.0, size=None) # lam should be input image
    # noise_rectangle = numpy.random.poisson
    noise_rectangle = image2 + random_nr*image2.std()*np.random.random(image2.shape)
    noisy_img_rectangle = Image.fromarray(noise_rectangle)
    
    if noisy_img_rectangle.mode != 'RGB':
        noisy_img_rectangle = noisy_img_rectangle.convert('RGB')
        noisy_img_rectangle.save('noisy_img_rectangle%000d.png' % n)
        # noisy_img_rectangle.show()

    pix_val_rectangle += list(noisy_img_rectangle.getdata())


pix_val_rectangle_flat = [x for sets in pix_val_rectangle for x in sets]
    
# create array with dim 10, image2.shape[0]. Better to use numpy arrays

# Split the list into smaller lists. 
# The first list contains the first 450'000 pixel values,
# the second list contains the second 450'000 pixel values.
# Nr has to be changed with the nr of iterations in the for loop.

chunks_rectangle = [pix_val_rectangle_flat[x:x+450000] for x in range(0, len(pix_val_rectangle_flat), 450000)]


# Calculate cc matrix between lists on index 0-n in chunks.
import pandas as pd
import seaborn as sn

# Any more efficient way to write this?
# Keep everything in a single np array
cc_data = {'1': chunks_rectangle[0],
           '2': chunks_rectangle[1],
           '3': chunks_rectangle[2],
           '4': chunks_rectangle[3],
           '5': chunks_rectangle[4],
           '6': chunks_rectangle[5],
           '7': chunks_rectangle[6],
           '8': chunks_rectangle[7],
           '9': chunks_rectangle[8],
           '10': chunks_rectangle[9],
            '11': chunks_ellipse[0],
            '12': chunks_ellipse[1],
            '13': chunks_ellipse[2],
            '14': chunks_ellipse[3],
            '15': chunks_ellipse[4],
            '16': chunks_ellipse[5],
            '17': chunks_ellipse[6],
            '18': chunks_ellipse[7],
            '19': chunks_ellipse[8],
            '20': chunks_ellipse[9]}


# Change the nr depending on nr of images
df = pd.DataFrame(cc_data, columns=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])
cc_matrix = df.corr()
print(cc_matrix)

# Plot the cc_matrix
sn.heatmap(cc_matrix, annot=True)
plt.show()



# Write cc values on infile.txt. 
import numpy as np

# col1_array is numbers in cc_data but have to add more to get 
# everything in cc_matrix
# Has to be changed with the nr of images
# Currently working on this!

col1_array = numpy.arange(start=1, stop=21, step=1)

# col2_array is what col1_array should be compared with
# acc to cc_matrix

col2_array = numpy.arange(start=1, stop=21, step=1)

# col3_array = correlation coefficients
col3_array = numpy.arange(start=1, stop=21, step=1)
#here is your data, in two numpy arrays

infile_data = np.array([col1_array, col2_array, col3_array])
infile_data = infile_data.T
#here you transpose your data, so to have it in two columns

datafile_path = "infile.txt"
with open(datafile_path, 'w+') as datafile_id:
#here you open the ascii file

    np.savetxt(datafile_id, infile_data, fmt=['%d','%d', '%d'])
    #here the ascii file is written. 



# random comment to see if changes are committed
    
