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
 
image1 = io.imread('ellipse.png',pilmode="L")

for n in range(10):
    noise_ellipse = image1 + numpy.random.poisson(lam=image1, size=None) 
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

image2 = io.imread('rectangle.png',pilmode="L")

for n in range(10): 
    # random_nr = numpy.random.poisson(lam=image2, size=None) # lam should be input image
    # noise_rectangle = numpy.random.poisson
    # noise_rectangle = image2 + random_nr*image2.std()*np.random.random(image2.shape)
    # noisy_img_rectangle = Image.fromarray(noise_rectangle)
    noise_rectangle = image2 + numpy.random.poisson(lam=image2, size=None) 
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

# QUESTION: S.O.S!!! How exactly can we keep this in a single np array?
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


# Prepare infile
# set() gets all possible pairs between x1-x4
pairs_to_drop = set()

# get_all_pairs(df) goes through the matrix and returns all
# possible pairs between the objects. 
def get_all_pairs(df):
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

# This function returns all correlations between all the items
# EXCEPT for correlations between the SAME items, x1-x1 etc
# n = nr of images in each dataset
def get_all_correlations(df, n=10):
    au_corr = df.corr().unstack(level=0)
    labels_to_drop = get_all_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]    

corr_pairs = get_all_pairs(df) 
table_cc = get_all_correlations(df, 210) # Should be 210 cc's but we only get 190 because no objects are compared with themselves. 

print("All correlations")
print(table_cc.to_string())


# Write to infile.txt
# QUESTION: the cc values in table_cc are of type float64 and when written to 
# a file, they change (but are close to the true values in table_cc).
# Is there any way to prevent that float64's change when 
# writing them to a file?

# Sort what is written to infile? 
f = open("infile.txt", "w")
print(table_cc.to_string(), file=f) 

# QUESTION: In the infile, we need to fill in with image numbers in col 1.
# How can we fix this?
# Only gives us 190 cc's. Should be 210 if we also compare images with themselves.
# Perhaps we need to add this in the infile. 

    