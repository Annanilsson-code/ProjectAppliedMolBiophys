#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:28:33 2020

@author: ylvjan
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import pearsonr

# creating image example data set with poisson noise to start testing the code

# Kopiera objekt och namnge var för sig 
# class NyaCirklar:
#     def __init__(self, name):
#         self.name = name
   
# Create a circle
im = Image.new('L', (500, 300))
draw = ImageDraw.Draw(im) 
draw.ellipse((200, 100, 300, 200), 255)
im.save('ellipse.jpg', quality=95)
del draw, im 
cirkel = Image.open('ellipse.jpg')

# Create a square
im = Image.new('L', (500, 300))
draw = ImageDraw.Draw(im)
draw.rectangle((200, 100, 300, 200), 255)
im.save('rectangle.jpg', quality=95)
del draw, im
fyrkant = Image.open('rectangle.jpg')

    
##                          kunna använda för separerade bilder till pc
# cirklar = {}
# for i in range(0,50):
#     name = 'cirkel{}'.format(i)
#     cirklar[name] = cirklar.get(name, NyaCirklar(name = name))
#     cirklar[name] = cirkel.copy()
#     cirklar[name].save(fil, quality=95)


# Generate 50 noisy copies of circle   
##adapt to reading other data from files 
for i in range(50): 
    noiseC = np.random.poisson(cirkel)  #typ bara formen som får brus, högre värden än 255? bg endast 0. Ha mer gråskala i bilden?  
##    noiseim.shape(500,300)
##    ni = Image.fromarray(noiseim)
#    fil = 'ellipse{}.jpg'.format(i)
#    noiseim.save(fil, quality=95)
    
    
# scale down pixel intensities in image before adding noise, larger effect    
    
# Generate 50 noisy image copies of rectangle
#for i in range(50):
for i in range(50): 
    noiseR = np.random.poisson(fyrkant)
#copy when done

   
    
    
    
# calculate correlation coefficients between noisy datasets 
## make adaptable on amount of images, convert all to same size

#pixelväden i kolumner en per bild, rader motsv pixel location  för räkna cc
#pearson correlation, R^2

arr_noiseC = [x for sets in noiseC for x in sets]
arr_noiseR = [x for sets in noiseR for x in sets]

#indata = np.array([arr_noiseC, arr_noiseR]) #nu två rader m 15000 pix


# calculate Pearson's correlation, fler invärden? tar bara 2 argumant
# loop som går igenom alla attribut? 
corr, _ = pearsonr(arr_noiseC, arr_noiseR) #spara i matris

print('Pearsons correlation: %.3f' % corr)










