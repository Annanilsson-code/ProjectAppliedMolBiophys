#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:00:06 2020

@author: ylvjan
"""
import numpy as np 
import string


#example matrix to start with to try to sort the elements. 
#values for proofreading output
exmatris = np.array([[1.0, 0, 0, 0, 0],
                     [0.2, 1.0, 0, 0, 0],
                     [0.3, 0.32, 1.0, 0, 0],
                     [0.4, 0.42, 0.43, 1.0, 0],
                     [0.5, 0.52, 0.53, 0.54, 1.0]])
print(exmatris)

# cc_list = np.array()
# #cc_list = []
# for i in range(1, len(exmatris)-1):
#     for j in range(0, len(exmatris)-2):
#         while i > j:  
# #           cc = exmatris.item((i, j))
#             cc = exmatris[i, j]
#             np.append(cc_list, [i, j, cc])
#             j = j + 1            
#     i = i + 1
#     j = 0        
# print(cc_list)

# interate through matrix and save index, axis and value
# cc has to be numpy array? 
cc_list = np.array((tuple, float))
it = np.nditer(exmatris, flags=['multi_index'])
for x in it:
    cc_list = np.append(cc_list, (it.multi_index, '%.2f' % x)) 
#    print("%s %.2f" % (it.multi_index, x))
print(cc_list)
    
# configure from [(0, 0) 'cc'] to [0 0 cc] for correct infile (remove special characters)
#not working yet also split rows 
cc_string = str(cc_list)
# printable = set(string.printable)
# filter(lambda x: x in printable, cc_string)
#for conv in cc_string.iter_lines():
conv = cc_string.replace("(", "")     #not really nice or adaptable code 
conv = conv.replace(")", "")
conv = conv.replace(",", "")
conv = conv.replace("'", "")
conv = conv.replace("<class tuple> <class float>", "")
conv = conv.replace("[", "")
conv = conv.replace("]", "")
conv = conv.replace("\n", "")
print(conv)

#rows = [conv[x:x+3] for x in range(0, len(conv), 3)
rows = str 
#for element in range(0, len(conv), 9):
i = 0
for element in conv:
    element = conv[i:(i+9)]
    rows = str(rows) + str(element + "\n") 
    i = i + 9
print(rows) # end of sting multiple \n

rows = rows.replace("<class 'str'> ", "")

#(create) save to file
with open('cc_file.txt', 'w') as file: #has duplicates (ie both [1 2 cc] and [2 1 cc]) and start indicies at 0
    file.write(rows)
    
    
    
    
    
    