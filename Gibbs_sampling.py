# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 15:42:33 2017

@author: cjiaen
"""

import os
import numpy as np
from functions.io_data import *
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy

#DEFINITIONS
FILEPATH = r"C:\Users\cjiaen\Documents\Sem1\CS5340_UNCERTAINTY\Project\CS5340-Project\a1"
OUTPUTPATH  = r"C:\Users\cjiaen\Documents\Sem1\CS5340_UNCERTAINTY\Project\CS5340-Project\a1\output"
#image_path_list = ['1_noise','2_noise','3_noise','4_noise']
image_path_list = ['2_noise']
EPOCHS = 100
BURNIN = 5

#set random seed
np.random.seed(seed=10)

def gibbs_sampling(epochs, burnin, image):  
    #preprocess image
    image[image == 255] = 1
    H,W = image.shape
    
    #initialize output matrix
    #0 = black, 1 = white
    x = np.zeros((epochs+burnin, H, W), dtype=np.int64)
    x[0,:,:] = copy.deepcopy(image)
    
    #initialize local evidence matrix
    unary_potential = np.array([[0.96, 0.04],
                               [0.04,0.96]])
    
    #initialize pairwise potential matrix
    # row --> neighbour value
    # col --> sample value
    pairwise = np.array([[0.85,0.15],
                         [0.15,0.85]])
    
    for epoch in range(epochs + burnin):       
        for row_idx in range(H):
            for col_idx in range(W):
                L,R,U,D = [1,1],[1,1],[1,1],[1,1]
                #check if left neighbour exists
                if (col_idx-1) != -1:
                    L = pairwise[x[epoch, row_idx, col_idx-1], :]
                #check if right neighbour exists
                if (col_idx+1) < W:
                    R = pairwise[x[epoch, row_idx, col_idx+1], :]
                #check if top neighbour exists
                if (row_idx-1) != -1:
                    U = pairwise[x[epoch, row_idx-1, col_idx], :]
                #check if bottom neighbour exists
                if (row_idx+1) < H:
                    D = pairwise[x[epoch, row_idx+1, col_idx], :]
                pairwise_potential = L*R*U*D
                
                #calculate probability that actual pixel is black
                black_prob = (unary_potential[x[epoch, row_idx, col_idx], 0] * pairwise_potential[0])/\
                        np.sum((unary_potential[x[epoch, row_idx, col_idx], :] * pairwise_potential))
                x[epoch, row_idx, col_idx] = int(np.random.rand() > black_prob)
                    
        #replicate values to next epoch if not last epoch
        if epoch < (epochs + burnin - 1):
            x[epoch+1,:,:] = x[epoch,:,:]
        print("Completed epoch {}".format(epoch))

    #Determine pixel colour based on average over iterations
    temp = np.sum((x[burnin+1:,:,:] == 1), axis = 0)/epochs
    image = np.round(temp)*255
    return image

for image_id in image_path_list:
    image = np.array(Image.open(os.path.join(FILEPATH,image_id + ".png")))
    output = gibbs_sampling(EPOCHS, BURNIN, image)
    output = output.astype(np.uint8)
    new_name = "denoised_" + image_id + "_96_85.png"
    
    img = Image.fromarray(output, "L")
    img.show()
    img.save(os.path.join(OUTPUTPATH, new_name))