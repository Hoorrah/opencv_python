import cv2
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd

WIDTH = 20
Nsimilar = 40
similar = 15
Vsimilar = 10
same = 5
vert_line_bound = 450
hor_line_bound = 250
line_diff = 3

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
    help="type of preprocessing to be done")
args = vars(ap.parse_args())

#_______________________________________________________________________________

    # a kép tetején lévő 20 pixel magas sáv mozgó ablakokkal
def moving_window_vert():
    av_vert = []
    for i in range(0, image.shape[0],WIDTH):
        window = image[i:i+WIDTH,image.shape[1]-WIDTH:image.shape[1]]
        mean_color = window.mean(axis=0).mean(axis=0)
        mean_color = tuple(reversed(mean_color))
        av_vert.append(mean_color)
    return av_vert

#_______________________________________________________________________________

    # a kép jobboldalán lévő 20 pixel széles sáv mozgó ablakokkal
def moving_window_hor():
    av_hor = []
    for i in range(0, image.shape[1],WIDTH):
        window = image[0:WIDTH,i:i+WIDTH]
        mean_color = window.mean(axis=0).mean(axis=0)
        mean_color = tuple(reversed(mean_color))
        av_hor.append(mean_color)    
    return av_hor

#_______________________________________________________________________________


    # a jobboldali sáv ábrázolásához
def create_RGB_for_vert_edge():
    length = len(av_vert)
    img_vert = []
    df_vert = pd.DataFrame({'R':av_vert[:,0], 'G':av_vert[:,1], 'B':av_vert[:,2]})
    R_v = np.reshape(df_vert.R.values/255, (length, 1))
    G_v = np.reshape(df_vert.G.values/255, (length, 1))
    B_v = np.reshape(df_vert.B.values/255, (length, 1))

    for i in range(length):
        img_vert.append([])
        for j in range(1):
            img_vert[i].append((R_v[i][j], G_v[i][j], B_v[i][j]))
    
    return img_vert
    
#_______________________________________________________________________________    
    
    # a felső sáv ábrázolásához  
def create_RGB_for_hor_edge():
    length=len(av_hor)
    img_hor = []
    df_hor = pd.DataFrame({'R':av_hor[:,0], 'G':av_hor[:,1], 'B':av_hor[:,2]})
    R_h = np.reshape(df_hor.R.values/255, (1, length))
    G_h = np.reshape(df_hor.G.values/255, (1, length))
    B_h = np.reshape(df_hor.B.values/255, (1, length))
    
    for i in range(1):
        img_hor.append([])
        for j in range(length):
            img_hor[i].append((R_h[i][j], G_h[i][j], B_h[i][j]))
    return img_hor
    
#_______________________________________________________________________________    
    
    # a sávok ábrázolása
def create_plot_for_edges():
    img_vert = create_RGB_for_vert_edge()
    img_hor = create_RGB_for_hor_edge()
    img_edges = [img_vert, img_hor]
    fig = plt.figure(figsize=(8, 8))
    for i in range(0,2):
        im = img_edges[i]
        fig.add_subplot(1, 2, i+1)
        plt.imshow(im)
    plt.show()




#_______________________________________________________________________________
    
   # az első körös döntéshez szükséges tömbök létrehozása 
def create_right_up_both_arrays():
    right, up, both = [], [], []

    for i in range(1,len(av_vert)):
        for j in range(3):
            right.append(abs(av_vert[i][j]-av_vert[i-1][j]))
    for i in range(1,len(av_hor)):
        for j in range(3):
            up.append(abs(av_hor[i][j]-av_hor[i-1][j]))
    for i in range(len(av_hor)-1,1,-1):
        for j in range(3):
            both.append(abs(av_hor[i][j]-av_vert[i][j]))
    right = np.max(right)
    up = np.max(up)
    both = np.max(both)  
    return right, up, both


#_______________________________________________________________________________

     
     # az első körös döntés
def create_right_up_both_guess():
    right, up, both = create_right_up_both_arrays()
    guess = ''
    corner = ''
    if right <Nsimilar and up >Nsimilar:
        guess = 'jobbra van asztal'
    elif right >Nsimilar and up<Nsimilar:
        guess = 'felül van asztal'
    elif right<Nsimilar and up<Nsimilar:
        guess = 'felül és jobbra van asztal'    
    else:
        guess = 'nincs asztal'    
    return guess,  right, up


#_______________________________________________________________________________


def show_image():
    image = cv2.imread(args["image"])
    cv2.imshow('image', image)
    cv2.waitKey(0)
    return image


#_______________________________________________________________________________


image=show_image()
image_copy=image.copy()  

av_vert=moving_window_vert()
av_hor=moving_window_hor()
av_vert=np.asarray(av_vert)
av_hor=np.asarray(av_hor)

#create_plot_for_edges()
guess,  right, up=create_right_up_both_guess()
print(guess, right, up)
