import cv2
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd
import where_table

table_guess=where_table.guess
image = where_table.image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
gray = clahe.apply(gray)
WIDTH = 20
vert_line_bound = 450
hor_line_bound = 250

#_______________________________________________________________________________



    # a sávokból készített függvény
def create_function(stripe):
    func = np.zeros((len(stripe), ), float)
    for i in range(len(stripe)-1,1, -1):
        func[i] = abs(stripe[i]-stripe[i-1])
    return func   
        
#_______________________________________________________________________________        

    # az első (fő) lehetséges  függőleges vágóegyenes megkeresése
def find_first_vert_line(stripe):
    fun = create_function(stripe)
    l = len(stripe)
    coord_X = 0
    coord_Y = 0
    for i in range(l-3,round(2*l/3), -1):
        if stripe[i]>vert_line_bound:
            coord_X = i
            coord_Y = stripe[i]
            
            break 
        else:
            coord_X = l-1
            coord_Y = fun[l-1]
    return coord_X, coord_Y
 
#_______________________________________________________________________________ 
 
 
    # a második (másodlagos) lehetséges függőleges vágóegyenes megkeresése
def find_second_vert_line(stripe):
    func = create_function(stripe)
    coord_X, coord_Y = find_first_vert_line(stripe)
    l = len(stripe)
    if coord_X>l-WIDTH:
        coord_X2 = l-1
        coord_Y2 = 0
    else:
        func_next = func[coord_X+int(round(WIDTH/2)):]
        coord_X2 = coord_X+np.argmax(func_next)+int(round(WIDTH/2))
        coord_Y2 = func[coord_X2]
    return coord_X, coord_Y, coord_X2, coord_Y2       

#_______________________________________________________________________________


    # az első (fő) vízszintes vágóegyenes megkeresése
def find_first_hor_line(stripe):
    fun = create_function(stripe)
    l = len(stripe)
    coord_X = 0
    coord_Y = 0
    for i in range(2,round(l/3)):
        if stripe[i]>hor_line_bound:
            coord_X = i
            coord_Y = stripe[i]
            break 
        else:
             coord_Y = stripe[0]
             coord_X = 0    
    #print(coord_X)
    return coord_X, coord_Y        
             
#_______________________________________________________________________________             
             
             
    # a második (másodlagos) lehetséges vágóegyenes megkeresése
def find_second_hor_line(stripe):
    #print(' ')
    #print('find_second_hor_line')
    func = create_function(stripe)
    l = len(stripe)
    coord_X, coord_Y = find_first_hor_line(stripe)
    #print(coord_X)
    if coord_X==0:
        coord_X2 = 0
        coord_Y2 = 0
    elif coord_X>int(round(WIDTH/2)):
        bound = coord_X-int(round(WIDTH/2))
        func_next = func[0:bound]
        coord_X2 = np.argmax(func_next)
        coord_Y2 = func[coord_X2]
    else:
        coord_X2 = 0
        coord_Y2 = 0
    return coord_X, coord_Y, coord_X2, coord_Y2       

#_______________________________________________________________________________


    # a függőleges vágáshoz szükséges vízszintes sáv a kép alján és a hozzá
    # tartozó lehetséges vágási koordináták
def crop_image_vert_bottom():
    crop_img_vert_bottom = gray[image.shape[0]-WIDTH:image.shape[0], 0:].copy()
    arr_vert_bottom = np.asarray(crop_img_vert_bottom)

    gray_level_vert_bottom = np.zeros((len(arr_vert_bottom[1]),), float)
    for i in range(len(arr_vert_bottom[1])):
        gray_level_vert_bottom[i] = math.fsum(arr_vert_bottom[k][i] for k in range(WIDTH-1))
                   
    vert_func_bottom = create_function(gray_level_vert_bottom)
    plot_title_vert_bottom = "vert bottom"
    #plot_function(vert_func_bottom,plot_title_vert_bottom)
    coord_x_bottom, coord_y_bottom, coord_x_bottom2,coord_y_bottom2 = find_second_vert_line(vert_func_bottom)
    coords_bottom = [coord_x_bottom, coord_y_bottom, coord_x_bottom2,coord_y_bottom2]
    return coords_bottom

#_______________________________________________________________________________


    # a függőleges végéhoz szükséges vízszintes sáv a kép közepén és a hozzá
    # tartozó lehetséges vágási koordináták
def crop_image_vert_middle():
    r = round(gray.shape[1]/2)
    crop_img_vert_middle = gray[r-int(round(WIDTH/2)):r+int(round(WIDTH/2)), 0:].copy()
    arr_vert_middle = np.asarray(crop_img_vert_middle)
    gray_level_vert_middle = np.zeros((len(arr_vert_middle[1]),), float)
    for i in range(len(arr_vert_middle[1])):
        gray_level_vert_middle[i] = math.fsum(arr_vert_middle[k][i] for k in range(WIDTH-1))
    vert_func_middle = create_function(gray_level_vert_middle)
    plot_title_vert_middle = "vert middle"
    #plot_function(vert_func_middle, plot_title_vert_middle)
    coord_x_middle, coord_y_middle, coord_x_middle2,coord_y_middle2 = find_second_vert_line(vert_func_middle)
    coords_middle = [coord_x_middle, coord_y_middle, coord_x_middle2,coord_y_middle2]
    return coords_middle
    
 #_______________________________________________________________________________   
    
    
    # alsó és középső vízszintes sávok közötti döntés a vágási koordináták alapján,
    # a végső elsődleges és másodlahos függőleges vágási egyenes koordinátáinak
    # meghatározása
def find_vert_crop_coords():
    coords_middle = crop_image_vert_middle()
    coords_bottom = crop_image_vert_bottom()
    
    if coords_bottom[1]>coords_middle[1]:
        w = coords_bottom[0]
    else:
        w = coords_middle[0]
    if coords_bottom[3]>coords_middle[3]:
        w_next = coords_bottom[2]
    else:
        w_next = coords_middle[2]   
    return w, w_next

#_______________________________________________________________________________

    
    # a vízszintes vágáshoz szükséges függőleges sáv a kép elején (bal szélén) és a hozzá
    # tartozó lehetséges vágási koordináták 
def crop_im_hor_front():
    crop_img_hor_front = gray[0:, 0:WIDTH].copy()
    arr_hor_front = np.asarray(crop_img_hor_front)
    gray_level_hor_front = np.zeros((len(arr_hor_front[:]),), float)
    for i in range(len(arr_hor_front[:])):
        gray_level_hor_front[i] = math.fsum(arr_hor_front[i][k] for k in range(WIDTH-1))

    hor_func_front = create_function(gray_level_hor_front)
    coord_x_front, coord_y_front, coord_x_front2, coord_y_front2 = find_second_hor_line(hor_func_front)
    coords_front = [coord_x_front, coord_y_front, coord_x_front2, coord_y_front2]
    #print(coords_front)
    return coords_front
#_______________________________________________________________________________    
    
    
    # a vízszintes vágáshoz szükséges függőleges sáv a kép közepén és a hozzá
    # tartozó lehetséges vágási koordináták   
def crop_im_hor_middle():   
    r = round(gray.shape[0]/2)
    crop_img_hor_middle = gray[0:, r-int(round(WIDTH/2)):r+int(round(WIDTH/2))].copy()
    arr_hor_middle = np.asarray(crop_img_hor_middle)
    gray_level_hor_middle = np.zeros((len(arr_hor_middle[:]),), float)
    for i in range(len(arr_hor_middle[:])):
        gray_level_hor_middle[i] = math.fsum(arr_hor_middle[i][k] for k in range(WIDTH-1))
    hor_funct_middle = create_function(gray_level_hor_middle)
    coord_x_middle, coord_y_middle,coord_x_middle2, coord_y_middle2 = find_second_hor_line(hor_funct_middle)
    coords_middle = [coord_x_middle, coord_y_middle,coord_x_middle2, coord_y_middle2]
    #print(coords_middle)
    return coords_middle
 
 
#_______________________________________________________________________________ 
 
 
    # balszélső és középső függőleges sávok közötti döntés a vágási koordináták alapján,
    # a végső elsődleges és másodlahos vízszintes vágási egyenes koordinátáinak
    # meghatározása
def find_hor_crop_coords():
    coords_front = crop_im_hor_front()
    coords_middle = crop_im_hor_middle()
    
    if coords_front[1]>coords_middle[1]:
        h = coords_front[0]
    else:
        h = coords_middle[0]
    if coords_front[3]>coords_middle[3]:
        h_next = coords_front[2]
    else:
        h_next = coords_middle[2]   
    return h, h_next   
 
 
#_______________________________________________________________________________ 
 
    # a sávokból készített függvény ábrázolása
def plot_function(func, title):
    fig=plt.figure()
    DPI = fig.get_dpi()
    fig.set_size_inches(1920.0/float(DPI),1080.0/float(DPI))
    plt.plot(func)
    plt.title(title)
    plt.show()


#_______________________________________________________________________________


    # elsődleges vágási egyenesek ábrázolása a képeken az első körös döntés alapján
def draw_the_lines():
    h, h_next, w, w_next=0,0,0,0
    if table_guess=='jobbra van asztal':
        w, w_next=find_vert_crop_coords()
        lineThickness = 2
        line_img=cv2.line(image, (w, 0), (w, image.shape[0]), (0,255,0), lineThickness)
        cv2.imshow('Line', line_img)
        cv2.waitKey(0)
    if table_guess=='felül van asztal':
        h, h_next=find_hor_crop_coords()    
        lineThickness = 2
        line_img=cv2.line(image, (0,h), (image.shape[0],h), (0,255,0), lineThickness)
        cv2.imshow('Line', line_img)
        cv2.waitKey(0)
    if table_guess=='felül és jobbra van asztal' :
        w,w_next=find_vert_crop_coords()
        print(w)
        h,h_next=find_hor_crop_coords()     
        lineThickness = 2
        line_img=cv2.line(image, (w, 0), (w, image.shape[0]), (0,255,0), lineThickness)   
        line_img=cv2.line(image, (w_next, 0), (w_next, image.shape[0]), (255,0,0), lineThickness) 
        line_img=cv2.line(image, (0,h), (image.shape[0],h), (0,255,0), lineThickness)
        line_img=cv2.line(image, (0,h_next), (image.shape[0],h_next), (255,0,0), lineThickness)
        cv2.imshow('Line', line_img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return w, w_next, h, h_next

if table_guess!='nincs asztal':
    w, w_next, h, h_next=draw_the_lines()     



    