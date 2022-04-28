import cv2
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
import os
from os.path import join
import argparse
import pandas as pd
import where_table
import find_the_lines
from PIL.ImageOps import crop

WIDTH = 20
Nsimilar = 40
similar = 15
Vsimilar = 10
same = 5
vert_line_bound = 450
hor_line_bound = 250
line_diff = 3

image=where_table.image_copy
table_guess=where_table.guess
right=where_table.right
up=where_table.up

if table_guess=='felül van asztal':
    h=find_the_lines.h-line_diff
    h_next=find_the_lines.h_next-line_diff
    w=0
    w_next=0
if table_guess=='jobbra van asztal':
    w=find_the_lines.w+line_diff
    w_next=find_the_lines.w_next+line_diff
    h=0
    h_next=0
if table_guess=='felül és jobbra van asztal':
    h=find_the_lines.h-line_diff
    h_next=find_the_lines.h_next-line_diff
    w=find_the_lines.w+line_diff
    w_next=find_the_lines.w_next+line_diff
print(table_guess) 
if table_guess !='nincs asztal':
    print(h,h_next,w,w_next)


#_______________________________________________________________________________


    # mozgó ablakok által meghatározott függőleges sáv a lehetséges függőleges 
    # vágái egyenes két oldalán,
    # a kép tetejétől a lehetséges vízszintes vágási egyenesig tart
def moving_window_before_hor_line():
    #print('__________________________________________')
    #print('moving_window_before_hor_line')
    av_vert_line_l = []
    av_vert_line_r = []
    if h != 0:
        if h>=int(round(WIDTH/2)):
            step = int(round(WIDTH/2))
        else: 
            step = int(round(h/2))
        for i in range(0, h, step):
            window_left = image[i:i+int(round(WIDTH/2)),w-int(round(WIDTH/2)):w]
            window_right = image[i:i+int(round((WIDTH/2))),w:w+int(round((WIDTH/2)))]
            mean_col_left = tuple(reversed(window_left.mean(axis=0).mean(axis=0)))
            mean_col_right = tuple(reversed(window_right.mean(axis=0).mean(axis=0)))
            av_vert_line_l.append(mean_col_left)
            av_vert_line_r.append(mean_col_right)
        av_vert_line_l = np.asarray(av_vert_line_l)
        av_vert_line_r = np.asarray(av_vert_line_r)
    else:
        av_vert_line_l = []
        av_vert_line_r = []
    return av_vert_line_l, av_vert_line_r


#_______________________________________________________________________________

    # a színek komponensenként vett különbségének maximuma a lehetséges függőleges
    # vágási egyenes két oldalán
def check_diffs_before_hor_line():
    #print('__________________________________________')
    #print('check_diffs_before_hor_line')
    av_vert_line_l, av_vert_line_r = moving_window_before_hor_line()
    if len(av_vert_line_l) != 0:
        diffs = []
        for i in range(0,len(av_vert_line_l)):
            for j in range(3):
                diffs.append(abs(av_vert_line_l[i][j] - av_vert_line_r[i][j]))
        max_diff = np.max(diffs)
    else:
        max_diff = 0
    #print("max diff along vert line: ", max_diff)

    return max_diff

#_______________________________________________________________________________

    # mozgó ablakok által meghatározott vízszintes sáv a lehetséges vízszintes 
    # vágái egyenes két oldalán,
    # a vízszintes vágási egyenestől a kép végéig (jobbszéléig) tart
def moving_window_after_vert_line():
    #print('__________________________________________')
    #print('moving_window_after_vert_line')
    av_hor_line_l = []
    av_hor_line_r = []
    if h!=0:
        if h<int(round(WIDTH/2)):
            step = h
        else:
            step=int(round(WIDTH/2))
        for i in range(w, w+2*step,step):
            window_left = image[h-step:h,i:i+step]
            window_right = image[h:h+step,i-step:i]
            mean_col_left = tuple(reversed(window_left.mean(axis=0).mean(axis=0)))
            mean_col_right = tuple(reversed(window_right.mean(axis=0).mean(axis=0)))
            av_hor_line_l.append(mean_col_left)
            av_hor_line_r.append(mean_col_right)
        av_hor_line_l = np.asarray(av_hor_line_l)
        av_hor_line_r = np.asarray(av_hor_line_r)
        
    return av_hor_line_l, av_hor_line_r

#_______________________________________________________________________________

    # a színek komponensenként vett különbségeinek maximuma a lehetséges vízszintes vágási 
    # egyenes két oldalán
def check_diffs_after_vert_line():
    #print('__________________________________________')
    #print('check_diffs_after_vert_line')
    av_hor_line_l, av_hor_line_r=moving_window_after_vert_line()
    if len(av_hor_line_l) != 0:
        diffs = []
        for i in range(1,len(av_hor_line_l)):
            for j in range(3):
                diffs.append(abs(av_hor_line_l[i][j] - av_hor_line_r[i][j]))
        max_diff = np.max(diffs)
    else:
        max_diff = 0
    
    return max_diff


#_______________________________________________________________________________



    # a függőleges és vízsintes vágási egyenesek metszéspontja körüli három 
    # négyzetből álló "sarok" létrehozása
def create_av_colors_for_cross_check():
    #print('__________________________________________')
    #print('create_av_colors_for_cross_check')
    if h < 2*WIDTH and image.shape[1]-w >= 2*WIDTH:
        step = h
        av1 = image[h-step:h, w-step:w]
        av2 = image[h-step:h, w+step:w+2*step]
        av3 = image[h:h+step,w+step:w+2*step]
    elif image.shape[1]-w < 2*WIDTH and h >= 2*WIDTH:
        step = image.shape[1]-w
        av1 = image[h-2*step:h-step, w-step:w]
        av2 = image[h-2* step:h-step, w:w+step]
        av3 = image[h:h+step,w:w+step]
    elif h<2*WIDTH and image.shape[1]-w < 2*WIDTH:
        step = min(h,image.shape[1]-w)
        av1 = image[h-step:h, w-step:w]
        av2 = image[h-step:h, w:w+step]
        av3 = image[h:h+step,w:w+step]
    elif h >= 2*WIDTH and image.shape[1]-w >= 2*WIDTH:
        step = WIDTH
        av1 = image[h-2*step:h-step, w-step:w]
        av2 = image[h-2* step:h-step, w+step:w+2*step]
        av3 = image[h:h+step,w+step:w+2*step] 
    return av1,av2,av3


#_______________________________________________________________________________


    # a függőleges és vízsintes vágási egyenesek metszéspontja körüli három 
    # négyzetből álló "sarok" áltagszíneinek ábrázolása
def create_plot_for_cross_check(mean_av1, mean_av2,mean_av3):
    #print('__________________________________________')
    #print('create_plot_for_cross_check')
    av1_plot = [[(mean_av1[0]/255, mean_av1[1]/255, mean_av1[2]/255)]]
    av2_plot = [[(mean_av2[0]/255, mean_av2[1]/255, mean_av2[2]/255)]]
    av3_plot = [[(mean_av3[0]/255, mean_av3[1]/255, mean_av3[2]/255)]]
    ims=[av1_plot, av2_plot, av3_plot]

    fig = plt.figure(figsize=(8, 8))
    for i in range(0,3):
        im = ims[i]
        fig.add_subplot(3, 1, i+1)
        plt.imshow(im)
    plt.show()


#_______________________________________________________________________________


    # a függőleges és vízsintes vágási egyenesek metszéspontja körüli három 
    # négyzetből álló "sarok" áltagszíneinek számítása
def create_mean_for_line_cross():
    #print('__________________________________________')
    #print('create_mean_for_line_cross')
    av1,av2,av3 = create_av_colors_for_cross_check()
    mean_av1 = np.asarray(tuple(reversed(av1.mean(axis=0).mean(axis=0))))
    mean_av2 = np.asarray(tuple(reversed(av2.mean(axis=0).mean(axis=0))))
    mean_av3 = np.asarray(tuple(reversed(av3.mean(axis=0).mean(axis=0))))
    means = [mean_av1, mean_av2, mean_av3]
    return means

#_______________________________________________________________________________

    # a függőleges és vízsintes vágási egyenesek metszéspontja körüli három 
    # négyzetből álló "sarok" áltagszíneinek különbségének maximuma
def lines_cross_diff():
    #print('__________________________________________')
    #print('check_line_cross')
    av1,av2,av3 = create_av_colors_for_cross_check()
    means = create_mean_for_line_cross()
    #create_plot_for_cross_check(means[0], means[1], means[2])

    diffs = []
    diff1 = abs(means[0]-means[1])
    diff2 = abs(means[0]-means[2])
    diff3 = abs(means[1]-means[2])
    for i in range(3):
        diffs.append(diff1[i])
        diffs.append(diff2[i])
        diffs.append(diff3[i])
    max_diff = np.max(diffs) 
    return max_diff

#_______________________________________________________________________________

    # a függőleges vágái egyenestől a kép jobbszéléig tartó vízszintes sávok
    # létrehozása mozgó ablakokkal, egy a kép aljához közel, egy a vízszintes 
    # vágási egyeneshez közel, a mozgó ablakok átlagszíneinek számítása
def create_rows_for_vert_line(H,W):
    #print('__________________________________________')
    #print('create_rows_for_vert_line')
    av_hor_start = []
    av_hor_end = []

    step = int(round((image.shape[1]-W)/4))
    if step != 0:
        for i in range(W-2*step,image.shape[1],step):
            window_hor_start = image[image.shape[0]-2*step:image.shape[0]-step,i:i+step]
            window_mean_start = tuple(reversed(window_hor_start.mean(axis=0).mean(axis=0)))
            av_hor_start.append(window_mean_start)
        av_hor_start = np.asarray(av_hor_start)
        
        for i in range(W-2*step,image.shape[1],step):
            window_hor_end = image[H+2*step:H+4*step,i:i+step]
            window_mean_end = tuple(reversed(window_hor_end.mean(axis=0).mean(axis=0)))
            av_hor_end.append(window_mean_end)
        av_hor_end = np.asarray(av_hor_end)
        #create_plot_for_nocrop(av_hor_start)
    return av_hor_start,av_hor_end


#_______________________________________________________________________________

 
    # az előbb létrehozott vízszintes sávok mozgó ablakai áltagszínének különbségének
    # átlaga
def diffs_of_rows_for_vert_lines(H,W):
    #print('__________________________________________')
    #print('diffs_of_rows_for_vert_lines')
    av_hor_start, av_hor_end = create_rows_for_vert_line(H,W)
    av_hor_start_diffs = []
    for i in range(3,len(av_hor_start)):
        for j in range(3):
            av_hor_start_diffs.append(abs(av_hor_start[i][j]-av_hor_start[i-1][j]))
    av_hor_start_mean = np.mean(av_hor_start_diffs)

    av_hor_end_diffs = []
    for i in range(3,len(av_hor_end)):
        for j in range(3):
            av_hor_end_diffs.append(abs(av_hor_end[i][j]-av_hor_end[i-1][j]))
    av_hor_end_mean = np.mean(av_hor_end_diffs)
    
   
    #print('av_hor_start_mean, av_hor_end_mean: ', av_hor_start_mean, av_hor_end_mean)  
    return av_hor_start_diffs, av_hor_end_diffs,av_hor_start_mean, av_hor_end_mean


#_______________________________________________________________________________


    # az előbbi mozgó ablakokhoz hozzáveszünk még kettőt, ami már a függőleges 
    # vágási egyenesen belül van, és kiszámítjuk ennek és az előbbi sávnak a 
    # különbségeinek maximumát
def diffs_for_nocrop_vert(H,W):
    #print('__________________________________________')
    #print('diffs_for_nocrop_vert')
    av_hor_start_diffs, av_hor_end_diffs,av_hor_start_mean, av_hor_end_mean=diffs_of_rows_for_vert_lines(H,W)
    diff_nocrop_start=av_hor_start_diffs
    av_hor_start,av_hor_end=create_rows_for_vert_line(H,W)
    if len(av_hor_start)>1:
        for i in range(3):
            diff_nocrop_start.append(abs(av_hor_start[0][i]-av_hor_start[1][i]))
            diff_nocrop_start.append(abs(av_hor_start[1][i]-av_hor_start[2][i]))
            max_nocrop_start=np.max(diff_nocrop_start)
    else:
        max_nocrop_start=0
    
    diff_nocrop_end=av_hor_end_diffs
    if len(av_hor_end)>1:
        for i in range(3):
            diff_nocrop_end.append(abs(av_hor_end[0][i]-av_hor_end[1][i]))
            diff_nocrop_end.append(abs(av_hor_end[1][i]-av_hor_end[2][i]))
            max_nocrop_end=np.max(diff_nocrop_end)
    else:
        max_nocrop_end=0


    #print('vert nocrop: ', max_nocrop_start, max_nocrop_end)

    return  max_nocrop_start, max_nocrop_end
    
#_______________________________________________________________________________    
    
    # a függőleges vágási egyenes két oldalán lévő vízszintes, mozgó ablakok
    # által meghatározott sáv
def moving_window_for_vert_line():
    #print('__________________________________________')
    #print('moving_window_for_vert_line')
    av_vert_in=[]
    av_vert_out=[]
    if image.shape[1]-w>=20:
        step=20
    else:
        step=image.shape[1]-w
    if step !=0:
        for i in range(0, image.shape[0],step):
            window_inside=image[i:i+step,w-step:w]
            window_outside=image[i:i+step,w:w+step]
            mean_col_inside=tuple(reversed(window_inside.mean(axis=0).mean(axis=0)))
            mean_col_outside=tuple(reversed(window_outside.mean(axis=0).mean(axis=0)))
            av_vert_in.append(mean_col_inside)
            av_vert_out.append(mean_col_outside)
    av_vert_in=np.asarray(av_vert_in)
    av_vert_out=np.asarray(av_vert_out)      
    return av_vert_in, av_vert_out  


#_______________________________________________________________________________

    
    # a függőleges vágási egyenes két oldalán lévő vízszintes, mozgó ablakok
    # által meghatározott sáv mozgó ablakai átlagszínének különbségei a két oldalon,
    # tehát mindig a belső (baloldali) négyzetet a külső (jobboldali) négyzethez
    # hasonlítjuk, meghatározzuk a különbségek maximumát
def diff_of_moving_window_for_vert_line():
    #print('__________________________________________')
    #print('diff_of_moving_window_for_vert_line')
    av_vert_in, av_vert_out=moving_window_for_vert_line()
    vert_diff=[]
    for i in range(1,int(round(len(av_vert_in)/3))):
        for j in range(3):
            vert_diff.append(abs(av_vert_in[i][j]-av_vert_out[i][j]))
    for i in range(int(2*round(len(av_vert_in)/3)),len(av_vert_in)):
        for j in range(3):
            vert_diff.append(abs(av_vert_in[i][j]-av_vert_out[i][j]))
    max_diff_vert=np.max(vert_diff)
    #print('max_diff_vert: ', max_diff_vert)
    return max_diff_vert



#_______________________________________________________________________________

    
    # a függőleges vágási egyenes baloldalán (belül) lévő mozgó ablakok által
    # meghatározott sáv mozgó ablakainek különbségei, a különbség maximuma
def diff_of_moving_window_for_vert_line_inside():
    #print('__________________________________________')
    #print('diff_of_moving_window_for_vert_line_inside')
    av_vert_in, av_vert_out=moving_window_for_vert_line()
    vert_in_diff=[]
    for i in range(1,int(round(len(av_vert_in)/3))):
        for j in range(3):
            vert_in_diff.append(abs(av_vert_in[i][j]-av_vert_in[i-1][j]))
    for i in range(int(2*round(len(av_vert_in)/3)),len(av_vert_in)):
        for j in range(3):
            vert_in_diff.append(abs(av_vert_in[i][j]-av_vert_in[i-1][j]))
    max_diff_vert_in=np.max(vert_in_diff)
    return max_diff_vert_in

#_______________________________________________________________________________

    # a függőleges vágási egyenes jobboladlán (kívül) lévő mozgó ablakok által
    # meghatározott sáv mozgó ablakainek különbségei, a különbségek maximuma
def diff_of_moving_window_for_vert_line_outside():   
    #print('__________________________________________')
    #print('diff_of_moving_window_for_vert_line_outside')
    av_vert_in, av_vert_out=moving_window_for_vert_line() 
    vert_out_diff=[]
    for i in range(1,int(round(len(av_vert_in)/3))):
        for j in range(3):
            vert_out_diff.append(av_vert_out[i][j]-av_vert_out[i-1][j])
    for i in range(int(2*round(len(av_vert_in)/3)),len(av_vert_in)):
        for j in range(3):
            vert_out_diff.append(av_vert_out[i][0]-av_vert_out[i-1][0])
    max_diff_vert_out=np.max(vert_out_diff)  
    vert_out_diff.remove(max_diff_vert_out)
    max_diff_vert_out=np.max(vert_out_diff)  
    return max_diff_vert_out


#_______________________________________________________________________________ 
 
    # a kép tetejétől  a vízszintes vágái egyenesig tartó függőleges sávok
    # létrehozása mozgó ablakokkal, egy a kép baloldalához közel, egy a függőleges 
    # vágási egyeneshez közel, a mozgó ablakok átlagszíneinek számítása
def create_columns_for_hor_line(H,W):
    #print('__________________________________________')
    #print('create_columns_for_hor_line')

    av_vert_start=[]
    av_vert_end=[]

    step=int(round(H/4))  
    if step!=0:
            
        #print('step, h: ', step, H)

        for i in range(0,H+2*step,step):
            window_vert_start=image[i:i+step, 2*step:4*step]
            window_mean_start=tuple(reversed(window_vert_start.mean(axis=0).mean(axis=0)))
            av_vert_start.append(window_mean_start)
        av_vert_start=np.asarray(av_vert_start)
        
        for i in range(0,H+2*step,step):
            window_vert_end=image[i:i+step, W-4*step:W-2*step]
            window_mean_end=tuple(reversed(window_vert_end.mean(axis=0).mean(axis=0)))
            av_vert_end.append(window_mean_end)
        av_vert_end=np.asarray(av_vert_end)
        #create_plot_for_nocrop(av_vert_start)

    return av_vert_start, av_vert_end

#_______________________________________________________________________________


    # az előbb létrehozott függőleges sávok átlagszíneinek különbsége, a különbség
    # maximuma
def diffs_of_columns_for_hor_line(H,W):
    #print('__________________________________________')
    #print('diffs_of_columns_for_hor_line')
    av_vert_start, av_vert_end=create_columns_for_hor_line(H,W)
    av_vert_start_diffs=[]
    av_vert_end_diffs=[]
    if len(av_vert_start) >4:
        for i in range(1,len(av_vert_start)-3):
            for j in range(3):
                av_vert_start_diffs.append(abs(av_vert_start[i][j]-av_vert_start[i-1][j]))
        av_vert_start_max=np.max(av_vert_start_diffs)
    else:
        av_vert_start_max=0
       
    if len(av_vert_end) >4:
        for i in range(1,len(av_vert_end)-3):
            for j in range(3):
                av_vert_end_diffs.append(abs(av_vert_end[i][j]-av_vert_end[i-1][j]))
        av_vert_end_max=np.max(av_vert_end_diffs)
    else:
        av_vert_end_max=0
    
    #print('av_vert_start_max, av_vert_end_max: ', av_vert_start_max, av_vert_end_max)
    return av_vert_start_diffs, av_vert_end_diffs, av_vert_start_max, av_vert_end_max

#_______________________________________________________________________________

    # azon sáv mozgó ablakai átlagszínéneinek ábrázolása, amely a vágási egyenesen
    # kívül van, de a vágási egyenesen belül is hozzáveszünk két négyzetet
def create_plot_for_nocrop(diff_nocrop):
    length=len(diff_nocrop)
    img_hor = []
    #print(diff_nocrop)
    df_hor = pd.DataFrame({'R':diff_nocrop[:,0], 'G':diff_nocrop[:,1], 'B':diff_nocrop[:,2]})
    R_h = np.reshape(df_hor.R.values/255, (1, length))
    G_h = np.reshape(df_hor.G.values/255, (1, length))
    B_h = np.reshape(df_hor.B.values/255, (1, length))
    
    for i in range(1):
        img_hor.append([])
        for j in range(length):
            img_hor[i].append((R_h[i][j], G_h[i][j], B_h[i][j]))
    
    fig=plt.figure(figsize=(8, 8))

    plt.imshow(img_hor)
    plt.show()
    
    
#_______________________________________________________________________________
    
 
    # az előbbi mozgó ablakok által meghatározott sávhoz hozzáveszünk még két 
    # négyzetet, azt már a vágási egyenesen belül és kiszámítjuk ennek és az 
    # egész előbbi sávnak a különségét, a különségek maximumát
def diffs_for_nocrop_hor(H,W):
    #print('__________________________________________')
    #print('diffs_for_nocrop_hor')
    av_vert_start_diffs, av_vert_end_diffs, av_vert_start_max, av_vert_end_max=diffs_of_columns_for_hor_line(H,W)
    av_vert_start, av_vert_end=create_columns_for_hor_line(H,W)
    diff_nocrop_start=av_vert_start_diffs
    if len(av_vert_start)>2:
        for i in range(3):
            diff_nocrop_start.append(abs(av_vert_start[-1][i]-av_vert_start[-2][i]))
            diff_nocrop_start.append(abs(av_vert_start[-2][i]-av_vert_start[-3][i]))
        max_nocrop_start=np.max(diff_nocrop_start)
    else:
        max_nocrop_start=0
    
    diff_nocrop_end=av_vert_end_diffs
    if len(av_vert_end)>2:
        for i in range(3):
            diff_nocrop_end.append(abs(av_vert_end[-1][i]-av_vert_end[-2][i]))
            diff_nocrop_end.append(abs(av_vert_end[-2][i]-av_vert_end[-3][i]))
        max_nocrop_end=np.max(diff_nocrop_end)
    else:
        max_nocrop_end=0
    
    #print(diff_nocrop_start)
    #print('hor nocrop:', max_nocrop_start, max_nocrop_end)
    return max_nocrop_start, max_nocrop_end


#_______________________________________________________________________________

 
    # mozgó ablakok által meghatározott sávok a vízszintes vágási egyenes két
    # oldalán, ablakok átlagszíneinek kiszámítása
def moving_window_for_hor_line():
    #print('__________________________________________')
    #print('moving_window_for_hor_line')
    av_hor_in=[]
    av_hor_out=[]
    if h>=20:
        step=20
    else:
        step=h
    for i in range(0, image.shape[1],step):
        window_inside=image[h:h+step, i:i+step]
        window_outside=image[h-step:h,i:i+step]
        mean_col_inside=tuple(reversed(window_inside.mean(axis=0).mean(axis=0)))
        mean_col_outside=tuple(reversed(window_outside.mean(axis=0).mean(axis=0)))
        av_hor_in.append(mean_col_inside)
        av_hor_out.append(mean_col_outside)
    av_hor_in=np.asarray(av_hor_in)
    av_hor_out=np.asarray(av_hor_out)      
    return av_hor_in, av_hor_out 


#_______________________________________________________________________________

    
    # a vízszintes vágási egyenes két oldalán lévő sáv mozgó ablakai átlagszínének
    # különségei, a különbségek maximuma, mindig az egyenes alatti négyzetet 
    # hasonlítjuk az egyenes felettivel
def diff_of_moving_window_for_hor_line():
    #print('__________________________________________')
    #print('diff_of_moving_window_for_hor_line')
    av_hor_in, av_hor_out=moving_window_for_hor_line()
    hor_diff=[]
    for i in range(1,int(round(len(av_hor_in)/3))):
        for j in range(3):
            hor_diff.append(abs(av_hor_in[i][j]-av_hor_out[i][j]))
    for i in range(int(2*round(len(av_hor_in)/3)),len(av_hor_in)):
        for j in range(3):
            hor_diff.append(abs(av_hor_in[i][j]-av_hor_out[i][j]))
    max_diff_hor=np.max(hor_diff)
    hor_diff.remove(max_diff_hor)
    max_diff_hor=np.max(hor_diff)
    #print('max_diff_hor: ', max_diff_hor)
    return max_diff_hor

#_______________________________________________________________________________
    
    # a vízszintes vágási egyenesen belül lévő sáv átlagszíneinek különbségének 
    # maximuma
def diffs_of_moving_window_for_hor_line_inside():
    #print('__________________________________________')
    #print('diffs_of_moving_window_for_hor_line_inside')
    av_hor_in, av_hor_out=moving_window_for_hor_line()
    hor_in_diff=[]
    for i in range(1,int(round(len(av_hor_in)/3))):
        for j in range(3):
            hor_in_diff.append(abs(av_hor_in[i][j]-av_hor_in[i-1][j]))
    for i in range(int(2*round(len(av_hor_in)/3)),len(av_hor_in)):
        for j in range(3):
            hor_in_diff.append(abs(av_hor_in[i][j]-av_hor_in[i-1][j]))
    max_diff_hor_in=np.max(hor_in_diff)
    hor_in_diff.remove(max_diff_hor_in)
    max_diff_hor_in=np.max(hor_in_diff)
    return max_diff_hor_in
 
 
#_______________________________________________________________________________ 
        
    # a vízszintes vágási egyenesen kívül lévő vízszintes sávátlagszíneinek 
    # különbségének maximuma    
def diffs_of_moving_window_for_hor_line_outside():
    #print('__________________________________________')
    #print('diffs_of_moving_window_for_hor_line_outside')
    av_hor_in, av_hor_out=moving_window_for_hor_line() 
    hor_out_diff=[]
    for i in range(1,int(round(len(av_hor_in)/3))):
        for j in range(3):
            hor_out_diff.append(av_hor_out[i][j]-av_hor_out[i-1][j])
    for i in range(int(2*round(len(av_hor_in)/3)),len(av_hor_in)):
        for j in range(3):
            hor_out_diff.append(av_hor_out[i][j]-av_hor_out[i-1][j])
    max_diff_hor_out=np.max(hor_out_diff)  
    hor_out_diff.remove(max_diff_hor_out)
    max_diff_hor_out=np.max(hor_out_diff)  
    return max_diff_hor_out


#_______________________________________________________________________________

    
def final_decision():
    print(' ')
    print(' ')
    print('__________________________________________')
    print('final_decision')
    h=find_the_lines.h-3
    h_next=find_the_lines.h_next-3
    w=find_the_lines.w+3
    w_next=find_the_lines.w_next+3
    crop_hor, crop_vert=False,False
    if h<=0 and image.shape[1]-w<=6:
        crop_hor=False
        crop_vert=False
        print('3', crop_vert, crop_hor)
        
    else:
        if image.shape[1]-w<=6:
            crop_vert=False
            print('1')
        else:
            if (max_nocrop_vert_start<5 or max_nocrop_vert_end<5) and abs(max_nocrop_vert_start-max_nocrop_vert_end)<10:
                crop_vert=False
                print('4', crop_vert, crop_hor)
            elif max_diff_vert>15 and (av_hor_start_max<10 or av_hor_end_max<10) and abs(av_hor_start_max-av_hor_end_max)<10:
                if max_nocrop_vert_start<5 or max_nocrop_vert_end<5:
                    crop_vert=False
                else:
                    crop_vert=True
                print('5', crop_vert, crop_hor)
            elif max_diff_vert<15 or av_hor_end_max>5 or av_hor_end_max>5:
                w=w_next
                max_nocrop_vert_start2, max_nocrop_vert_end2=diffs_for_nocrop_vert(h_next,w)
                max_diff_vert2=diff_of_moving_window_for_vert_line()
                av_hor_start_diffs, av_hor_end_diffs,av_hor_start_max2, av_hor_end_max2=diffs_of_rows_for_vert_lines(h,w_next)
                print('max_nocrop_vert_start2, max_nocrop_vert_end2: ', max_nocrop_vert_start2, max_nocrop_vert_end2)
                if (av_hor_start_max2<5 or av_hor_end_max2<5) and max_nocrop_vert_start2<15 and max_nocrop_vert_end2<15 and max_diff_vert2<15:
                    crop_vert=False
                    print('6', crop_vert, crop_hor)
                else:
                    crop_vert=True  
                    print('7', crop_vert,crop_hor)  
            else:
                print('else_vert')
            
        if h<=0:
            crop_hor=False
        else:
            if (max_nocrop_hor_start<5 or max_nocrop_hor_end<5) and abs(max_nocrop_hor_start-max_nocrop_hor_end)<10:
                crop_hor=False  
                print('8', crop_vert,crop_hor)  
            elif max_diff_hor>15 and (av_vert_start_max<5 or av_vert_end_max<5):
                if max_nocrop_hor_start<5 or max_nocrop_hor_end<5:
                    crop_hor=False
                    print('8.5')
                else:
                    crop_hor=True
                    print('9', crop_vert,crop_hor)  
            elif max_diff_hor<15 or av_vert_end_max>5 or av_vert_end_max>5:
                h=h_next
                max_diff_hor2=diff_of_moving_window_for_hor_line()
                max_nocrop_hor_start2, max_nocrop_hor_end2=diffs_for_nocrop_hor(h,w_next)
                av_vert_start_diffs, av_vert_end_diffs, av_vert_start_max2, av_vert_end_max2=diffs_of_columns_for_hor_line(h_next,w)
                print('av_vert_start_max2, av_vert_end_max2: ', av_vert_start_max2, av_vert_end_max2)
                print('max_nocrop_hor_start2, max_nocrop_hor_end2: ', max_nocrop_hor_start2, max_nocrop_hor_end2)
                if max_nocrop_hor_start2<15 and max_nocrop_hor_end2<15 and (av_vert_start_max2<5 or av_vert_end_max2<5) and max_diff_hor2<15:
                    crop_hor=False
                    print('10', crop_vert,crop_hor)  
                else:
                    crop_hor=True 
                    print('11', crop_vert,crop_hor)  
            else:
                print('else')
    print(h,w)       
    return crop_vert, crop_hor, h, w


#_______________________________________________________________________________

if table_guess != 'nincs asztal':
    print(h,w)
    if h>0 and w<image.shape[1]-6:
        max_diff_before_hor_line=check_diffs_before_hor_line()
        max_diff_after_vert_line=check_diffs_after_vert_line()
        max_diff_cross=lines_cross_diff()
    else:
        max_diff_before_hor_line=np.nan
        max_diff_after_vert_line=np.nan
        max_diff_cross=np.nan
    #________________________________________________________________
    if w<image.shape[1]-6:
        av_hor_start_diffs, av_hor_end_diffs,av_hor_start_max, av_hor_end_max=diffs_of_rows_for_vert_lines(h,w)
        max_nocrop_vert_start, max_nocrop_vert_end=diffs_for_nocrop_vert(h,w)
        max_diff_vert=diff_of_moving_window_for_vert_line()
        max_diff_vert_inside=diff_of_moving_window_for_vert_line_inside()
        max_diff_vert_outside=diff_of_moving_window_for_vert_line_outside()

    #______________________________________________________________
    if h>0:
        av_vert_start_diffs, av_vert_end_diffs, av_vert_start_max, av_vert_end_max=diffs_of_columns_for_hor_line(h,w)
        max_nocrop_hor_start, max_nocrop_hor_end=diffs_for_nocrop_hor(h,w)
        max_diff_hor=diff_of_moving_window_for_hor_line()
        max_diff_hor_inside=diffs_of_moving_window_for_hor_line_inside()
        max_diff_hor_outside=diffs_of_moving_window_for_hor_line_outside()   

    
    crop_vert, crop_hor, h, w=final_decision()
    if w<4:
        crop_vert=False
    if h<4:
        crop_hor=False
    print(crop_vert,crop_hor,h,w)
    if crop_hor==True and crop_vert==False:
        cropped_img=image[h+3:, 0:].copy()
    elif crop_hor==False and crop_vert==True:
        cropped_img=image[:, :w-3].copy()
    elif crop_hor==True and crop_vert==True:
        cropped_img=image[h+3:, :w-3].copy() 
    else:
        cropped_img=image[:,:].copy()     
    print(table_guess)
    cv2.imshow('CROPPED IMAGE', cropped_img)
    cv2.waitKey(0)    
else:
    cropped_img=image[:,:].copy()     
    cv2.waitKey(0)   
 


    