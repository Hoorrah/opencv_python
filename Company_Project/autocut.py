from os import listdir
import cv2
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
import os
from os.path import join, basename
import argparse
import pandas as pd
from PIL.ImageOps import crop
from datetime import datetime



dir_path = r'C:/Users/Windows10/Desktop/Internship/Image/onebook'
save_dir_path=r'C:/Users/Windows10/Desktop/Internship/Image'


WIDTH = 20
Nsimilar = 40
similar = 15
Vsimilar = 10
same = 5
vert_line_bound = 450
hor_line_bound = 250
line_diff = 3



#_______________________________________________________________________________

    #a 20 pixel high bar at the top of the image with moving windows
def moving_window_vert():
    av_vert = []
    for i in range(0, image.shape[0],WIDTH):
        window = image[i:i+WIDTH,image.shape[1]-WIDTH:image.shape[1]]
        mean_color = window.mean(axis=0).mean(axis=0)
        mean_color = tuple(reversed(mean_color))
        av_vert.append(mean_color)
    return av_vert

#_______________________________________________________________________________

    #a 20-pixel-wide band on the right side of the image with moving windows
def moving_window_hor():
    av_hor = []
    for i in range(0, image.shape[1],WIDTH):
        window = image[0:WIDTH,i:i+WIDTH]
        mean_color = window.mean(axis=0).mean(axis=0)
        mean_color = tuple(reversed(mean_color))
        av_hor.append(mean_color)
    return av_hor

#_______________________________________________________________________________

    #to display the right bar
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

    # to display the top bar
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

    # representation of the bars
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

   # creating the arrays needed for the first round decision
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

     # the first round decision
def create_right_up_both_guess():
    right, up, both = create_right_up_both_arrays()
    guess = ''
    corner = ''
    if right <Nsimilar and up >Nsimilar:
        guess = 'jobbra van asztal'
    elif right >Nsimilar and up<Nsimilar:
        guess = 'fel??l van asztal'
    elif right<Nsimilar and up<Nsimilar:
        guess = 'fel??l ??s jobbra van asztal'
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



    #a function made of bands
def create_function(stripe):
    func = np.zeros((len(stripe), ), float)
    for i in range(len(stripe)-1,1, -1):
        func[i] = abs(stripe[i]-stripe[i-1])
    return func

#_______________________________________________________________________________


    # finding the first (main) possible vertical cutting line
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

    # finding the second (secondary) possible vertical cutting line
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

    # finding the first (main) horizontal cutting line
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

    # finding a second (secondary) possible cutting line
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

    # the horizontal bar required for vertical cropping at the bottom of the image and to it
    # possible cutting coordinates
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


    # a f??gg??leges v??g??hoz sz??ks??ges v??zszintes s??v a k??p k??zep??n ??s a hozz??
    # tartoz?? lehets??ges v??g??si koordin??t??k
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

    # als?? ??s k??z??ps?? v??zszintes s??vok k??z??tti d??nt??s a v??g??si koordin??t??k alapj??n,
    # a v??gs?? els??dleges ??s m??sodlahos f??gg??leges v??g??si egyenes koordin??t??inak
    # meghat??roz??sa
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

    # a v??zszintes v??g??shoz sz??ks??ges f??gg??leges s??v a k??p elej??n (bal sz??l??n) ??s a hozz??
    # tartoz?? lehets??ges v??g??si koordin??t??k
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

    # a v??zszintes v??g??shoz sz??ks??ges f??gg??leges s??v a k??p k??zep??n ??s a hozz??
    # tartoz?? lehets??ges v??g??si koordin??t??k
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

    # balsz??ls?? ??s k??z??ps?? f??gg??leges s??vok k??z??tti d??nt??s a v??g??si koordin??t??k alapj??n,
    # a v??gs?? els??dleges ??s m??sodlahos v??zszintes v??g??si egyenes koordin??t??inak
    # meghat??roz??sa
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

    # a s??vokb??l k??sz??tett f??ggv??ny ??br??zol??sa
def plot_function(func, title):
    fig = plt.figure()
    DPI = fig.get_dpi()
    fig.set_size_inches(1920.0/float(DPI),1080.0/float(DPI))
    plt.plot(func)
    plt.title(title)
    plt.show()


#_______________________________________________________________________________


    # els??dleges v??g??si egyenesek ??br??zol??sa a k??peken az els?? k??r??s d??nt??s alapj??n
def draw_the_lines():
    h, h_next, w, w_next=0,0,0,0
    if table_guess=='jobbra van asztal':
        w, w_next = find_vert_crop_coords()
        lineThickness = 2
        line_img = cv2.line(image, (w, 0), (w, image.shape[0]), (0,255,0), lineThickness)
        #cv2.imshow('Line', line_img)
        #cv2.waitKey(0)
    if table_guess=='fel??l van asztal':
        h, h_next = find_hor_crop_coords()
        lineThickness = 2
        line_img = cv2.line(image, (0,h), (image.shape[0],h), (0,255,0), lineThickness)
        #cv2.imshow('Line', line_img)
        #cv2.waitKey(0)
    if table_guess=='fel??l ??s jobbra van asztal' :
        w,w_next = find_vert_crop_coords()
        h,h_next = find_hor_crop_coords()
        lineThickness = 2
        line_img = cv2.line(image, (w, 0), (w, image.shape[0]), (0,255,0), lineThickness)
        line_img = cv2.line(image, (0,h), (image.shape[0],h), (0,255,0), lineThickness)
        #cv2.imshow('Line', line_img)
        #cv2.waitKey(0)
    cv2.destroyAllWindows()
    return w, w_next, h, h_next



#_______________________________________________________________________________

    # mozg?? ablakok ??ltal meghat??rozott f??gg??leges s??v a lehets??ges f??gg??leges
    # v??g??i egyenes k??t oldal??n,
    # a k??p tetej??t??l a lehets??ges v??zszintes v??g??si egyenesig tart
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

    # a sz??nek komponensenk??nt vett k??l??nbs??g??nek maximuma a lehets??ges f??gg??leges
    # v??g??si egyenes k??t oldal??n
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

    # mozg?? ablakok ??ltal meghat??rozott v??zszintes s??v a lehets??ges v??zszintes
    # v??g??i egyenes k??t oldal??n,
    # a v??zszintes v??g??si egyenest??l a k??p v??g??ig (jobbsz??l??ig) tart
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

    # a sz??nek komponensenk??nt vett k??l??nbs??geinek maximuma a lehets??ges v??zszintes v??g??si
    # egyenes k??t oldal??n
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


    # a f??gg??leges ??s v??zsintes v??g??si egyenesek metsz??spontja k??r??li h??rom
    # n??gyzetb??l ??ll?? "sarok" l??trehoz??sa
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

    # a f??gg??leges ??s v??zsintes v??g??si egyenesek metsz??spontja k??r??li h??rom
    # n??gyzetb??l ??ll?? "sarok" ??ltagsz??neinek ??br??zol??sa
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

    # a f??gg??leges ??s v??zsintes v??g??si egyenesek metsz??spontja k??r??li h??rom
    # n??gyzetb??l ??ll?? "sarok" ??ltagsz??neinek sz??m??t??sa
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

    # a f??gg??leges ??s v??zsintes v??g??si egyenesek metsz??spontja k??r??li h??rom
    # n??gyzetb??l ??ll?? "sarok" ??ltagsz??neinek k??l??nbs??g??nek maximuma
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

    # a f??gg??leges v??g??i egyenest??l a k??p jobbsz??l??ig tart?? v??zszintes s??vok
    # l??trehoz??sa mozg?? ablakokkal, egy a k??p alj??hoz k??zel, egy a v??zszintes
    # v??g??si egyeneshez k??zel, a mozg?? ablakok ??tlagsz??neinek sz??m??t??sa
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

    # az el??bb l??trehozott v??zszintes s??vok mozg?? ablakai ??ltagsz??n??nek k??l??nbs??g??nek
    # ??tlaga
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

    # az el??bbi mozg?? ablakokhoz hozz??vesz??nk m??g kett??t, ami m??r a f??gg??leges
    # v??g??si egyenesen bel??l van, ??s kisz??m??tjuk ennek ??s az el??bbi s??vnak a
    # k??l??nbs??geinek maximum??t
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

    # a f??gg??leges v??g??si egyenes k??t oldal??n l??v?? v??zszintes, mozg?? ablakok
    # ??ltal meghat??rozott s??v
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

    # a f??gg??leges v??g??si egyenes k??t oldal??n l??v?? v??zszintes, mozg?? ablakok
    # ??ltal meghat??rozott s??v mozg?? ablakai ??tlagsz??n??nek k??l??nbs??gei a k??t oldalon,
    # teh??t mindig a bels?? (baloldali) n??gyzetet a k??ls?? (jobboldali) n??gyzethez
    # hasonl??tjuk, meghat??rozzuk a k??l??nbs??gek maximum??t
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

    # a f??gg??leges v??g??si egyenes baloldal??n (bel??l) l??v?? mozg?? ablakok ??ltal
    # meghat??rozott s??v mozg?? ablakainek k??l??nbs??gei, a k??l??nbs??g maximuma
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

    # a f??gg??leges v??g??si egyenes jobboladl??n (k??v??l) l??v?? mozg?? ablakok ??ltal
    # meghat??rozott s??v mozg?? ablakainek k??l??nbs??gei, a k??l??nbs??gek maximuma
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

    # a k??p tetej??t??l  a v??zszintes v??g??i egyenesig tart?? f??gg??leges s??vok
    # l??trehoz??sa mozg?? ablakokkal, egy a k??p baloldal??hoz k??zel, egy a f??gg??leges
    # v??g??si egyeneshez k??zel, a mozg?? ablakok ??tlagsz??neinek sz??m??t??sa
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

    # az el??bb l??trehozott f??gg??leges s??vok ??tlagsz??neinek k??l??nbs??ge, a k??l??nbs??g
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

    # az el??bbi mozg?? ablakok ??ltal meghat??rozott s??vhoz hozz??vesz??nk m??g k??t
    # n??gyzetet, azt m??r a v??g??si egyenesen bel??l ??s kisz??m??tjuk ennek ??s az
    # eg??sz el??bbi s??vnak a k??l??ns??g??t, a k??l??ns??gek maximum??t
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

    # azon s??v mozg?? ablakai ??tlagsz??n??neinek ??br??zol??sa, amely a v??g??si egyenesen
    # k??v??l van, de a v??g??si egyenesen bel??l is hozz??vesz??nk k??t n??gyzetet
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

    # mozg?? ablakok ??ltal meghat??rozott s??vok a v??zszintes v??g??si egyenes k??t
    # oldal??n, ablakok ??tlagsz??neinek kisz??m??t??sa
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

    # a v??zszintes v??g??si egyenes k??t oldal??n l??v?? s??v mozg?? ablakai ??tlagsz??n??nek
    # k??l??ns??gei, a k??l??nbs??gek maximuma, mindig az egyenes alatti n??gyzetet
    # hasonl??tjuk az egyenes felettivel
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

    # a v??zszintes v??g??si egyenesen bel??l l??v?? s??v ??tlagsz??neinek k??l??nbs??g??nek
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

    # a v??zszintes v??g??si egyenesen k??v??l l??v?? v??zszintes s??v??tlagsz??neinek
    # k??l??nbs??g??nek maximuma
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


def final_decision(h,h_next, w, w_next):
    #print(' ')
    #print(' ')
    #print('__________________________________________')
    #print('final_decision')
    crop_hor, crop_vert=False,False
    if h<=0 and image.shape[1]-w<=6:
        crop_hor=False
        crop_vert=False
        #print('3', crop_vert, crop_hor)

    else:
        if image.shape[1]-w<=6:
            crop_vert=False
            #print('1')
        else:
            if (max_nocrop_vert_start<same or max_nocrop_vert_end<same) and abs(max_nocrop_vert_start-max_nocrop_vert_end)<Vsimilar:
                crop_vert=False
                #print('4', crop_vert, crop_hor)
            elif max_diff_vert>similar and (av_hor_start_mean<Vsimilar or av_hor_end_mean<Vsimilar) and abs(av_hor_start_mean-av_hor_end_mean)<Vsimilar:
                if max_nocrop_vert_start<same or max_nocrop_vert_end<same:
                    crop_vert=False
                else:
                    crop_vert=True
                #print('5', crop_vert, crop_hor)
            elif max_diff_vert<similar or av_hor_end_mean>same or av_hor_end_mean>same:
                w=w_next
                max_nocrop_vert_start2, max_nocrop_vert_end2=diffs_for_nocrop_vert(h_next,w)
                max_diff_vert2=diff_of_moving_window_for_vert_line()
                av_hor_start_diffs, av_hor_end_diffs,av_hor_start_mean2, av_hor_end_mean2=diffs_of_rows_for_vert_lines(h,w_next)
                #print('max_nocrop_vert_start2, max_nocrop_vert_end2: ', max_nocrop_vert_start2, max_nocrop_vert_end2)
                if (av_hor_start_mean2<same or av_hor_end_mean2<same) and max_nocrop_vert_start2<similar and max_nocrop_vert_end2<similar and max_diff_vert2<similar:
                    crop_vert=False
                    #print('6', crop_vert, crop_hor)
                else:
                    crop_vert=True
                    #print('7', crop_vert,crop_hor)
            else:
                print('')

        if h<=0:
            crop_hor=False
        else:
            if (max_nocrop_hor_start<same or max_nocrop_hor_end<same) and abs(max_nocrop_hor_start-max_nocrop_hor_end)<Vsimilar:
                crop_hor=False
                #print('8', crop_vert,crop_hor)
            elif max_diff_hor>similar and (av_vert_start_max<same or av_vert_end_max<same):
                if max_nocrop_hor_start<same or max_nocrop_hor_end<same:
                    crop_hor=False
                    #print('8.5')
                else:
                    crop_hor=True
                    #print('9', crop_vert,crop_hor)
            elif max_diff_hor<similar or av_vert_end_max>same or av_vert_end_max>same:
                h=h_next
                max_diff_hor2=diff_of_moving_window_for_hor_line()
                max_nocrop_hor_start2, max_nocrop_hor_end2=diffs_for_nocrop_hor(h,w_next)
                av_vert_start_diffs, av_vert_end_diffs, av_vert_start_max2, av_vert_end_max2=diffs_of_columns_for_hor_line(h_next,w)
                #print('av_vert_start_max2, av_vert_end_max2: ', av_vert_start_max2, av_vert_end_max2)
                #print('max_nocrop_hor_start2, max_nocrop_hor_end2: ', max_nocrop_hor_start2, max_nocrop_hor_end2)
                if max_nocrop_hor_start2<similar and max_nocrop_hor_end2<similar and (av_vert_start_max2<same or av_vert_end_max2<same) and max_diff_hor2<similar:
                    crop_hor=False
                    #print('10', crop_vert,crop_hor)
                else:
                    crop_hor=True
                    #print('11', crop_vert,crop_hor)
            else:
                print('')
    #print(h,w)
    return crop_vert, crop_hor, h, w



start_time = datetime.now()

imgs = [join(dir_path,f) for f in listdir(dir_path)]


# K??PEK BEOLVAS??SA
for i in range(len(imgs)):
    filen=str(i+1)+'.png'
    img=imgs[i]
    print(' ')
    print(' ')
    print('__________________________________________')
    str2=img.split('/')
    title=str2[6].split('-')[:-2]
    print(' '.join(title))
    image=cv2.imread(img)
    image_copy=image.copy()
    av_vert=moving_window_vert()
    av_hor=moving_window_hor()
    av_vert=np.asarray(av_vert)
    av_hor=np.asarray(av_hor)
    table_guess,  right, up=create_right_up_both_guess()


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    if table_guess!='nincs asztal':
        w, w_next, h, h_next=draw_the_lines()

    image=image_copy
    if table_guess=='fel??l van asztal':
        h=h-line_diff
        h_next=h_next-line_diff
        w=0
        w_next=0
    if table_guess=='jobbra van asztal':
        w=w+line_diff
        w_next=w_next+line_diff
        h=0
        h_next=0
    if table_guess=='fel??l ??s jobbra van asztal':
        h=h-line_diff
        h_next=h_next-line_diff
        w=w+line_diff
        w_next=w_next+line_diff

    if table_guess != 'nincs asztal':
        #print(h,w)
        if h>0 and w<image.shape[1]-2*line_diff:
            max_diff_before_hor_line=check_diffs_before_hor_line()
            max_diff_after_vert_line=check_diffs_after_vert_line()
            max_diff_cross=lines_cross_diff()
        else:
            max_diff_before_hor_line=np.nan
            max_diff_after_vert_line=np.nan
            max_diff_cross=np.nan
        #________________________________________________________________
        if w<image.shape[1]-2*line_diff:
            av_hor_start_diffs, av_hor_end_diffs,av_hor_start_mean, av_hor_end_mean=diffs_of_rows_for_vert_lines(h,w)
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


     # V??G??S
        crop_vert, crop_hor, h, w=final_decision(h,h_next, w, w_next)
        if w<line_diff:
            crop_vert=False
        if h<line_diff:
            crop_hor=False
        print('vertical crop: ', crop_vert, '    horizontal crop: ', crop_hor)
        print('vertical line: ', w, '     horizontal line: ', h)
        if crop_hor==True and crop_vert==False:
            cropped_img=image[h+line_diff:, 0:].copy()
        elif crop_hor==False and crop_vert==True:
            cropped_img=image[:, :w-line_diff].copy()
        elif crop_hor==True and crop_vert==True:
            cropped_img=image[h+line_diff:, :w-line_diff].copy()
        else:
            cropped_img=image[:,:].copy()

    else:
        cropped_img=image[:,:].copy()






    cv2.imwrite(join(save_dir_path, filen), cropped_img)
end_time = datetime.now()
print('Duration: {}'.format((end_time - start_time)/5))
