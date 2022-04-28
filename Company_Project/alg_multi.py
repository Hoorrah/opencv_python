import time
import multiprocessing 
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import join, basename
import argparse
from numpy.core.fromnumeric import argmax
import timeit
plt.style.use('ggplot')


W,H, W2, H2=600,80,300,400
NUM_CHANNELS = 3
crop_value=5
dir_path = '/mnt/paris/share/kellerd/vagott_kepek/'
# /develop/kellerd/python_ws/Antik/vágott_képek/
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
    help="type of preprocessing to be done")
args = vars(ap.parse_args())
orig_im = cv2.imread(args["image"])
orig_im = cv2.resize(orig_im, (W,H)) 
batch_size = 20
num_images = 100

num_threads = int(num_images/batch_size)  # 20
imgs = [join(dir_path,f) for f in listdir(dir_path)]

def initialize_queue():
    queue = Queue()
    return queue

def get_img_label(fpath):
    img_name=(fpath.split('/')[-1]).split('.')[0]
    return img_name

def function(colors_arr):
    func=np.zeros((3, W), float)
    for i in range(3):
        for j in range(W):
            func[i][j]=colors_arr[i][j]
        
    return func

def gray_function(gray_arr):
    func=np.zeros((W,), float)
    for i in range(W):
        func[i]=gray_arr[i]
        
    return func

def get_colors(image):
    reds=np.zeros((W,), float)
    greens=np.zeros((W,), float)
    blues=np.zeros((W,), float)
    for i in range(W):
        reds[i]=math.fsum(image[k][i][0] for k in range(H))
        greens[i]=math.fsum(image[k][i][1] for k in range(H))
        blues[i]=math.fsum(image[k][i][2] for k in range(H))
       
    return [reds, greens, blues]

def get_graylevel(image):
    grays=np.zeros((W,), float)
    for i in range(W):
        grays[i]=math.fsum(image[k][i] for k in range(H))
    return grays


def get_lag(orig_func, func):
    shifted_func=np.zeros((3, W), float)
    for i in range(3):
        event1 = [int(x) for x in orig_func[i]]
        event2 = [int(x) for x in func[i]]
        xcor = np.correlate(event1, event2, "full")
        nR = np.argmax(xcor)
        maxLag = int(-np.argmax(xcor)+W)
        if maxLag>0:
            for j in range(-maxLag,W-maxLag):
                shifted_func[i][j+maxLag]=orig_func[i][j]
                func2=func
        else:
            maxLag=int(-np.argmax(xcor)+W)+W
            for j in range(-maxLag,W-maxLag):
                shifted_func[i][j+maxLag]=func[i][j]
                func2=orig_func
        shifted_func =function(shifted_func)
        shifted_func=np.asarray(shifted_func)
    return shifted_func, func2


def get_graylag(orig_func, func):
    shifted_func=np.zeros((W,), float)
    event1 = [int(x) for x in orig_func]
    event2 = [int(x) for x in func]
    xcor = np.correlate(event1, event2, "full")
    nR = np.argmax(xcor)
    maxLag = int(-np.argmax(xcor)+W)
    if maxLag>0:
        for j in range(-maxLag,W-maxLag):
            shifted_func[j+maxLag]=orig_func[j]
            func2=func
    else:
        maxLag=int(-np.argmax(xcor)+W)+W
        for j in range(-maxLag,W-maxLag):
            shifted_func[j+maxLag]=func[j]
            func2=orig_func
    shifted_func =gray_function(shifted_func)
    shifted_func=np.asarray(shifted_func)
    return shifted_func, func2
    
def get_correlation(shifted_func, func2):
    coeff=np.zeros((3, ), float)
    for i in range(3):
        coeff[i]=round(np.corrcoef(shifted_func[i], func2[i])[0,1]*100,2)
    return coeff

def get_gray_correlation(shifted_func, func2):
    coeff=round(np.corrcoef(shifted_func, func2)[0,1]*100,2)
    return coeff

def get_images(i,orig_func,orig_grayfunc,send_end):
    img=imgs[i]
    img_name=(img.split('/')[-1]).split('.')[0]
    img=cv2.imread(img)
    img=cv2.resize(img, (W,H)) 
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_colors=get_colors(img)
    img_colors=np.asarray(img_colors)
    img_gray=get_graylevel(img_gray)
    func=function(img_colors)
    gray_func=gray_function(img_gray)
    
    coeff_gray_orig=get_gray_correlation(orig_grayfunc, gray_func)
    coeff_orig=get_correlation(orig_func, func)
    coeff_arr_orig[i]=[img_name,np.mean(coeff_orig)]
    gray_shifted_func, gray_func2=get_graylag(orig_grayfunc,gray_func)
    shifted_func, func2=get_lag(orig_func, func)
    coeff=get_correlation(shifted_func, func2)
    gray_coeff=get_gray_correlation(gray_shifted_func, gray_func2)
    if np.mean(coeff)<np.mean(coeff_orig):
        coeff=np.mean(coeff_orig)
    if gray_coeff<coeff_gray_orig:
        gray_coeff=coeff_gray_orig
    coeff_arr[i]=[img_name,round(np.mean(coeff),2), round(np.mean(gray_coeff),2)]  
    send_end.send(coeff_arr[i])
    
    


if __name__ == '__main__':
    starttime = time.time()
    processes = []
    orig_im0 = cv2.imread(args["image"])
    orig_im = cv2.resize(orig_im0, (W,H)) 
    orig_im_colors=get_colors(orig_im)
    orig_im_gray=cv2.cvtColor(orig_im, cv2.COLOR_BGR2GRAY)
    orig_im_colors=np.asarray(orig_im_colors)
    orig_func=function(orig_im_colors)
    orig_gray=get_graylevel(orig_im_gray)
    orig_grayfunc=gray_function(orig_gray)
    coeff_arr=np.zeros((len(imgs),3), float)
    coeff_arr_orig=np.zeros((len(imgs),2), float)
    pipe_list = []
    for i in range(1000):
        print(i)
        recv_end, send_end = multiprocessing.Pipe(False)
        p = multiprocessing.Process(target=get_images,args=(i,orig_func, orig_grayfunc, send_end))
        processes.append(p)
        pipe_list.append(recv_end)
        p.start()
    for process in processes:
            send_end.close()
            process.join()

    coeff_arr = [x.recv() for x in pipe_list]
    print('That took {} seconds'.format(time.time() - starttime))
    coeff_arr=np.array(coeff_arr)
    ordered=coeff_arr[coeff_arr[:,2].argsort()]
    print(ordered[-5:][::-1])

 
    
    im1=cv2.imread(dir_path +str(int(ordered[-1][0]))+'.png')
    print(str(int(ordered[-1][0])))
    im1= cv2.resize(im1, (W2,H2))
    im2=cv2.imread(dir_path +str(int(ordered[-2][0]))+'.png')
    im2= cv2.resize(im2, (W2,H2))
    
    im3=cv2.imread(dir_path +str(int(ordered[-3][0]))+'.png')
    im3= cv2.resize(im3, (W2,H2))
    im4=cv2.imread(dir_path +str(int(ordered[-4][0]))+'.png')
    im4= cv2.resize(im4, (W2,H2))
    im5=cv2.imread(dir_path +str(int(ordered[-5][0]))+'.png')
    im5= cv2.resize(im5, (W2,H2))
    cv2.imshow('SIMILAR PHOTOS', np.hstack((im1, im2, im3,im4,im5)))
    cv2.imshow('ORIGINAL IMAGE', orig_im0)
    cv2.waitKey(0)
 

    