import sklearn
import numpy as np
import time
import argparse
# 0.0 <= H <= 50.0 and 0.23 <= S <= 0.68 and
# R > 95 and G > 40 and B > 20 and R > G and R > B and | R - G | > 15 and A > 15
# OR
# R > 95 and G > 40 and B > 20 and R > G and R > B and | R - G | > 15 and A > 15 and Cr > 135 and
# Cb > 85 and Y > 80 and Cr <= (1.5862*Cb)+20 and Cr>=(0.3448*Cb)+76.2069 and
# Cr >= (-4.5652*Cb)+234.5652 and
# Cr <= (-1.15*Cb)+301.75 and
# Cr <= (-2.2857*Cb)+432.85


import cv2
import matplotlib.pyplot as plt
import os
import PIL
from numba import vectorize,float64,uint8



# @vectorize([uint8[:,:](uint8[:,:],uint8[:,:],uin8[:,:])])
# def mask_pixel1(pixel):
#     return mask_pixel(pixel[0],pixel[1],pixel[2])
@vectorize([uint8(uint8,uint8,uint8,uint8,uint8,uint8,uint8,uint8,uint8)])
def mask_pixel(B,G,R,H,S,V,Y,Cr,Cb):
    # B,G,R = pixel_bgr
    # H,S,V = pixel_hsv
    # Y,Cr,Cb = pixel_ycrcb
    if (
        (0.0 <= H <= 50.0 and 0.23 <= S <= 0.68 and\
        R > 95 and G > 40 and B > 20 and R > G and R > B and abs(R - G) > 15)\
        or\
        (R > 95 and G > 40 and B > 20 and R > G and R > B and abs(R - G )> 15 and Cr > 135 and\
        Cb > 85 and Y > 80 and Cr <= (1.5862*Cb)+20 and Cr>=(0.3448*Cb)+76.2069 and\
        Cr >= (-4.5652*Cb)+234.5652 and\
        Cr <= (-1.15*Cb)+301.75 and\
        Cr <= (-2.2857*Cb)+432.85)
        # R > 10
    ):
        return np.uint8(255)
    else:
        return np.uint8(0)
    # hsv_pixel = cv2.cvtColor()


def detect_skin(filename):
    start_time = time.time()
    im = cv2.imread(filename)
    w,h,_ = im.shape
    im = cv2.resize(im,(h//2,w//2))
    # r,g,b = np.split(im,3,axis=2)
    
   
    im2 = cv2.cvtColor(im,cv2.COLOR_BGR2YCrCb)
    im3 = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)

    t = []
    for i in (im,im3,im2):
       t.extend(np.split(i,3,axis=2))


    # im_dst = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    for i in range(len(t)):
        t[i] = t[i].reshape(t[i].shape[0]*t[i].shape[1])
    # im_c = np.dstack((im,im2,im3)) 
    # w,h,_ = im.shape
    # for i in range(w):
    #     for j in range(h):
    #         im_dst[i][j] = mask_pixel(im[i][j],im2[i][j],im3[i][j])
    # cv2.imshow('Hand',im_dst)
    im_dst  = mask_pixel(*t)
    im_dst = im_dst.reshape(im.shape[0],im.shape[1])
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    fileroot = filename.rsplit('.',maxsplit=1)
    # print(fileroot)
    filesave = fileroot[0] + '_1.' + fileroot[1]
    cv2.imwrite(filesave,im_dst)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'Skin Detector')
    parser.add_argument('f', metavar = 'stop_front.png', type = str, help = 'Input File')
    args = parser.parse_args()
    detect_skin(args.f)