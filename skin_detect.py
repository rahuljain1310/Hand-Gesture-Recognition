# import sklearn
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
from numba import vectorize,float64,uint8,guvectorize



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
@vectorize([uint8(uint8,uint8,uint8)])
def is_skin(B,G,R):
    gbyr = G/(R+1)
    bbyg = B/(G+1)
    bbyr = B/(R+1)
    R1 = False
    R2 = False
    R3 = False
    if (R>0.9) and G>B:
        R1 = 0.5941 <= gbyr and gbyr < 0.8992 
    else:
        R1 = 0.4412 <= gbyr and gbyr <0.8686
    
    if (B>0.85):
        R2 = 0.8255 <= bbyr and bbyr <= 1.0262
    else:
        R2 = 0.4059 <= bbyr and bbyr <= 0.7902\
        or\
        gbyr < 0.6667 and bbyr <0.4059

    if (B>0.3333):
        R3 = 0.5157 <= bbyg and bbyg <= 1.0761
    else:
        R3 = 0.5157 <= bbyg and bbyg <= 0.8882\
        or\
        gbyr <0.6667 and bbyr <0.8882
    C =  R1 and R2 and R3

    if C:
        return np.uint8(255)
    else:
        return np.uint8(0)


@vectorize([uint8(uint8,uint8)])
def mask_nonskin(im_src, mask):
    return uint8((mask//255)*im_src)

def detect_skin(im):
    start_time = time.time()
    # im = cv2.imread(filename)
    w,h,_ = im.shape
    
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
    im_dst_l  = mask_pixel(*t)
    # im_dst_l  = is_skin(t[0],t[1],t[2])
    im_dst = im_dst_l.reshape(im.shape[0],im.shape[1])
   
    # fileroot = filename.rsplit('.',maxsplit=1)
    # # print(fileroot)
    # filesave = fileroot[0] + '_1.' + fileroot[1]
    # cv2.imwrite(filesave,im_dst)
    # print(t[0].shape)
    for i in range(3):
        t[i] = mask_nonskin(t[i],im_dst_l)
    # print(t[0].shape)
    im_masked = np.dstack((t[0],t[1],t[2])).reshape(im.shape[0],im.shape[1],3)
    print(im_masked.shape)
    elapsed_time = time.time() - start_time
    print(elapsed_time)

    return im_dst,im_masked

def get_ROI(im):
    # dilation
    kernel = np.ones((10, 1), np.uint8)
    img_dilation = cv2.dilate(im, kernel, iterations=1)
    cv2.imshow('dilated', img_dilation)

    # find contours
    # cv2.findCountours() function changed from OpenCV3 to OpenCV4: now it have only two parameters instead of 3
    cv2MajorVersion = cv2.__version__.split(".")[0]
    # check for contours on thresh
    if int(cv2MajorVersion) >= 4:
        ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = image[y:y + h, x:x + w]

        # show ROI
        # cv2.imshow('segment no:'+str(i),roi)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if w > 15 and h > 15:
            cv2.imwrite('C:\\Users\\PC\\Desktop\\output\\{}.png'.format(i), roi)

    cv2.imshow('marked areas', image)
    cv2.waitKey(0)

if __name__=='__main__':
    # parser = argparse.ArgumentParser(description = 'Skin Detector')
    # parser.add_argument('f', metavar = 'stop_front.png', type = str, help = 'Input File')
    # args = parser.parse_args()
    # im = cv2.imread(args.f)
    # vd = cv2.VideoCapture(0)
    ret = True
    # ret,im = vd.read()
    ret = False
    while(ret):
        w,h,_ = im.shape
        im = cv2.resize(im,(h//2,w//2))
        # def get_dillated():
        mask,im_masked = detect_skin(im)

        fileroot = args.f.rsplit('.',maxsplit=1)
        # print(fileroot)
        filesave = fileroot[0] + '_1.' + fileroot[1]

        kernel = np.ones((5, 5), np.uint8)
        img_dilation = cv2.erode(mask, kernel, iterations=1)
        img_dilation = cv2.dilate(mask, kernel, iterations=5)
        # cv2.imshow('dilated', img_dilation)
        # cv2.imwrite('undillated.png',mask)
        # cv2.imwrite('dillated.png',img_dilation)
        cv2MajorVersion = cv2.__version__.split(".")[0]
        # check for contours on thresh
        if int(cv2MajorVersion) >= 4:
            ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)

            # Getting ROI
            roi = im_masked[y:y + h, x:x + w]

            # show ROI
            # cv2.imshow('segment no:'+str(i),roi)
            if w > 50 and h > 50:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Hand',im)
        waitk = cv2.waitKey(30)==27
        # 
        ret,im = vd.read()
        # cv2.imwrite('marked_areas.jpeg', im)
        # cv2.waitKey(30)


@guvectorize([(uint8[:], uint8[:])], '(n), (m)')
def classify_image(im,cim):
    pass

# @vectorize
def c_image(im):
    a=  np.average(im.reshape(im.shape[0]*im.shape[1]*im.shape[2]))
    if (a>255/2):
        return 255
    else:
        return 0

def sliding_window(im,window,scale,scale_factor):

    # im = im1.resize((640,320,3))
    # outuput needed to be 50 X 50
    # step size = 5
    outputs = []
     
    l,b= window
    step1  = l//scale
    step2 = b//scale

    # num_scales = 5
    # scale_factor = 1.5
    i=0
    w,h,_ = im.shape

    while(True):
        # print(i)
        im = np.resize(im,(int(w/scale_factor),int(h/scale_factor),3))
        w,h,_ = im.shape
        x,y = (w-l)//step1+1,(h-b)//step2+1
        if (x<=0 or y<=0):
            break
        outputs.append(np.zeros((x,y),dtype=np.uint8))
        cim = outputs[len(outputs)-1]
        # print(x,y)
        print(w,h)
        print(w-l,h-b)
        for j in range(0,w-l,step1):
            for k in range(0,h-b,step2):
                # print(j,k)
                cim[j//step1][k//step2] = c_image(im[j:l+j,k:b+k])
        # print(cim)
        outputs[i] = pool2d(cim,2,1,0,'min')
        i+=1
    return outputs

def get_label(outputs):
    for i in range(len(outputs)):
        kernel = np.ones((5, 5), np.uint8)
        outputs[i] = cv2.erode(outputs[i], kernel, iterations=1)

from numpy.lib.stride_tricks import as_strided

    # cv2.imwrite('hand.jpeg',img_dilation)
def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size, 
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'min':
        return A_w.min(axis=(1,2)).reshape(output_shape)