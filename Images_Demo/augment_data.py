import os
import cv2
import sys
import argparse
import glob
from os import path
image_dir = 'Next'
import skimage
import PIL
# mypath = 'Previous'
# despath = 'Next'
from os import walk
# f = []
# for (dirpath, dirnames, filenames) in walk(mypath):
#     for fi in filenames:
#         if (fi.find('_r')==-1):
#             f.append(fi)
#     # print(filenames)
# #     # break
# for file1 in f:
#     im = cv2.imread(os.path.join(mypath,file1))
#     im_gen = cv2.flip(im,1)
#     tosave = file1.rsplit('.',maxsplit=1)
#     tosave = tosave[0] + '_r.' + tosave[1] 
#     cv2.imwrite('/'.join([despath,tosave]),im_gen)

# mypath = 'Previous_test'
# despath = 'Next_test'
# from os import walk
# f = []
# for (dirpath, dirnames, filenames) in walk(mypath):
#     for fi in filenames:
#         if (fi.find('_r')==-1):
#             f.append(fi)
#     # print(filenames)
# #     # break
# for file1 in f:
#     im = cv2.imread(os.path.join(mypath,file1))
#     im_gen = cv2.flip(im,1)
#     tosave = file1.rsplit('.',maxsplit=1)
#     tosave = tosave[0] + '_r.' + tosave[1] 
#     cv2.imwrite('/'.join([despath,tosave]),im_gen)
import matplotlib.pyplot as plt
paths = ['Next','Previous','Stop']

dest_paths = ['next_b','prev_b','stop_b']
def convert():
    
    a = [3 ,6, 8, 10 ,11 ,12 ,13, 15 ,16, 17 ,18 ,19]
    for mypath,despath in zip(paths,dest_paths):
        f = []
    # import matplotlib.pyplot as plt
        for (dirpath, dirnames, filenames) in walk(mypath):
            for fi in filenames:
                # if (fi.find(8_r')!=-1):
                if (fi.find('13')==0):
                # or fi.find('12')==0 or fi.find('13')==0):
                    f.append(fi)
            # print(filenames)
        #     # break
        for file1 in f:
            im = cv2.imread(os.path.join(mypath,file1))
            edges = cv2.Canny(im,threshold1 = 40,threshold2 = 140,L2gradient=True)
            # cv2.imwrite()
            # im_gen = cv2.flip(i,1)
            tosave = file1.rsplit('.',maxsplit=1)
            tosave = tosave[0] + '_b.' + tosave[1] 
            cv2.imwrite('/'.join([despath,tosave]),edges)


def get_edge_image(img):
    edges = cv2.Canny(img,threshold1 = 40,threshold2 = 140,L2gradient=True)
    cv2.imshow('h',edges)
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
def show_img(file):
    for x in paths:
        img = cv2.imread(os.path.join(x,file))
        get_edge_image(img)
        # plt.close()




# cv2.waitKey(100)
# print
# get_edge_image('17_2_21.jpg'mg)
# show_img('13_1_26.jpg')
#
convert()