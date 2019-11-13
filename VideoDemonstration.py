import cv2
from CNN_Architecture import Net

from Loader import loading
from DatasetPrepare import bgInit

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

### Initialization of Camera / Display Functions 
### ================================================================================================================================

vd = cv2.VideoCapture(0)

def displayFrame(frame, text):
  cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
  cv2.putText(frame, text, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
  cv2.imshow('Hand Recognition', frame)

### Transformation Functions and Background Substraction 
### ================================================================================================================================

bs = cv2.createBackgroundSubtractorKNN(history=4000)
print("Initializing Background...", end='\r', flush=True)
bgInit(vd,bs)

def transformImage(im):
  x = cv2.resize(im,(50,50))
  x = bs.apply(x)
  cv2.imshow('BS',x)
  return cv2.cvtColor(x,cv2.COLOR_GRAY2BGR)

### Fetching Neural Network
### ================================================================================================================================
print("Fetching Neural Network... ", end='\r', flush=True)

PATH = 'gesture_net.pth'
net = Net()
net.load_state_dict(torch.load(PATH))
classes = ('Background', 'Next', 'Previous' , 'Stop')

print("Neural Network Fetched.\t\t\t")

### Video Demonstraion
### ================================================================================================================================

ret,waitk = True, False
w = torchvision.transforms.ToTensor()
print("Capturing Frame ...\t\t", end='\r', flush=True)
try:
  while ret and not waitk :
    ret, im = vd.read()
    tensorIm = w(transformImage(im)).unsqueeze(0)
    output = net(tensorIm)
    _, predicted = torch.max(output, 1)
    outClass = classes[predicted]
    displayFrame(im,outClass)
    waitk = cv2.waitKey(30)==27
except: pass
finally: print("Record Finished.\t\t\t")