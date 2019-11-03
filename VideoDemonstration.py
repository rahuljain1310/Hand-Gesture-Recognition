import cv2
from CNN_Architecture import Net

from Loader import loading
from DatasetPrepare import bgInit

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

### Initialization of Camera and Background Substraction
### ================================================================================================================================

vd = cv2.VideoCapture(0)
bs = cv2.createBackgroundSubtractorKNN(history=4000)
print("Initializing Background...", end='\r', flush=True)
bgInit(vd,bs)

def displayFrame(frame, text):
  cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
  cv2.putText(frame, text, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
  cv2.imshow('Hand Recognition', frame)

### Fetching Neural Network
### ================================================================================================================================
print("Fetching Neural Network... ", end='\r', flush=True)

PATH = 'gesture_net.pth'
net = Net()
net.load_state_dict(torch.load(PATH))
classes = ('Background', 'Next', 'Previous' , 'Stop')

print("Neural Network Fetched.       ")

### Start Camera
### ================================================================================================================================
ret,waitk = True, False
w = torchvision.transforms.ToTensor()
print("Capturing Frame ...", end='\r', flush=True)
while ret and not waitk :
  ret, im = vd.read()
  x = cv2.resize(im,(50,50))
  x = bs.apply(x)
  # x = cv2.cvtColor(x,cv2.COLOR_GRAY2BGR)
  x = w(x).unsqueeze(0)
  output = net(x)
  _, predicted = torch.max(output, 1)
  outClass = classes[predicted]
  displayFrame(im,outClass)
  waitk = cv2.waitKey(30)==27

print("Record Finished.      ")





