import cv2
from _CNN_Architecture import Net, trainTransform,classes

from _Loader import loading
from _DatasetPrepare import bgInit
import skin_detect
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


### Initialization of Camera / Display Functions 
### ================================================================================================================================

def getRectangle(sqSize,w,h):
  mw = (w-sqSize)//2
  mh = (h-sqSize)//2
  p1 = (mw, mh)
  p2 = (mw+sqSize, mh+sqSize)
  return p1,p2

vd = cv2.VideoCapture(0)
ret,im = vd.read()
h,w,_ = im.shape
boundSquare = 240
p1,p2 = getRectangle(boundSquare,w,h)

def displayFrame(frame, text):
  cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
  cv2.rectangle(frame,p1,p2,(0,0,255), thickness=2)
  cv2.putText(frame, text, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
  cv2.imshow('Hand Recognition', frame)

### Transformation Functions and Background Substraction 
### ================================================================================================================================

# bs = cv2.createBackgroundSubtractorKNN(history=4000)
# print("Initializing Background...", end='\r', flush=True)
# bgInit(vd,bs)

def transformImage(im):
  # x,y = p1[0],p1[1]
  # img = im[y:y+boundSquare,x:x+boundSquare]
  # cv2.imshow('Sample',img)
  x = cv2.resize(im,(50,50))
  return x

### Fetching Neural Network
### ================================================================================================================================
print("Fetching Neural Network... ", end='\r', flush=True)

PATH = 'gesture_net.pth'
net = Net()
net.load_state_dict(torch.load(PATH))

print("Neural Network Fetched.\t\t\t")

### Video Demonstraion
### ================================================================================================================================

ret,waitk = True, False
w = torchvision.transforms.ToTensor()
print("Capturing Frame ...\t\t", end='\r', flush=True)

bs = cv2.createBackgroundSubtractorMOG2()
bgInit(vd,bs)

while ret and not waitk :
  ret, im = vd.read()
  points,frame = skin_detect.track_using_background(bs,im)
  print(1)
  outputs = []
  predictions = []
  outclasses = []
  if len(points)==0:
    cv2.putText(frame, 'Background', (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,255,0))
    continue
  i_max = -1
  p_max = -1
  label_max = -1
  i = -1
  for p1, p2 in points:
    i = i+1
    try:
      tensorIm = trainTransform(transformImage(im[p1[1]:p2[1],p1[0]:p2[0]]))
      output = net(tensorIm.unsqueeze(0))
      outputs.append(output)
      value, predicted = torch.max(output, 1)
      if value>1:
        p_max = value
        i_max = i
        label_max = predicted
        cv2.putText(frame, classes[label_max],tuple(points[i_max][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        cv2.rectangle(frame, tuple(points[i_max][0]), tuple(points[i_max][1]), (0, 255, 0), 2)
      # predictions.append(predicted)
      # outClass = classes[predicted]
      # outclasses.append(outClass)
      # cv2.putText(frame, outClass,tuple(p1), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    except:
      cv2.putText(frame, 'Background', (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,255,0))
      continue
  if p_max>1.5:
    pass
    # cv2.putText(frame, classes[label_max],tuple(points[i_max][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    # cv2.rectangle(frame, tuple(points[i_max][0]), tuple(points[i_max][1]), (0, 255, 0), 2)
  else:
    cv2.putText(frame, 'Background', (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,255,0))
  cv2.imshow('Detect',frame)
  waitk = cv2.waitKey(30)==27