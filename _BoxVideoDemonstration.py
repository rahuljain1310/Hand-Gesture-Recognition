import cv2
from _CNN_Architecture import Net
from _CNN_Architecture import Net, trainTransform,classes

from _Loader import loading
from _DatasetPrepare import bgInit
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import PIL.Image as Image

### Initialization of Camera / Display Functions 
### ================================================================================================================================

def getRectangle(sqSize,w,h):
  mw = (w-sqSize)//2
  mh = (h-sqSize)//2
  p1 = (mw, mh)
  p2 = (mw+sqSize, mh+sqSize)
  return p1,p2

def displayFrame(frame, text):
  cv2.rectangle(frame, (10, 2), (120,20), (255,255,255), -1)
  cv2.rectangle(im,p1,p2,(0,0,255), thickness=2)
  cv2.rectangle(frame,p1,p2,(0,0,255), thickness=2)
  cv2.putText(frame, text, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
  cv2.imshow('Hand Recognition', frame)

def transformImage(im):
  x,y = p1[0],p1[1]
  img = im[y:y+boundSquare,x:x+boundSquare]
  x = cv2.resize(img,(50,50))
  # y = bs.apply(x)
  # y = cv2.morphologyEx(y, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
  # x = cv2.bitwise_and(x,x,mask=y)
  # cv2.imshow('Test',x)
  x = Image.fromarray(x)
  return x

vd = cv2.VideoCapture(0)
ret,im = vd.read()
h,w,_ = im.shape
boundSquare = 320
p1,p2 = getRectangle(boundSquare,w,h)

### Fetching Neural Network
### ================================================================================================================================

PATH = 'gesture_net.pth'
net = Net()
net.cuda()
net.load_state_dict(torch.load(PATH))
sm = torch.nn.Softmax(dim=1)
print("Neural Network Fetched.\t\t\t")

### Transformation Functions and Background Substraction 
### ================================================================================================================================

# bs = cv2.createBackgroundSubtractorKNN(history=4000)
# print("Initializing Background...", end='\r', flush=True)
# bgInit(vd,bs)

ret,waitk = True, False
w = torchvision.transforms.ToTensor()

while ret and not waitk :
  ret, im = vd.read()
  tensorIm = trainTransform(transformImage(im))
  output = net(tensorIm.unsqueeze(0).cuda())
  value, predicted = torch.max(output.cuda(), 1)
  x = torch.Tensor.cpu(sm(output))
  probabilities = x.detach().numpy()
  prob = probabilities[0][predicted]
  if value.double() > 2 and prob > 0.6:
    outClass = classes[predicted]
  else:
    outClass = classes[3]
  displayFrame(im,outClass+' {0:.2f} %'.format(prob*100))
  waitk = cv2.waitKey(30)==27 