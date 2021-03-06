import cv2
import math
from Loader import wait
import sys
import os

# if not sys.warnoptions:
#     import warnings
#     warnings.simplefilter("ignore")

sqSize = 50

def showAndget(vd):
  ret,im = vd.read()
  if ret:
    cv2.imshow('Video',im)
    cv2.waitKey(20)
    return ret,cv2.resize(im,(50,50))
  else:
    return ret,im

def Partpermit():
  print("Press Y/n to Continue To Next Part of Recording.. ")
  x = input()
  if x=='y' or x=='Y':
    return True
  else:
    return False

def bgInit(vd,bs):
  wait(2)
  print("Initializing Background ...", end='\r', flush=True)
  for _ in range(100):
    _, im = vd.read()
    im = cv2.resize(im,(50,50))
    fim = bs.apply(im)
    cv2.imshow('Background Initialization',cv2.resize(fim,(640,480)))
    cv2.waitKey(30)
  cv2.destroyWindow('Background Initialization')
  print("Background Initialized.                  ")

def getRectangle(sqSize,w,h):
  mw = (w-sqSize)//2
  mh = (h-sqSize)//2
  p1 = (mw, mh)
  p2 = (mw+sqSize, mh+sqSize)
  return p1,p2

def getImageWithBox(vd, sqSize, p1, p2 ,ws, text):
  ret,im = vd.read()
  x,y = p1[0], p1[1]
  if ret:
    img = im[y:y+sqSize,x:x+sqSize]
    cv2.rectangle(im,p1,p2,(0,0,255), thickness=2)
    cv2.imshow('Capture Video',cv2.putText(cv2.flip(im,1), text, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (255,255,255)))
    cv2.waitKey(ws)
    return ret, img
  else:
    return ret,im

def RecordVideo(vname, totalFrame):
  vd = cv2.VideoCapture(0)
  _,im = vd.read()
  h,w,_ = im.shape
  e = (w-h)//2
  vdw = cv2.VideoWriter('Videos/{0}.mp4'.format(vname),-1,30,(sqSize,sqSize))
  for i in range(-100,totalFrame):
    ret,im = vd.read()
    im = cv2.resize(im,(w,h))
    img = im[:,e:h+e,:]
    cv2.imshow('Video',img)
    cv2.waitKey(20)
    if i>0:
      print("Receiving Frame {0}".format(i), end='\r',flush=True)
      img = cv2.resize(img,(sqSize,sqSize))
      vdw.write(img)
  vdw.release()

def ConcatenateVideos(output, in1, in2, outputShape=None):
  vd1 = cv2.VideoCapture('Videos/{0}.mp4'.format(in1))
  vd2 = cv2.VideoCapture('Videos/{0}.mp4'.format(in2))

  ret1,fr1 = vd1.read()
  h,w,_= fr1.shape

  shape = (w,h)

  vdw = cv2.VideoWriter('Videos/{0}.mp4'.format(output),-1,30,shape)
  i = 0
  while ret1:
    print(i, end='\r',flush=True)
    i += 1 
    vdw.write(fr1)
    ret1,fr1 = vd1.read()
  ret2,fr2 = vd2.read()
  while ret2:
    print(i, end='\r',flush=True)
    i+=1
    resizeWriteFrame(vdw,fr2, shape)
    vsw.write(fr2)
    ret2,fr2 = vd2.read()
  vdw.release()

def resizeWriteFrame(video,img,shape):
	img = cv2.resize(img,shape)
	video.write(img)

def ResizeVideo(output,inputV,shape):
  vdi = cv2.VideoCapture('Videos/{0}.mp4'.format(inputV))
  vdw = cv2.VideoWriter('Videos/{0}.mp4'.format(output),-1,30,shape)
  ret,fr = vdi.read()
  i = 0
  while ret:
    print(i, end='\r',flush=True)
    i+=1
    resizeWriteFrame(vdw, fr, shape)
    ret,fr = vdi.read()

def getImagesfromVideos(inputV, outputf, part, total=math.inf):
  vdi = cv2.VideoCapture(inputV)
  ret,fr = vdi.read()
  i = 0
  while ret and i<total:
    print('Images Collected: {0}'.format(i), end='\r',flush=True)
    i+=1
    cv2.imwrite('{0}/{1}_{2}.jpg'.format(outputf,part,i), cv2.resize(fr,(50,50)) )
    ret,fr = vdi.read()
  print("Images Collected from {0} in folder{1}".format(inputV,outputf))

def getBackgroundSubsfromVideos(inputV, outputf, part, total=math.inf):
  backSub = cv2.createBackgroundSubtractorMOG2()
  vdi = cv2.VideoCapture(inputV)
  ret,fr = vdi.read()
  _ = backSub.apply(fr)
  ret,fr = vdi.read()
  fr = backSub.apply(fr)
  i = 0
  while ret and i<total:
    print('Images Collected: {0}'.format(i), end='\r',flush=True)
    i+=1
    cv2.imwrite('{0}/{1}_{2}.jpg'.format(outputf,part,i), cv2.resize(fr,(50,50)) )
    ret,fr = vdi.read()
    fr = backSub.apply(fr)
  print("Images Collected from {0} in folder{1}".format(inputV,outputf))

def RecordClassImagesWithBS(outputf, totalFrame, totalParts):
  vd = cv2.VideoCapture(0)
  bs = cv2.createBackgroundSubtractorKNN(history=4000)

  print("Recording Background For Initilization of background Substractor...")
  bgInit(vd,bs)
  cv2.destroyAllWindows()

  framePerPart = int(totalFrame/totalParts)
  for part in range(totalParts):
    print("Recording Part:{0} in 10 seconds..".format(part), end='\r', flush=True)
    wait(10)
    for i in range(framePerPart):
      print("Recording Frame {0}".format(i),end='\r',flush=True)
      ret,im = showAndget(vd)
      fim = bs.apply(im)
      cv2.imwrite('{0}/{1}_{2}.jpg'.format(outputf,part,i), fim)
    print("Part {0} Recorded.                 ".format(part))
  
def RecordClassImages(outputf, framesPerPart, part):
  print("Continue to Part {0} of {1}[Y/n]: ".format(part,outputf), end='', flush=True)
  vd = cv2.VideoCapture(0)
  _,im = vd.read()
  h,w,_ = im.shape
  boundSquare = 480
  sec = 2
  fs = 30
  p1,p2 = getRectangle(boundSquare,w,h)
  # res = input()
  # if (not res == 'y') and (not res =='Y'):
  #   return
  print("Recording Part:{0} in 10 seconds..".format(part), end='\r', flush=True)
  for i in range(sec*23):
    getImageWithBox(vd,360,p1,p2,30,'Wait {0} seconds.'.format(int(sec-i/23)))
  for i in range(framesPerPart):
    print("Recording Frame {0}\t\t\t\t".format(i),end='\r',flush=True)
    ret,im = getImageWithBox(vd,boundSquare,p1,p2,1000//fs,"Recording Frame {0}".format(i))
    if ret:
      im = cv2.resize(im,(50,50))
      cv2.imwrite('{0}/{1}_{2}.jpg'.format(outputf,part,i), im)
  print("Part {0} Recorded.\t\t".format(part))
  vd.release()
  cv2.destroyAllWindows()
  
if __name__ == "__main__":
  x = ['Images_TM/'+c for c in os.listdir('Images_TM')]  
  for d in x:
    y = os.listdir(d)
    for img in y:
      img_ = cv2.imread(d+'/'+img)
      img_ = cv2.resize(img_,(50,50))
      cv2.imwrite(d+'/'+img,img_)                                                                                                  

  # part = input()
  # frame = 100
  # RecordClassImages('Images_Demo/Stop',frame,part+'_1')
  # RecordClassImages('Images_Demo/Next',frame//2,part+'_1')
  # RecordClassImages('Images_Demo/Previous',frame//2,part+'_1')
  # RecordClassImages('Images_Demo/Stop',frame,part+'_2')
  # RecordClassImages('Images_Demo/Next',frame//2,part+'_2')
  # RecordClassImages('Images_Demo/Previous',frame//2,part+'_2')
  # RecordClassImages('Images_Demo_Train/Background',2000,part+'_1')
  # RecordClassImages('Images_Demo_Test/Background',200,part+'_1')
  # RecordClassImages('Images_Demo_CV/Background',400,part+'_1')
