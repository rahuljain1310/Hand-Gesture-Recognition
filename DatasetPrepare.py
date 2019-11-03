import cv2
import math
from Loader import wait

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
  for _ in range(200):
    _, im = vd.read()
    im = cv2.resize(im,(50,50))
    fim = bs.apply(im)
    cv2.imshow('Background Initialization',cv2.resize(fim,(640,480)))
    cv2.waitKey(30)
  cv2.destroyWindow('Background Initialization')
  print("Background Initialized.                  ")

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

def RecordClassImages(outputf, totalFrame, totalParts):
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
  
if __name__ == "__main__":
  RecordClassImages('Images_RecordBS/Background',1000,2)
