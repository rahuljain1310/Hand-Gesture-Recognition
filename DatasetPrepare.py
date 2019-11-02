import cv2

def RecordVideo(vname,sqSize, totalFrame):
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

def getImagesfromVideos(inputV, outputf, total, part):
  vdi = cv2.VideoCapture('Videos/{0}.mp4'.format(inputV))
  ret,fr = vdi.read()
  i = 0
  while i<total:
    print('\n{0}'.format(i), end='\r',flush=True)
    i+=1
    cv2.imwrite('Images/{0}/{1}_{2}.jpg'.format(outputf,part,i),fr)
    ret,fr = vdi.read()

def RecordAndImages(output,total, part):
  inputV = "{0}{1}".format(output,part)
  RecordVideo(inputV,50,2000)
  getImagesfromVideos(inputV,output,total, part)
