import cv2
from CNN_Architecture import Net,trainTransform,classes
from Loader import loading
from DatasetPrepare import bgInit

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

### ================================================================================================================================
### Functions
### ================================================================================================================================
def getPermission():
  print("Train Data. [Y/n]. ")
  t = input()
  Train = (t=='y' or t=='Y')
  print("Initialize The Weights [Y/n].")
  i = input()
  Initialize = (i=='Y' or i=='y')
  return Train,Initialize

def getAccuracy(loader):
  correct = 0
  total = 0
  with torch.no_grad():
    for data in loader:
      print('Total :{0}, Correct: {1}  '.format(total,correct), end='\r', flush=True)
      images, labels = data
      outputs = net(images.cuda())
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels.cuda()).sum().item()
  return 100 * correct / total

def imshow(img):
  # img = img / 2 + 0.5
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

### ================================================================================================================================
### Load Training Dataset
### ================================================================================================================================

print("Loading Train Dataset")
PATH = 'gesture_net.pth'

torch.cuda.set_device(0)
print(torch.cuda.get_device_name())
Train,Initialize = getPermission()

train_dataset = torchvision.datasets.ImageFolder( root='Images_Demo_Train/', transform=trainTransform)
train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=32, num_workers=0, shuffle=True)
test_dataset = torchvision.datasets.ImageFolder( root='Images_Demo_Test/', transform=trainTransform)
test_loader = torch.utils.data.DataLoader( test_dataset, batch_size=32, num_workers=0, shuffle=True)

### ================================================================================================================================
### Initialize Loss, Optimizer and Neural Network
### Training Neural Network
### ================================================================================================================================

trainingLossList = []
crossValidationLossList = []
if Train:
  print("Starting CUDA neural ")
  net = Net()
  net.cuda()
  if not Initialize:
    net.load_state_dict(torch.load(PATH))
    net.eval()
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.8)
  print("Training Neural Network.")
  for epoch in range(8):  # loop over the dataset multiple times
    print("Running epoch: {0}".format(epoch))
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
      inputs, labels = data
      # ==== forward + backward + optimize ==== #
      optimizer.zero_grad()
      outputs = net(inputs.cuda())
      loss = criterion(outputs, labels.cuda())
      loss.backward()
      optimizer.step()
      # ==== print every 40 mini-batches ==== #
      running_loss += loss.item()
      if i % 40 == 39:   
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 40))
        running_loss = 0.0
    trainingLoss = getAccuracy(train_loader)
    crossValidationLoss = getAccuracy(test_loader)
    trainingLossList.append(trainingLoss)
    crossValidationLossList.append(crossValidationLoss)
    print("Epoch {0}: Training Accuracy: {1}, CrossValidation Accuracy: {2}".format(epoch,trainingLoss,crossValidationLoss))

  optimizer = torch.optim.SGD(net.parameters(), lr=0.0004, momentum=0.6)
  for epoch in range(6):  # loop over the dataset multiple times
    print("Running epoch: {0}".format(8+epoch))
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
      inputs, labels = data
      # ==== forward + backward + optimize ==== #
      optimizer.zero_grad()
      outputs = net(inputs.cuda())
      loss = criterion(outputs, labels.cuda())
      loss.backward()
      optimizer.step()
      # ==== print every 40 mini-batches ==== #
      running_loss += loss.item()
      if i % 40 == 39:   
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 40))
        running_loss = 0.0
    trainingLoss, crossValidationLoss = getAccuracy(train_loader),getAccuracy(test_loader)
    trainingLossList.append(trainingLoss)
    crossValidationLossList.append(crossValidationLoss)
    print("Epoch {0}: Training Accuracy: {1}, CrossValidation Accuracy: {2}".format(epoch,trainingLoss,crossValidationLoss))

  optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.2)
  for epoch in range(6):  # loop over the dataset multiple times
    print("Running epoch: {0}".format(8+epoch))
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
      inputs, labels = data
      # ==== forward + backward + optimize ==== #
      optimizer.zero_grad()
      outputs = net(inputs.cuda())
      loss = criterion(outputs, labels.cuda())
      loss.backward()
      optimizer.step()
      # ==== print every 40 mini-batches ==== #
      running_loss += loss.item()
      if i % 40 == 39:   
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 40))
        running_loss = 0.0
    trainingLoss, crossValidationLoss = getAccuracy(train_loader),getAccuracy(test_loader)
    trainingLossList.append(trainingLoss)
    crossValidationLossList.append(crossValidationLoss)
    print("Epoch {0}: Training Accuracy: {1}, CrossValidation Accuracy: {2}".format(epoch,trainingLoss,crossValidationLoss))

  print('Finished Training. \n Saving The Model...')
  plt.plot(trainingLossList)
  plt.plot(crossValidationLossList)
  plt.show()
  torch.save(net.state_dict(), PATH)

### ================================================================================================================================
### Test Neural Network
### ================================================================================================================================

print("Proceeding to Test Neural Network.\t\t")

net = Net()
net.cuda()
net.load_state_dict(torch.load(PATH))

dataiter = iter(test_loader)
images, labels = dataiter.next()
outputs = net(images.cuda())
_, predicted = torch.max(outputs, 1)

imshow(torchvision.utils.make_grid(images[0:40]))
total = labels.size(0)
correct = (predicted == labels.cuda()).sum().item()
print("Total :{0}, Correct: {1}".format(total,correct))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(40)))
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(40)))

### ================================================================================================================================
### Determine Accuracy
### ================================================================================================================================

print("Determing Accuracy on DataLoader ... \t\t\n")
print("Calculating Accuracy of the Train Neural Network..\t")
print('Training Accuracy\t: %d %%' % getAccuracy(train_loader))
print("Calculating Accuracy of the Cross ValidationNeural Network..\t")
print('Cross Validation Accuracy: %d %%' % getAccuracy(test_loader))