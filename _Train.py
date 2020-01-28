import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
# from sklearn.metrics import confusion_matrix

from _CNN_Architecture import Net, trainTransform, classes

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
  Accuracy = True
  return Train,Initialize,Accuracy

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

def SplitArray(Samples,train):
  train_size = int(train*Samples)
  cross_size = Samples - train_size
  return [train_size, cross_size]

def getConfusionMatrix(size, loader):
  cf = torch.zeros(3,3)
  with torch.no_grad():
    for data in test_loader:
      images, labels = data
      outputs = net(images.cuda())
      _, predicted = torch.max(outputs.data, 1)
      for t, p in zip(labels, predicted):
        cf[t, p] += 1
  return cf

### ================================================================================================================================
### Load Training Dataset
### ================================================================================================================================

PATH = 'gesture_net.pth'
torch.cuda.set_device(0)

print("Loading Train Dataset")
print(torch.cuda.get_device_name())

dataset = torchvision.datasets.ImageFolder( root='Images_Demo/', transform=trainTransform)
Samples = len(dataset)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, SplitArray(Samples,0.8))

train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=64, num_workers=0, shuffle=True)
test_loader = torch.utils.data.DataLoader( test_dataset, batch_size=64, num_workers=0, shuffle=True)

### ================================================================================================================================
### Initialize Loss, Optimizer and Neural Network
### Training Neural Network
### ================================================================================================================================

trainingLossList = []
crossValidationLossList = []

print("Starting CUDA neural ")
net = Net()
net.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.8)

print("Training Neural Network.")
for epoch in range(0,55):
  print("Running epoch: {0}".format(epoch))
  if epoch==40: optimizer = torch.optim.SGD(net.parameters(), lr=0.0005, momentum=0.7)
  if epoch==50: optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.6)
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

### ================================================================================================================================
### Saving and Plotting Model
### ================================================================================================================================

print('Finished Training.\nSaving The Model...')
torch.save(net.state_dict(), PATH)
plt.plot(trainingLossList)
plt.plot(crossValidationLossList)
plt.show()

### ================================================================================================================================
### Determine Accuracy
### ================================================================================================================================

print("\nDeterming Accuracy on DataLoader ... \t\t")

net = Net()
net.cuda()
net.load_state_dict(torch.load(PATH))
net.eval()

print("Calculating Accuracy of the Train Neural Network..\t")
print('Training Accuracy\t: %d %%' % getAccuracy(train_loader))
print("Calculating Accuracy of the Cross ValidationNeural Network..\t")
print('Cross Validation Accuracy: %d %%' % getAccuracy(test_loader))

### ================================================================================================================================
### Test Neural Network
### ================================================================================================================================

print("Confusion Matrix \t\t")
print(classes)
print(getConfusionMatrix(3,test_loader))