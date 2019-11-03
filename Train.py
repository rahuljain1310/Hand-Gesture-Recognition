import cv2
from CNN_Architecture import Net
from Loader import loading
from DatasetPrepare import bgInit

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


Train = True

### ================================================================================================================================
### Load Training Dataset
### ================================================================================================================================

classes = ('Background', 'Next', 'Previous' , 'Stop')
data_path = 'Images_RecordBS/'
PATH = 'gesture_net.pth'
trainTransform  = transforms.Compose( [transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]) ])
train_dataset = torchvision.datasets.ImageFolder( root=data_path, transform=trainTransform)
train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=64, num_workers=0, shuffle=True)

net = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

### ================================================================================================================================
### Initialize Loss, Optimizer and Neural Network
### Training Neural Network
### ================================================================================================================================

if Train:
  for epoch in range(8):  # loop over the dataset multiple times
    print("Running epoch: {0}".format(epoch))
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
      inputs, labels = data
      # ==== forward + backward + optimize ==== #
      optimizer.zero_grad()
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # ==== print every 20 mini-batches ==== #
      running_loss += loss.item()
      if i % 20 == 19:   
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 20))
        running_loss = 0.0

  print('Finished Training. \n Saving The Model...')
  torch.save(net.state_dict(), PATH)

### ================================================================================================================================
### Test Neural Network
### ================================================================================================================================

def imshow(img):
  img = img / 2 + 0.5
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

net = Net()
net.load_state_dict(torch.load(PATH))

dataiter = iter(train_loader)
images, labels = dataiter.next()
outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(10)))
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(10)))
imshow(torchvision.utils.make_grid(images[0:10]))

### ================================================================================================================================
### Determine Accuracy
### ================================================================================================================================

print("Calculating Accuracy of the Neural Network..")
correct = 0
total = 0
with torch.no_grad():
  for data in train_loader:
    print('Total :{0}, Correct: {1}  '.format(total,correct), end='/r', flush=True)
    images, labels = data
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 12000 test images: %d %%' % (100 * correct / total))