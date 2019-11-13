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

torch.cuda.set_device(0)

### ================================================================================================================================
### Load Training Dataset
### ================================================================================================================================

classes = ('Background', 'Next', 'Previous' , 'Stop')
PATH = 'gesture_net.pth'
# trainTransform  = transforms.Compose( [transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]) ])
trainTransform  = transforms.Compose( [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]) ])

train_dataset = torchvision.datasets.ImageFolder( root='Images_RecordBS_Train/', transform=trainTransform)
train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=64, num_workers=0, shuffle=True)

net = Net()
net.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

### ================================================================================================================================
### Initialize Loss, Optimizer and Neural Network
### Training Neural Network
### ================================================================================================================================

epochLoss = []
if Train:
  print("Training Neural Network.")
  for epoch in range(12):  # loop over the dataset multiple times
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

      # ==== print every 20 mini-batches ==== #
      running_loss += loss.item()
      if i % 20 == 19:   
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 20))
        epochLoss.append(running_loss)
        running_loss = 0.0

  print('Finished Training. \n Saving The Model...')
  plt.plot(epochLoss)
  plt.show
  torch.save(net.state_dict(), PATH)

### ================================================================================================================================
### Test Neural Network
### ================================================================================================================================

print("Proceeding to Test Neural Network.\t\t")

def imshow(img):
  img = img / 2 + 0.5
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

net = Net()
net.cuda()
net.load_state_dict(torch.load(PATH))

dataiter = iter(train_loader)
images, labels = dataiter.next()
outputs = net(images.cuda())
_, predicted = torch.max(outputs, 1)

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(10)))
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(10)))
# imshow(torchvision.utils.make_grid(images[0:10]))

### ================================================================================================================================
### Determine Accuracy
### ================================================================================================================================

test_dataset = torchvision.datasets.ImageFolder( root='Images_RecordBS_Test/', transform=trainTransform)
test_loader = torch.utils.data.DataLoader( test_dataset, batch_size=64, num_workers=0, shuffle=True)

print("Calculating Accuracy of the Neural Network..\t")
correct = 0
total = 0
with torch.no_grad():
  for data in test_loader:
    print('Total :{0}, Correct: {1}  '.format(total,correct), end='\r', flush=True)
    images, labels = data
    outputs = net(images.cuda())
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum().item()
print('Accuracy of the network on the Test images: %d %%' % (100 * correct / total))