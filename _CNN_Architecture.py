import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

classes = ('Previous','Next','Stop','Background')
trainTransform  = transforms.Compose( [transforms.RandomCrop(50, padding=5), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]) ])
# trainTransform  = transforms.Compose( [transforms.ToTensor(), transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]) ])

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 12, 3,padding = 0)
    self.layer1 = nn.Sequential( nn.Conv2d(1, 30, kernel_size=3, stride=1, padding=0), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer2 = nn.Sequential( nn.Conv2d(30, 25, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer3 = nn.Sequential( nn.Conv2d(25, 20, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer4 = nn.Sequential( nn.Conv2d(20, 10, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
    self.drop_out = nn.Dropout()
    self.fc1 = nn.Linear(720,50)
    self.fc2 = nn.Linear(50,10)
    self.fc3 = nn.Linear(10,4)

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = x.reshape(x.size(0), -1)
    x = self.drop_out(x)
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return x