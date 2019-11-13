import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

# trainTransform  = transforms.Compose( [transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]) ])
# trainTransform  = transforms.Compose( [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]) ])
trainTransform  = transforms.Compose( [transforms.ToTensor(), transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]) ])
classes = ('Previous','Next','Stop' )

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 12, 3,padding = 0)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(12, 30, 3,padding = 1)
    self.pool = nn.MaxPool2d(2,2)
    self.conv3 = nn.Conv2d(30,40,3)
    self.pool = nn.MaxPool2d(2,2)
    self.fc1 = nn.Linear(40*5*5,100)
    self.fc2 = nn.Linear(100,3)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = x.view(-1, 40*5*5)
    x = F.relu(self.fc1(x))
    x = (self.fc2(x))
    # x = self.fc3(x)
    return x