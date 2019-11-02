import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# class Net(nn.Module):
#   def __init__(self):
#     super(Net, self).__init__()
#     # 1 input image channel, 6 output channels, 3x3 square convolution
#     # kernel
#     self.conv1 = nn.Conv2d(1, 6, 3)
#     self.conv2 = nn.Conv2d(6, 16, 3)
#     # an affine operation: y = Wx + b
#     self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
#     self.fc2 = nn.Linear(120, 84)
#     self.fc3 = nn.Linear(84, 4)  ## 4 Output 

#   def forward(self, x):
#     x = F.max_pool2d(F.relu(self.conv1(x)), 2)
#     x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#     x = x.view(-1, self.num_flat_features(x))
#     x = F.relu(self.fc1(x))
#     x = F.relu(self.fc2(x))
#     x = self.fc3(x)
#     return x

#   def num_flat_features(self, x):
#     size = x.size()[1:]  # all dimensions except the batch dimension
#     num_features = 1
#     for s in size:
#         num_features *= s
#     return num_features

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 9 * 9, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 3)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 9 * 9)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

net = Net()