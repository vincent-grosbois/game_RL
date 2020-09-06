import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN1(nn.Module):

    def __init__(self, n_channels, outputs):
        super(DQN1, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.head = nn.Linear(32, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        #print(x)
        x = F.relu(self.bn1(self.conv1(x)))
        #print(x)
        x = F.relu(self.conv2(x))
        #print(x)
        x = self.head(x.view(x.size(0), -1))
        #print(x)
        return x