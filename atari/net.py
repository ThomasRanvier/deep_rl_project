import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Initialize Torch variable of the first conv layer
        self.conv_1 = torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=0)
        # Initialize Torch variable of the second conv layer
        self.conv_2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        # Initialize Torch variable of the third conv layer, add a padding of 1 to keep same dimensions
        self.conv_3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Initialize Torch variable of the two linear layers
        self.linear_1 = torch.nn.Linear(9 * 9 * 64, 512)
        self.linear_2 = torch.nn.Linear(512, N_ACTIONS)
        # Add a dropout of 50%
        # Used to dropout 50% of the neurons of the fc layer 1
        self.dropout = torch.nn.Dropout(p=0.5)
        # Define the activation function

    def forward(self, x):
        # conv, relu, max pool, conv, relu, max pool, fc, relu, dropout, fc
        # x : 84x84x4
        x = self.conv_1(x)
        x = F.relu(x)
        # x : 20x20x32
        x = self.conv_2(x)
        x = F.relu(x)
        # x : 9x9x64
        x = self.conv_3(x)
        x = F.relu(x)
        # x : 9x9x64
        x = x.reshape(x.size(0), -1)
        x = self.linear_1(x)
        x = F.relu(x)
        #x = self.dropout(x)
        x = self.linear_2(x)
        return x
