import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Initialize Torch variable of the first conv layer
        self.conv_1 = torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4, padding=0)
        # Initialize Torch variable of the second conv layer
        self.conv_2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0)
        # Initialize Torch variable of the two linear layers
        self.linear_1 = torch.nn.Linear(9 * 9 * 32, 256)
        self.linear_2 = torch.nn.Linear(256, N_ACTIONS)
        # Add a dropout of 50%
        # Used to dropout 50% of the neurons of the fc layer 1
        self.dropout = torch.nn.Dropout(p=0.5)
        # Define the activation function

    def forward(self, x):
        # conv, relu, max pool, conv, relu, max pool, fc, relu, dropout, fc
        # x : 84x84x4
        x = self.conv_1(x)
        x = F.relu(x)
        # x : 20x20x16
        x = self.conv_2(x)
        x = F.relu(x)
        # x : 9x9x32
        x = x.reshape(x.size(0), -1)
        x = self.linear_1(x)
        x = F.relu(x)
        #x = self.dropout(x)
        x = self.linear_2(x)
        return x
