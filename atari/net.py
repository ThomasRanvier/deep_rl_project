import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class Net(nn.Module):
    def __init__(self, heavy_model = False):
        super(Net, self).__init__()
        self._heavy_model = heavy_model
        out_1 = 16 * (heavy_model + 1)
        fc_hidden = 256 * (heavy_model + 1)
        # Initialize Torch variable of the first conv layer
        self.conv_1 = torch.nn.Conv2d(in_channels=4, out_channels=out_1, kernel_size=8, stride=4, padding=0)
        # Weights init: zero-mean Gaussian distribution whose standard deviation (std) is sqrt(2/nl). This is our way of initialization.
        # Xavier normal seams to be close enough, with this the loss is high enough to be properly minimized
        # With the default weights init the loss was always extremely low
        nn.init.xavier_normal_(self.conv_1.weight)
        # Initialize Torch variable of the second conv layer
        self.conv_2 = torch.nn.Conv2d(in_channels=out_1, out_channels=out_1*2, kernel_size=4, stride=2, padding=0)
        nn.init.xavier_normal_(self.conv_2.weight)
        if heavy_model:
            # Initialize Torch variable of the third conv layer, add a padding of 1 to keep same dimensions
            self.conv_3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
            nn.init.xavier_normal_(self.conv_3.weight)
        # Initialize Torch variable of the two linear layers
        self.linear_1 = torch.nn.Linear(9 * 9 * out_1 * 2, fc_hidden)
        nn.init.xavier_normal_(self.linear_1.weight)
        self.linear_2 = torch.nn.Linear(fc_hidden, N_ACTIONS)
        nn.init.xavier_normal_(self.linear_2.weight)

    def forward(self, x):
        # conv, relu, max pool, conv, relu, max pool, fc, relu, dropout, fc
        # x : 84x84x4
        x = self.conv_1(x)
        x = F.relu(x)
        # x : 20x20x32
        x = self.conv_2(x)
        x = F.relu(x)
        # x : 9x9x64
        if self._heavy_model:
            x = self.conv_3(x)
            x = F.relu(x)
            # x : 9x9x64
        x = x.reshape(x.size(0), -1)
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)
        return x
