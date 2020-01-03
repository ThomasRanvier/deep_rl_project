import math
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
        self._init_weights(self.conv_1.weight)

        # Initialize Torch variable of the second conv layer
        self.conv_2 = torch.nn.Conv2d(in_channels=out_1, out_channels=out_1*2, kernel_size=4, stride=2, padding=0)
        self._init_weights(self.conv_2.weight)

        if heavy_model:
            # Initialize Torch variable of the third conv layer, add a padding of 1 to keep same dimensions
            self.conv_3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
            self._init_weights(self.conv_3.weight)

        # Initialize Torch variable of the adv hidden layer
        self.h_adv = torch.nn.Linear(9 * 9 * out_1 * 2, fc_hidden)
        self._init_weights(self.h_adv.weight)

        # Initialize Torch variable of the v hidden layer
        self.h_v = torch.nn.Linear(9 * 9 * out_1 * 2, fc_hidden)
        self._init_weights(self.h_v.weight)

        # Initialize Torch variable of the adv layer
        self.adv = torch.nn.Linear(fc_hidden, N_ACTIONS)
        self._init_weights(self.adv.weight)

        # Initialize Torch variable of the v layer
        self.v = torch.nn.Linear(fc_hidden, 1)
        self._init_weights(self.v.weight)

    def _init_weights(self, tensor):
        """
        In the paper : Weights init: zero-mean Gaussian distribution whose standard deviation (std) is sqrt(2/nl).
        From https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb :
        In tensorflow use tf.variance_scaling_initializer with scale = 2
        This function replicates the initialization from tensorflow

        Before this function we initialized the weights in this way : nn.init.xavier_normal_(self.conv_1.weight, gain=2)

        :param tensor: The weight tensor to initialize
        """
        # Compute fan in
        # code from https://github.com/pytorch/pytorch/blob/cb1af5f61fb338c591e6427fd274ea5b44df4f26/torch/nn/init.py#L202
        dimensions = tensor.dim()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
        num_input_fmaps = tensor.size(1)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        # scale = 2
        std = math.sqrt(2. / fan_in)
        # mean = 0
        with torch.no_grad():
            tensor.normal_(0., std)

    def forward(self, x):
        # conv, relu, max pool, conv, relu, max pool, fc, relu, dropout, fc
        # x : 84x84x4
        x = F.relu(self.conv_1(x))
        # x : 20x20x32
        x = F.relu(self.conv_2(x))
        # x : 9x9x64
        if self._heavy_model:
            x = F.relu(self.conv_3(x))
            # x : 9x9x64
        x = x.reshape(x.size(0), -1)
        adv = self.adv(F.relu(self.h_adv(x)))
        v = self.v(F.relu(self.h_v(x)))
        q = v + (adv - torch.mean(adv, dim=1, keepdim=True))
        # https://github.com/Kaixhin/Rainbow/blob/master/model.py
        # q = F.log_softmax(q, dim=0)
        return q
