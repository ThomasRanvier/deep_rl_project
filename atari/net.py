import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, f_s):
        super(Net, self).__init__()
        self.layers = []
        for i in range(len(f_s) - 1):
            exec('self.l_' + str(i) + ' = nn.Linear(f_s[' + str(i) + '], f_s[' + str(i + 1) + '])')
            exec('self.layers.append(self.l_' + str(i) + ')')

    def forward(self, x):
        for l in self.layers[:-1]:
            x = F.relu(l(x))
        x = self.layers[-1](x)
        return x