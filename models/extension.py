import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Concat(nn.Module):

    def __init__(self, dim):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)

class Shuffle(nn.Module):

    def __init__(self, groups):
        super(Shuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        N, C, H, W = x.size()
        g = self.groups
        x = x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(N, C, H, W)
        return x