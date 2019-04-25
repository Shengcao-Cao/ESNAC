import torch
import torch.nn as nn
import torch.nn.functional as F
from .extension import *

class BottleneckM(nn.Module):

    def __init__(self, in_planes, out_planes, stride, groups):
        super(BottleneckM, self).__init__()
        self.stride = stride
        mid_planes = out_planes // 4
        g = 1 if in_planes == 24 else groups

        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.relu1 = nn.ReLU()
        self.shuffle = Shuffle(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3,
                               stride=stride, padding=1, groups=mid_planes,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1,
                               groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu3 = nn.ReLU()

        if stride == 2:
            self.conv4 = nn.Conv2d(in_planes, in_planes, kernel_size=1,
                                   groups=2, bias=False)
            self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
            self.concat = Concat(dim=1)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.shuffle(out)
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))

        if self.stride == 2:
            res = self.avgpool(self.conv4(x))
            out = self.relu3(self.concat([out, res]))
        else:
            res = x
            out = self.relu3(out + res)
        return out

class ShuffleNetM(nn.Module):

    def __init__(self, cfg, num_classes=100):
        super(ShuffleNetM, self).__init__()
        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']

        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.relu1 = nn.ReLU()
        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        self.avgpool = nn.AvgPool2d(4)
        self.flatten = Flatten()
        self.fc = nn.Linear(out_planes[2], num_classes)

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(BottleneckM(self.in_planes, out_planes - cat_planes,
                                      stride=stride, groups=groups))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.fc(self.flatten(self.avgpool(x)))
        return x

def shufflenet(**kwargs):
    cfg = {
        'out_planes': [200, 400, 800],
        'num_blocks': [4, 8, 4],
        'groups': 2
    }
    return ShuffleNetM(cfg, **kwargs)