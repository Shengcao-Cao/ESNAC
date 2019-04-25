import torch
import torch.nn as nn
import torch.nn.functional as F
from models.extension import *

class Layer(nn.Module):
    supported_base = (Identity, Flatten, Concat, Shuffle,
                      nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, nn.ReLU,
                      nn.BatchNorm2d, nn.Linear)

    def __init__(self, base, in_shape=None, out_shape=None):
        super(Layer, self).__init__()
        self.base = base
        self.base_type = base.__class__.__name__
        if not isinstance(base, Layer.supported_base):
            raise NotImplementedError('Unknown base layer!')
        self.in_shape = torch.Size([-1] + list(in_shape[1:]))
        self.out_shape = torch.Size([-1] + list(out_shape[1:]))
        self.init_rep()

    def replace(self, base):
        if not isinstance(base, Layer.supported_base):
            raise NotImplementedError('Unknown base layer!')
        self.base = base
        self.base_type = base.__class__.__name__
        self.init_rep()

    def shrink(self, Fi, Fo):
        in_shape = list(self.in_shape)
        in_shape[1] = Fi
        self.in_shape = torch.Size(in_shape)
        out_shape = list(self.out_shape)
        out_shape[1] = Fo
        self.out_shape = torch.Size(out_shape)

        b = self.base
        if isinstance(b, nn.Conv2d):
            groups = b.groups
            if (groups == b.in_channels and b.in_channels == b.out_channels and
                Fi == Fo):
                groups = Fi
            conv = nn.Conv2d(Fi, Fo, b.kernel_size, stride=b.stride,
                             padding=b.padding, dilation=b.dilation,
                             groups=groups, bias=(b.bias is not None))
            conv.weight = nn.Parameter(b.weight[:Fo, :(Fi // groups)].clone())
            if b.bias is not None:
                conv.bias = nn.Parameter(b.bias[:Fo].clone())
            self.replace(conv)
        elif isinstance(b, nn.BatchNorm2d):
            bn = nn.BatchNorm2d(Fi, eps=b.eps, momentum=b.momentum,
                                affine=b.affine,
                                track_running_stats=b.track_running_stats)
            bn.weight = nn.Parameter(b.weight[:Fi].clone())
            bn.bias = nn.Parameter(b.bias[:Fi].clone())
            self.replace(bn)
        elif isinstance(b, nn.Linear):
            ln = nn.Linear(Fi, Fo, bias=(b.bias is not None))
            ln.weight = nn.Parameter(b.weight[:Fo, :Fi].clone())
            if b.bias is not None:
                ln.bias = nn.Parameter(b.bias[:Fo].clone())
            self.replace(ln)
        else:
            self.init_rep()

    def forward(self, x):
        return self.base(x)

    def init_param(self):
        b = self.base
        if isinstance(b, nn.Conv2d):
            nn.init.kaiming_normal_(b.weight, mode='fan_out',
                                    nonlinearity='relu')
            if b.bias is not None:
                nn.init.constant_(b.bias, 0)
        elif isinstance(b, nn.BatchNorm2d):
            nn.init.constant_(b.weight, 1)
            nn.init.constant_(b.bias, 0)
        elif isinstance(b, nn.Linear):
            nn.init.normal_(b.weight, 0, 0.01)
            nn.init.constant_(b.bias, 0)

    def init_rep(self):
        b = self.base
        lt = Layer.supported_base.index(type(b))
        lr = [0] * 10
        lr[lt] = 1
        k = getattr(b, 'kernel_size', 0)
        k = k[0] if type(k) is tuple else k
        s = getattr(b, 'stride', 0)
        s = s[0] if type(s) is tuple else s
        p = getattr(b, 'padding', 0)
        p = p[0] if type(p) is tuple else p
        g = getattr(b, 'groups', 0)
        i = 0
        o = 0
        if isinstance(b, (nn.Conv2d, nn.Linear)):
            i = list(self.in_shape)[1]
            o = list(self.out_shape)[1]
        self.rep = lr + [k, s, p, g, i, o]

    def param_n(self):
        return sum([len(w.view(-1)) for w in self.base.parameters()])

class LayerGroup(object):

    def __init__(self, F, in_layers, out_layers):
        self.F = F
        self.in_layers = set(in_layers)
        self.out_layers = set(out_layers)
        self.union = self.in_layers | self.out_layers
        self.inter = self.in_layers & self.out_layers
        self.in_only = self.union - self.out_layers
        self.out_only = self.union - self.in_layers