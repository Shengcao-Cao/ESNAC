import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models
import graph as gr
import layer
import options as opt
import copy
import os
import random
from math import ceil

class Architecture(nn.Module):

    def __init__(self, n, V, E):
        super(Architecture, self).__init__()
        self.n = n
        self.V = V
        for i in range(n):
            self.add_module('layer_%d' % (i), V[i])
        self.groups = gr.get_groups(V)
        self.E = E
        self.in_links, self.out_links = gr.get_links(E)
        self.init_rep()

    def get_layer(self, i):
        return getattr(self, 'layer_%d' % (i))

    def forward(self, x):
        y = [None] * self.n
        y[0] = self.get_layer(0)(x)
        for j in range(1, self.n):
            x = []
            for i in self.in_links[j]:
                x.append(y[i])
                if j == self.out_links[i][-1]:
                    y[i] = None
            if not x:
                y[j] = None
            else:
                layer = self.get_layer(j)
                if isinstance(layer.base, models.Concat):
                    y[j] = layer(x)
                else:
                    x = sum(x)
                    y[j] = layer(x)
        return y[-1]

    def init_param(self):
        V = self.V
        for i in range(self.n):
            V[i].init_param()

    def param_n(self):
        V = self.V
        cnt = 0
        for i in range(self.n):
            cnt += V[i].param_n()
        return cnt

    def init_rep(self):
        n = self.n
        V = self.V
        base_mat = [(V[i].rep) for i in range(n)]
        in_mat = [([0] * opt.ar_max_layers) for i in range(n)]
        out_mat = [([0] * opt.ar_max_layers) for i in range(n)]
        for i in range(n):
            for j in self.in_links[i]:
                in_mat[i][i - j] = 1
            for j in self.out_links[i]:
                out_mat[i][j - i] = 1
        self.rep = [(base_mat[i] + in_mat[i] + out_mat[i]) for i in range(n)]
        self.rep = torch.tensor(self.rep, dtype=torch.float, device=opt.device)

    def comp_action_rand(self):
        n = self.n
        V = self.V
        p1 = random.choice(opt.ar_p1)
        action = []
        for i in range(n):
            if random.random() < p1 and V[i].in_shape == V[i].out_shape:
                action.append(1.0)
            else:
                action.append(0.0)
        for i in range(len(self.groups)):
            action.append(random.uniform(*opt.ar_p2))
        p3 = random.choice(opt.ar_p3)
        for i in range(n):
            for j in range(i + 1, n):
                if V[i].out_shape == V[j].in_shape:
                    if random.random() < p3 and not action[j]:
                        action.append(1.0)
                    else:
                        action.append(0.0)
        return np.array(action)

    def comp_rep(self, action):
        n = self.n        
        V = self.V
        p = 0
        base_mat = [(V[i].rep.copy()) for i in range(n)]
        in_mat = [([0] * opt.ar_max_layers) for i in range(n)]
        out_mat = [([0] * opt.ar_max_layers) for i in range(n)]
        for i in range(n):
            if action[p]:
                base_mat[i][:16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            p += 1
        for g in self.groups:
            F = max(1, int((1.0 - action[p]) * g.F))
            for j in g.in_layers:
                if (isinstance(V[j].base, (nn.Conv2d, nn.Linear)) and
                    not action[j]):
                    base_mat[j][14] = F
            for j in g.out_layers:
                if (isinstance(V[j].base, (nn.Conv2d, nn.Linear)) and
                    not action[j]):
                    base_mat[j][15] = F
            p += 1
        for i in range(n):
            for j in range(i + 1, n):
                if V[i].out_shape == V[j].in_shape:
                    if self.E[i][j] or action[p]:
                        in_mat[j][j - i] = 1
                        out_mat[i][j - i] = 1
                    p += 1
        rep = [(base_mat[i] + in_mat[i] + out_mat[i]) for i in range(n)]
        rep = torch.tensor(rep, dtype=torch.float, device=opt.device)
        return rep

    def comp_arch(self, action):
        arch = copy.deepcopy(self)
        arch.action = action
        n = arch.n
        V = arch.V
        p = 0
        for i in range(n):
            if action[p]:
                V[i].replace(layer.Identity())
            p += 1
        in_shapes = [V[i].in_shape for i in range(n)]
        out_shapes = [V[i].out_shape for i in range(n)]
        for g in self.groups:
            F = max(1, int((1.0 - action[p]) * g.F))
            for j in g.inter:
                V[j].shrink(F, F)
            for j in g.in_only:
                Fo = list(V[j].out_shape)[1]
                V[j].shrink(F, Fo)
            for j in g.out_only:
                Fi = list(V[j].in_shape)[1]
                V[j].shrink(Fi, F)
            p += 1
        for i in range(n):
            for j in range(i + 1, n):
                if out_shapes[i] == in_shapes[j]:
                    if action[p]:
                        arch.E[i][j] = True
                    p += 1
        arch.in_links, arch.out_links = gr.get_links(arch.E)
        arch.init_rep()
        arch.to(opt.device)
        return arch

    def comp_arch_rand_sfn(self):

        def shrink_n(F, ratio):
            m = opt.ar_channel_mul
            return max(1, int(ceil((1.0 - ratio) * F / m))) * m

        arch = copy.deepcopy(self)
        n = arch.n
        V = arch.V

        p1 = random.choice(opt.ar_p1)
        for i in range(n):
            if (random.random() < p1 and V[i].in_shape == V[i].out_shape and
                i not in [11, 50, 125]):
                V[i].replace(models.Identity())

        opt.ar_p2[1] = min(0.9, opt.ar_p2[1])
        for g in self.groups:
            p2 = random.uniform(*opt.ar_p2)
            for j in g.inter:
                Fi = shrink_n(list(V[j].in_shape)[1], p2)
                Fo = shrink_n(list(V[j].out_shape)[1], p2)
                V[j].shrink(Fi, Fo)
            for j in g.in_only:
                Fi = shrink_n(list(V[j].in_shape)[1], p2)
                Fo = list(V[j].out_shape)[1]
                V[j].shrink(Fi, Fo)
            for j in g.out_only:
                Fi = list(V[j].in_shape)[1]
                Fo = shrink_n(list(V[j].out_shape)[1], p2)
                V[j].shrink(Fi, Fo)

        F1 = list(V[2].out_shape)[1]
        F4 = list(V[13].out_shape)[1]
        F3 = list(V[13].in_shape)[1]
        F2 = F4 - F3
        V[11].shrink(F1, F2)
        V[12].shrink(F2, F2)

        F1 = list(V[41].out_shape)[1]
        F4 = list(V[52].out_shape)[1]
        F3 = list(V[52].in_shape)[1]
        F2 = F4 - F3
        V[50].shrink(F1, F2)
        V[51].shrink(F2, F2)

        F1 = list(V[116].out_shape)[1]
        F4 = list(V[127].out_shape)[1]
        F3 = list(V[127].in_shape)[1]
        F2 = F4 - F3
        V[125].shrink(F1, F2)
        V[126].shrink(F2, F2)

        p3 = random.choice(opt.ar_p3)
        for i in range(n):
            for j in range(i + 1, n):
                if (random.random() < p3 and V[i].out_shape == V[j].in_shape and
                    not isinstance(V[j].base, (models.Concat, models.Identity))):
                    arch.E[i][j] = True

        arch.in_links, arch.out_links = gr.get_links(arch.E)
        arch.init_rep()
        arch.to(opt.device)
        return arch

    def save(self, save_path):
        path = os.path.dirname(save_path)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self, save_path)