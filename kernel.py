import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import options as opt
from math import sqrt
from scipy.stats import norm
import os

class Kernel(object):

    def __init__(self, x0, y0, alpha=opt.ke_alpha, beta=opt.ke_beta,
                 input_size=opt.ke_input_size, hidden_size=opt.ke_hidden_size,
                 num_layers=opt.ke_num_layers, bidirectional=opt.ke_bidirectional,
                 lr=opt.ke_lr, weight_decay=opt.ke_weight_decay):

        super(Kernel, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bidirectional=bidirectional)
        self.lstm = self.lstm.to(opt.device)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.bi = 2 if bidirectional else 1

        self.x = [x0]
        self.y = torch.tensor([y0], dtype=torch.float, device=opt.device,
                              requires_grad=False)
        self.x_best = x0
        self.y_best = y0
        self.i_best = 0

        self.n = 1
        self.E = self.embedding(x0).view(1, -1)
        self.K = self.kernel(self.E[0], self.E[0]).view(1, 1)
        self.K_inv = torch.inverse(self.K + self.beta *
                                   torch.eye(self.n, device=opt.device))
        self.optimizer = optim.Adam(self.lstm.parameters(), lr=lr,
                                    weight_decay=weight_decay)

    def embedding(self, xi):
        inputs = xi.view(-1, 1, self.input_size)
        outputs, (hn, cn) = self.lstm(inputs)
        outputs = torch.mean(outputs.squeeze(1), dim=0)
        outputs = outputs / torch.norm(outputs)
        return outputs

    def kernel(self, ei, ej):
        d = ei - ej
        d = torch.sum(d * d)
        k = torch.exp(-d / (2 * self.alpha))
        return k

    def kernel_batch(self, en):
        n = self.n
        k = torch.zeros(n, device=opt.device)
        for i in range(n):
            k[i] = self.kernel(self.E[i], en)
        return k

    def predict(self, xn):
        n = self.n
        en = self.embedding(xn)
        k = self.kernel_batch(en)
        kn = self.kernel(en, en)
        t = torch.mm(k.view(1, n), self.K_inv)
        mu = torch.mm(t, self.y.view(n, 1))
        sigma = kn - torch.mm(t, k.view(n, 1))
        sigma = torch.sqrt(sigma + self.beta)
        return mu, sigma

    def acquisition(self, xn):
        with torch.no_grad():
            mu, sigma = self.predict(xn)
            mu = mu.item()
            sigma = sigma.item()
            y_best = self.y_best
            z = (mu - y_best) / sigma
            ei = (mu - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

    def kernel_batch_ex(self, t):
        n = self.n
        k = torch.zeros(n - 1, device=opt.device)
        for i in range(t):
            k[i] = self.kernel(self.E[i], self.E[t])
        for i in range(t + 1, n):
            k[i - 1] = self.kernel(self.E[t], self.E[i])
        return k

    def predict_ex(self, t):
        n = self.n
        k = self.kernel_batch_ex(t)
        kt = self.kernel(self.E[t], self.E[t])
        indices = list(range(t)) + list(range(t + 1, n))
        indices = torch.tensor(indices, dtype=torch.long, device=opt.device)
        K = self.K
        K = torch.index_select(K, 0, indices)
        K = torch.index_select(K, 1, indices)
        K_inv = torch.inverse(K + self.beta *
                              torch.eye(n - 1, device=opt.device))
        y = torch.index_select(self.y, 0, indices)

        t = torch.mm(k.view(1, n - 1), K_inv)
        mu = torch.mm(t, y.view(n - 1, 1))
        sigma = kt - torch.mm(t, k.view(n - 1, 1))
        sigma = torch.sqrt(sigma + self.beta)
        return mu, sigma

    def add_sample(self, xn, yn):
        self.x.append(xn)
        self.y = torch.cat((self.y, torch.tensor([yn], dtype=torch.float,
                                                 device=opt.device,
                                                 requires_grad=False)))
        n = self.n
        if yn > self.y_best:
            self.x_best = xn
            self.y_best = yn
            self.i_best = n
        en = self.embedding(xn)
        k = self.kernel_batch(en)
        kn = self.kernel(en, en)
        self.E = torch.cat((self.E, en.view(1, -1)), 0)
        self.K = torch.cat((torch.cat((self.K, k.view(n, 1)), 1),
                            torch.cat((k.view(1, n), kn.view(1, 1)), 1)), 0)
        self.n += 1
        self.K_inv = torch.inverse(self.K + self.beta *
                                   torch.eye(self.n, device=opt.device))

    def add_batch(self, x, y):
        self.x.extend(x)
        self.y = torch.cat((self.y, y))
        m = len(x)
        for i in range(m):
            n = self.n
            if y[i].item() > self.y_best:
                self.x_best = x[i]
                self.y_best = y[i].item()
                self.i_best = n
            en = self.embedding(x[i])
            k = self.kernel_batch(en)
            kn = self.kernel(en, en)
            self.E = torch.cat((self.E, en.view(1, -1)), 0)
            self.K = torch.cat((torch.cat((self.K, k.view(n, 1)), 1),
                                torch.cat((k.view(1, n), kn.view(1, 1)), 1)), 0)
            self.n += 1
        self.K_inv = torch.inverse(self.K + self.beta *
                                   torch.eye(self.n, device=opt.device))

    def update_EK(self):
        n = self.n
        E_ = torch.zeros((n, self.E.size(1)), device=opt.device)
        for i in range(n):
            E_[i] = self.embedding(self.x[i])
        self.E = E_
        K_ = torch.zeros((n, n), device=opt.device)
        for i in range(n):
            for j in range(i, n):
                k = self.kernel(self.E[i], self.E[j])
                K_[i, j] = k
                K_[j, i] = k
        self.K = K_
        self.K_inv = torch.inverse(self.K + self.beta *
                                   torch.eye(self.n, device=opt.device))

    def loss(self):
        n = self.n
        l = torch.zeros(n, device=opt.device)
        for i in range(n):
            mu, sigma = self.predict_ex(i)
            d = self.y[i] - mu
            l[i] = -(0.918939 + torch.log(sigma) + d * d / (2 * sigma * sigma))
        l = -torch.mean(l)
        return l

    def opt_step(self):
        if self.n < 2:
            return 0.0
        self.optimizer.zero_grad()
        l = self.loss()
        ll = -l.item()
        l.backward()
        self.optimizer.step()
        self.update_EK()
        return ll

    def save(self, save_path):
        path = os.path.dirname(save_path)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self, save_path)