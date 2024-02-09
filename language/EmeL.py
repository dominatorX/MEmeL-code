import torch
from torch import nn


class EmeL(nn.Module):
    def __init__(self, h_upper=10, ratio=1., t=1, h_mean=None, h_var=None):
        super(EmeL, self).__init__()
        self.h_var = h_var
        self.h_mean = h_mean
        self.h_count = 0
        self.h_upper = h_upper
        self.emel_loss = 0.
        self.ratio = ratio
        self.t = t

    def punish_best(self, h):
        shape = h.shape
        _batch_size = shape[0]

        dev = ((h.detach() - self.h_mean).pow(2) / (self.h_var + self.h_var.mean() / 100)).reshape((_batch_size, -1))
        max_in_batch, _ = torch.max(dev, 1, keepdim=True)
        update_idx = (max_in_batch == dev).reshape(shape)

        # diff = torch.square(h - self.h_mean)
        # self.emel_loss = torch.sum(diff * update_idx) / 2

        inc = self.h_mean + self.ratio*h.detach()
        h = (1 - update_idx * (1+self.ratio)) * h + inc * update_idx
        return h

    def forward(self, h):
        if self.training:
            h_ = torch.tensor(h.detach(), dtype=torch.float32)
            h_u = h_.mean(dim=0, keepdim=True)
            h_v = h_.var(dim=0, keepdim=True, unbiased=False)
            self.h_count = self.h_count+1
            _batch_size = h_.shape[0]
            if self.h_var is None:
                self.h_var = h_v / (_batch_size - 1.) * _batch_size
                self.h_mean = h_u
            elif self.h_count <= self.h_upper:
                self.h_var = (self.h_var * (self.h_count - 1. / _batch_size) + h_v +
                              (h_u - self.h_mean) ** 2 / (1 + 1. / self.h_count)) / \
                             (self.h_count + 1 - 1. / _batch_size)
                self.h_mean = (self.h_mean * self.h_count + h_u) / (self.h_count + 1.)

            else:
                self.h_var = (self.h_var * (self.h_upper - 1. / _batch_size) + h_v +
                              (h_u - self.h_mean) ** 2 / (1 + 1. / self.h_upper)) / \
                             (self.h_upper + 1 - 1. / _batch_size)
                self.h_mean = (self.h_mean * self.h_upper + h_u) / (self.h_upper + 1.)
                if self.t > 1:
                    h = self.punish_best_k(h, self.t)
                else:
                    h = self.punish_best(h)
            if self.h_var.mean().isfinite() is False:
                raise ValueError('h_var is not finite')
        return h

    def punish_best_k(self, h, k=10):
        shape = h.shape
        _batch_size = shape[0]

        dev = -((h.detach() - self.h_mean).pow(2) /
                (self.h_var + self.h_var.mean() / 100)).reshape(_batch_size, -1)
        kth_value = torch.kthvalue(dev, k, keepdim=True)[0]
        update_idx = torch.where(dev <= kth_value, torch.ones_like(dev), torch.zeros_like(dev)).reshape(shape)

        # diff = torch.square(h - self.h_mean)
        # self.emel_loss = torch.sum(diff * update_idx) / 2

        inc = self.h_mean + self.ratio*h.detach()
        h = (1 - update_idx * (1+self.ratio)) * h + inc * update_idx
        return h
