import torch
from torch import nn
from copy import deepcopy


class EmeL(nn.Module):
    def __init__(self, h_upper=10, h_mean=None, h_var=None):
        super(EmeL, self).__init__()
        self.h_var = h_var
        self.h_mean = h_mean
        self.h_count = 0
        self.h_upper = h_upper
        self.emel_loss = 0.
        self._batch_size = 0

    def punish_best(self, h):
        shape = h.shape
        diff = torch.square(h - self.h_mean)
        max_in_batch, idx = torch.max(((h - self.h_mean).pow(2) /
                                       (self.h_var + self.h_var.mean() / 100)).reshape((self._batch_size, -1)), 1)
        # find the h with largest phi in each batch        

        total_dim = h.reshape(self._batch_size, -1).shape[1]
        h_idx = deepcopy(idx)
        for i in range(self._batch_size):
            h_idx[i] += i * total_dim
        change = torch.zeros_like(h).reshape(-1)
        change[h_idx] = 1.
        
        # update h
        h = h.reshape(-1)
        h_mean = self.h_mean.reshape(-1)
        inc = torch.zeros_like(h)
        inc[h_idx] = h_mean[idx] + h.detach()[h_idx]
        h = (1 - change * 2) * h + inc
        h = h.reshape(shape)

        # change = change.reshape(shape)
        # self.emel_loss = torch.sum(diff * change) / 2
        return h

    def forward(self, h):
        if self.training:
            h_ = h.detach()
            h_u = h_.mean(dim=0, keepdim=True)
            h_v = h_.var(dim=0, keepdim=True, unbiased=False)
            self.h_count = self.h_count+1
            if self.h_var is None:
                batch = h_.shape[0]
                self._batch_size = batch
                self.h_var = h_v * batch / (batch - 1)
                self.h_mean = h_u

            elif self.h_count <= self.h_upper:
                self.h_var = (self.h_var * (self.h_count - 1. / self._batch_size) + h_v +
                              (h_u - self.h_mean) ** 2 / (1 + 1. / self.h_count)) / \
                             (self.h_count + 1 - 1. / self._batch_size)
                self.h_mean = (self.h_mean * self.h_count + h_u) / (self.h_count + 1)

            else:
                self.h_var = (self.h_var * (self.h_upper - 1. / self._batch_size) + h_v +
                              (h_u - self.h_mean) ** 2 / (1 + 1. / self.h_upper)) / \
                             (self.h_upper + 1 - 1. / self._batch_size)
                self.h_mean = (self.h_mean * self.h_upper + h_u) / (self.h_upper + 1)
                h = self.punish_best(h)
                # h = self.punish_best_k(h, 10)
        return h

    def punish_best_k(self, h, k=3):
        shape = h.shape
        diff = torch.square(h - self.h_mean)
        total_dim = h.reshape(self._batch_size, -1).shape[1]
        h_idx = torch.zeros((self._batch_size * k), dtype=torch.long)
        idx_ = torch.zeros((self._batch_size * k), dtype=torch.long)
        for i in range(self._batch_size):
            max_in_batch, idx = torch.topk(((h[i].reshape(1, *shape[1:]) - self.h_mean).pow(2) /
                                            (self.h_var + self.h_var.mean() / 100)).reshape(-1), k)
            for j in range(k):
                h_idx[j + i * k] = idx[j] + i * total_dim
                idx_[j+i*k] = idx[j]
        change = torch.zeros_like(h).reshape(-1)
        change[h_idx] = 1.

        h = h.reshape(-1)
        h_mean = self.h_mean.reshape(-1)
        inc = torch.zeros_like(h)
        inc[h_idx] = h_mean[idx_] + h.detach()[h_idx]
        h = (1 - change * 2) * h + inc
        h = h.reshape(shape)

        # change = change.reshape(shape)
        # self.emel_loss = torch.sum(diff * change) / 2
        return h
