import torch
from torch import nn
from copy import deepcopy


class EmeLMP(nn.Module):
    def __init__(self, shape, h_upper=10, h_mean=None, h_var=None, dtype=torch.float16):
        super(EmeLMP, self).__init__()
        '''
        :param shape: shape of input tensor
        :param h_upper: upper bound of saved history
        :param h_mean: mean of saved history
        :param h_var: variance of saved history
        h_count: current number of saved history
        emel_loss: the loss of emel
        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if h_var is not None:
            self.h_var = h_var.cpu()
        else:
            self.h_var = torch.zeros(1, *shape, dtype=dtype).cpu()
        if h_mean is not None:
            self.h_mean = h_mean.cpu()
        else:
            self.h_mean = torch.zeros(1, *shape, dtype=dtype).cpu()
        self.h_upper = h_upper

        self.h_count = torch.zeros(1, dtype=torch.long).cpu()

    def punish_best(self, h):
        shape = h.shape
        batch_size = shape[0]
        h_mean = self.h_mean.to(self.device)
        h_var = self.h_var.to(self.device)
        # diff = torch.square(h - h_mean)
        max_in_batch, idx = torch.max(((h - h_mean).pow(2) /
                                       (h_var + h_var.mean() / 100)).reshape((batch_size, -1)), 1)
        # find the h with largest phi in each batch  

        total_dim = h.reshape(batch_size, -1).shape[1]
        h_idx = deepcopy(idx)
        for i in range(batch_size):
            h_idx[i] += i * total_dim
        change = torch.zeros_like(h).reshape(-1)
        change[h_idx] = 1.

        # update h
        h = h.reshape(-1)
        h_mean = h_mean.reshape(-1)
        inc = torch.zeros_like(h)
        inc[h_idx] = h_mean[idx] + h.detach()[h_idx]
        h = (1 - change * 2) * h + inc
        h = h.reshape(shape)

        # change = change.reshape(shape)
        # emel_loss = torch.sum(diff * change) / 2

        return h # , emel_loss

    def forward(self, h):
        if self.training:
            h_ = h.detach()
            h_u = h_.mean(dim=0, keepdim=True).cpu()
            h_v = h_.var(dim=0, keepdim=True, unbiased=False).cpu()

            self.h_count += 1
            h_count = self.h_count
            # print(h_count)
            batch_size = h.shape[0]
            if h_count == 1:
                self.h_mean += h_u
                self.h_var += h_v * batch_size / (batch_size - 1)

            elif h_count <= self.h_upper:
                self.h_mean += (-self.h_mean + h_u) / (h_count + 1)

                self.h_var += (-self.h_var + h_v + (h_u - self.h_mean) ** 2 / (1 + 1. / h_count)) / \
                              (h_count + 1 - 1. / batch_size)
                # print(self.h_var[0, 0, 0])
            else:
                self.h_var += (-self.h_var + h_v +
                               (h_u - self.h_mean) ** 2 / (1 + 1. / self.h_upper)) / \
                              (self.h_upper + 1 - 1. / batch_size)
                self.h_mean += (-self.h_mean + h_u) / (self.h_upper + 1)
                h = self.punish_best(h)

        return h
