import torch
from torch import nn


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
        _batch_size = shape[0]
        h_mean = self.h_mean.to(self.device)
        h_var = self.h_var.to(self.device)

        dev = ((h.detach() - h_mean).pow(2) / (h_var + h_var / 100)).reshape((_batch_size, -1))
        max_in_batch, _ = torch.max(dev, 1, keepdim=True)
        update_idx = (max_in_batch == dev).reshape(shape)

        # diff = torch.square(h - self.h_mean)
        # emel_loss = torch.sum(diff * update_idx) / 2

        inc = h_mean + h.detach()
        h = (1 - update_idx * 2) * h + inc * update_idx

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
                self.h_var += h_v / (batch_size - 1.) * batch_size

            elif h_count <= self.h_upper:
                self.h_var += (-self.h_var + h_v + (h_u - self.h_mean) ** 2 / (1 + 1. / h_count)) / \
                              (h_count + 1 - 1. / batch_size)
                self.h_mean += (-self.h_mean + h_u) / (h_count + 1.)
                # print(self.h_var[0, 0, 0])
            else:
                self.h_var += (-self.h_var + h_v +
                               (h_u - self.h_mean) ** 2 / (1 + 1. / self.h_upper)) / \
                              (self.h_upper + 1 - 1. / batch_size)
                self.h_mean += (-self.h_mean + h_u) / (self.h_upper + 1.)
                h = self.punish_best(h)

        return h
