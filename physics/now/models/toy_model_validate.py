import torch
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


class layer2model(torch.nn.Module):
    def __init__(self):
        super(layer2model, self).__init__()
        self.w1 = torch.nn.Parameter(torch.FloatTensor([0.8, -0.1, -.1, -0.1, -0.1]), requires_grad=True)
        self.w2 = torch.nn.Parameter(0.5*torch.ones(5, requires_grad=True))
        print(self.w1)
        print(self.w2)
        self.loss = 0.

    def forward_(self, x):
        mid = x*self.w1
        mid_ = mid.detach()
        # mid = 0.8, activated
        mid_mean = 0.5
        # direct
        # compensate = mid_mean-mid_
        # mid = mid + compensate
        # reverse value
        compensate = mid_mean+mid_[0]
        mid[0] = -mid[0] + compensate
        # reverse result
        # compensate = mid_mean-mid_[0]
        # mid[0] = mid[0] - compensate

        print(mid)
        return torch.sum(mid*self.w2)

    def forward(self, x):
        mid = x*self.w1
        # mid_ = mid.detach()
        # mid = 0.8, activated
        mid_mean = 0.2
        # 1. direct
        # compensate = mid_mean-mid_
        # mid = mid + compensate
        # 2. reverse value
        # compensate = mid_mean+mid_[0]
        # mid[0] = -mid[0] + compensate
        # 3. reverse result
        # compensate = mid_mean-mid_[0]
        # mid[0] = mid[0] - compensate
        t_m, idx = torch.square(mid-mid_mean).max()
        print(t_m)
        print(idx)
        1/0
        self.loss = torch.sum(t_m)/2
        print('loss:', self.loss)
        return torch.sum(mid*self.w2)

encoder_forecaster = layer2model()

LR = 1e-3
optimizer = torch.optim.Adam(encoder_forecaster.parameters(), lr=LR)  # Adadelta

encoder_forecaster.train()
optimizer.zero_grad()

input_, real = torch.from_numpy(np.array([1.])), torch.from_numpy(np.array([0.3]))
output = encoder_forecaster(input_)
loss = (real-output)**2/2# +encoder_forecaster.loss
print(loss)
loss.backward()
optimizer.step()
print(encoder_forecaster.w1)
print(encoder_forecaster.w2)
# directly run 0.8->0.7998; 0.5->0.4997
# add an adding compensation
# -> 0.8->0.8018; 0.5->0.5010   now its too weak! so weighting on it is even larger
# add a reverse value compensation
# -> 0.8->0.0.7990; 0.5->0.5010   the first one is reversed, means that previous connection to it is reduced
# add a reverse output compensation
# -> 0.8->0.0.7990; 0.5->0.4990   both are reversed
