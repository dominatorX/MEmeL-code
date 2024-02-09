import torch
from torch import nn
from now.config import cfg
from now.models.EmeL import EmeL
# input: B, C, H, W
# flow: [B, 2, H, W]


class ConvGRU(nn.Module):
    # b_h_w: input feature map size
    def __init__(self, input_channel, num_filter, b_h_w, kernel_size, stride=1, padding=1,
                 st_kernel=3, rnn_act_type=cfg.MODEL.RNN_ACT_TYPE, cnn_act_type=cfg.MODEL.CNN_ACT_TYPE,
                 em_flag=False):
        super(ConvGRU, self).__init__()
        self._batch_size, self._state_height, self._state_width = b_h_w
        self._input_channel = input_channel
        self._cnn_act_type = cnn_act_type
        self._rnn_act_type = rnn_act_type
        self._num_filter = num_filter
        # 对应 wxz, wxr, wxh
        # reset_gate, update_gate, new_mem
        self.i2h = nn.Conv2d(in_channels=input_channel,
                             out_channels=self._num_filter * 3,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        self.h2h = nn.Conv2d(in_channels=num_filter,
                             out_channels=self._num_filter * 3,
                             kernel_size=st_kernel,
                             stride=stride,
                             padding=st_kernel // 2)
        self.em = em_flag
        if em_flag:
            self.emel = EmeL() 

    def forward(self, inputs=None, states=None, seq_len=cfg.HKO.BENCHMARK.IN_LEN):
        if states is None:
            states = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                                  self._state_width), dtype=torch.float).to(cfg.GLOBAL.DEVICE)

        outputs = []

        for index in range(seq_len):
            # initial inputs
            if inputs is None:
                x = torch.zeros((states.size(0), self._input_channel, self._state_height,
                                 self._state_width), dtype=torch.float).to(cfg.GLOBAL.DEVICE)
            else:
                x = inputs[index, ...]
            conv_x = self.i2h(x)
            conv_h = self.h2h(states)
            if self._cnn_act_type:
                conv_x = self._cnn_act_type(conv_x)
                conv_h = self._cnn_act_type(conv_h)
            xz, xr, xh = torch.chunk(conv_x, 3, dim=1)
            hz, hr, hh = torch.chunk(conv_h, 3, dim=1)

            zt = torch.sigmoid(xz+hz)
            rt = torch.sigmoid(xr+hr)
            h_ = self._rnn_act_type(xh+rt*hh)
            if self.em:
                h_ = self.emel(h_)
            h = (1-zt)*h_+zt*states
            outputs.append(h)
            states = h
        return torch.stack(outputs), states


