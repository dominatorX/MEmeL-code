import sys
sys.path.insert(0, '..')
from now.models.trajGRU import TrajGRU
from now.hko.evaluation import *
from now.models.ConvGRU import ConvGRU
from now.models.convLSTM import ConvLSTM
import torch


batch_size = cfg.GLOBAL.BATCH_SIZE
IN_LEN = cfg.HKO.BENCHMARK.IN_LEN
OUT_LEN = cfg.HKO.BENCHMARK.OUT_LEN

f1, f2, f3 = 8, 64, 192
em = cfg.GLOBAL.EMEL
endegru120_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [cfg.HKO.ITERATOR.CHANNEL, f1, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [f2, f2, 5, 3, 1]}),
        OrderedDict({'conv3_leaky_1': [f3, f3, 3, 2, 1]}),
    ],

    [
        ConvGRU(input_channel=f1, num_filter=f2, b_h_w=(batch_size, 24, 24),
                kernel_size=3, stride=1, padding=1, st_kernel=5, em_flag=em),
        ConvGRU(input_channel=f2, num_filter=f3, b_h_w=(batch_size, 8, 8),
                kernel_size=3, stride=1, padding=1, st_kernel=5, em_flag=em),
        ConvGRU(input_channel=f3, num_filter=f3, b_h_w=(batch_size, 4, 4),
                kernel_size=3, stride=1, padding=1, st_kernel=3, em_flag=em),
    ]
]
endegru120_forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [f3, f3, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [f3, f3, 5, 3, 1]}),
        OrderedDict({
            'deconv3_leaky_1': [f2, f1, 7, 5, 1],
            # 'conv3_leaky_2': [f1, f1, 3, 1, 1],
            'conv3_3': [f1, 1, 1, 1, 0]
        }),
    ],

    [
        ConvGRU(input_channel=f3, num_filter=f3, b_h_w=(batch_size, 4, 4),
                kernel_size=3, stride=1, padding=1, st_kernel=3, em_flag=em),
        ConvGRU(input_channel=f3, num_filter=f3, b_h_w=(batch_size, 8, 8),
                kernel_size=3, stride=1, padding=1, st_kernel=5, em_flag=em),
        ConvGRU(input_channel=f3, num_filter=f2, b_h_w=(batch_size, 24, 24),
                kernel_size=3, stride=1, padding=1, st_kernel=5, em_flag=em),
    ]
]

# trajgru
f1, f2, f3, ls1, ls2 = 8, 64, 192, 13, 9
endetraj_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [cfg.HKO.ITERATOR.CHANNEL, f1, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [f2, f2, 5, 3, 1]}),
        OrderedDict({'conv3_leaky_1': [f3, f3, 3, 2, 1]}),
    ],

    [
        TrajGRU(input_channel=f1, num_filter=f2, b_h_w=(batch_size, 32, 42), zoneout=0.0, L=ls1,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                norm=None),

        TrajGRU(input_channel=f2, num_filter=f3, b_h_w=(batch_size, 10, 14), zoneout=0.0, L=ls1,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                norm=None),
        TrajGRU(input_channel=f3, num_filter=f3, b_h_w=(batch_size, 5, 7), zoneout=0.0, L=ls2,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                norm=None)
    ]
]
endetraj_forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [f3, f3, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [f3, f3, 5, 3, (0, 1)]}),
        OrderedDict({
            'deconv3_leaky_1': [f2, f1, (7, 8), 5, 0],
            'conv3_leaky_2': [f1, f1, 3, 1, 1],
            'conv3_3': [f1, 1, 1, 1, 0]
        }),
    ],

    [
        TrajGRU(input_channel=f3, num_filter=f3, b_h_w=(batch_size, 5, 7), zoneout=0.0, L=ls2,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                norm=None),
        TrajGRU(input_channel=f3, num_filter=f3, b_h_w=(batch_size, 10, 14), zoneout=0.0, L=ls1,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                norm=None),
        TrajGRU(input_channel=f3, num_filter=f2, b_h_w=(batch_size, 32, 42), zoneout=0.0, L=ls1,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                norm=None)
    ]
]

# convgru
f1, f2, f3 = 8, 64, 192
endegru_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [cfg.HKO.ITERATOR.CHANNEL, f1, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [f2, f2, 5, 3, 1]}),
        OrderedDict({'conv3_leaky_1': [f3, f3, 3, 2, 1]}),
    ],

    [
        ConvGRU(input_channel=f1, num_filter=f2, b_h_w=(batch_size, 96, 96), #24, 24),#  32, 42),
                kernel_size=3, stride=1, padding=1, st_kernel=5),
        ConvGRU(input_channel=f2, num_filter=f3, b_h_w=(batch_size, 32, 32), #8, 8), #  10, 14),
                kernel_size=3, stride=1, padding=1, st_kernel=5),
        ConvGRU(input_channel=f3, num_filter=f3, b_h_w=(batch_size, 16, 16), #4, 4), #  5, 7),
                kernel_size=3, stride=1, padding=1, st_kernel=3),
    ]
]
endegru_forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [f3, f3, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [f3, f3, 5, 3, 1]}),# (0, 1)]}),
        OrderedDict({
            'deconv3_leaky_1': [f2, f1, 7, 5, 1],# (7, 8), 5, 0],
            # 'conv3_leaky_2': [f1, f1, 3, 1, 1],
            'conv3_3': [f1, 1, 1, 1, 0]
        }),
    ],

    [
        ConvGRU(input_channel=f3, num_filter=f3, b_h_w=(batch_size, 16, 16),#4, 4), #  5, 7),
                kernel_size=3, stride=1, padding=1, st_kernel=3),
        ConvGRU(input_channel=f3, num_filter=f3, b_h_w=(batch_size, 32, 32), #8, 8), #  10, 14),
                kernel_size=3, stride=1, padding=1, st_kernel=5),
        ConvGRU(input_channel=f3, num_filter=f2, b_h_w=(batch_size, 96, 96), #24, 24), #  32, 42),
                kernel_size=3, stride=1, padding=1, st_kernel=5),
    ]
]

# convlstm
f1, f2, f3 = 8, 64, 192
endelstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [cfg.HKO.ITERATOR.CHANNEL, f1, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [f2, f2, 5, 3, 1]}),
        OrderedDict({'conv3_leaky_1': [f3, f3, 3, 2, 1]}),
    ],

    [
        ConvLSTM(input_channel=f1, num_filter=f2, b_h_w=(batch_size, 32, 42),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=f2, num_filter=f3, b_h_w=(batch_size, 10, 14),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=f3, num_filter=f3, b_h_w=(batch_size, 5, 7),
                 kernel_size=3, stride=1, padding=1),
    ]
]
endelstm_forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [f3, f3, 2, 2, 0]}),
        OrderedDict({'deconv2_leaky_1': [f3, f3, 5, 3, (0, 1)]}),
        OrderedDict({
            'deconv3_leaky_1': [f2, f1, (7, 8), 5, 0],
            'conv3_leaky_2': [f1, f1, 3, 1, 1],
            'conv3_3': [f1, 1, 1, 1, 0]
        }),
    ],

    [
        ConvLSTM(input_channel=f3, num_filter=f3, b_h_w=(batch_size, 5, 7),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=f3, num_filter=f3, b_h_w=(batch_size, 10, 14),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=f3, num_filter=f2, b_h_w=(batch_size, 32, 42),
                 kernel_size=3, stride=1, padding=1),
    ]
]
