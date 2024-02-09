from now.helpers.ordered_easydict import OrderedEasyDict as edict
import numpy as np
import os
import torch


__C = edict()
cfg = __C
__C.GLOBAL = edict()


__C.GLOBAL.BATCH_SIZE = 16
__C.GLOBAL.MODEL_NAME = 'model/name'
__C.GLOBAL.EMEL = True
__C.GLOBAL.EMEL_RATIO = 1.
__C.GLOBAL.EMEL_TOPK = 1
__C.GLOBAL.TUNE_NAME = 'finetune/output/name'
__C.GLOBAL.TUNE_STEP = 2000
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
__C.GLOBAL.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__C.GLOBAL.MODEL_SAVE_DIR = 'model/save/dir'
__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
__C.HKO_DATA_BASE_PATH = os.path.join(__C.ROOT_DIR, 'data')

for dirs in ['HKO_radar_images/radarPNG']:
    if os.path.exists(dirs):
        __C.HKO_PNG_PATH = dirs
for dirs in ['HKO_radar_images/radarPNG_mask']:
    if os.path.exists(dirs):
        __C.HKO_MASK_PATH = dirs

__C.HKO = edict()


__C.HKO.EVALUATION = edict()
__C.HKO.EVALUATION.THRESHOLDS = np.array([0.5, 2, 5, 10, 30])
__C.HKO.EVALUATION.CENTRAL_REGION = (0, 0, 480, 480)
__C.HKO.EVALUATION.BALANCING_WEIGHTS = (1, 1, 2, 5, 10, 30)

__C.HKO.EVALUATION.VALID_DATA_USE_UP = False
# __C.HKO.EVALUATION.VALID_TIME = 100
__C.HKO.EVALUATION.VALID_TIME = 128/__C.GLOBAL.BATCH_SIZE


__C.HKO.BENCHMARK = edict()
__C.HKO.BENCHMARK.VISUALIZE_SEQ_NUM = 10  # Number of sequences that will be plotted and saved to the benchmark directory
__C.HKO.BENCHMARK.IN_LEN = 10  # The maximum input length to ensure that all models are tested on the same set of input data
__C.HKO.BENCHMARK.OUT_LEN = 10  # The maximum output length to ensure that all models are tested on the same set of input data
__C.HKO.BENCHMARK.STRIDE = 5   # The stride

# pandas data
__C.HKO_PD_BASE_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'pd')
if not os.path.exists(__C.HKO_PD_BASE_PATH):
    os.makedirs(__C.HKO_PD_BASE_PATH)

__C.HKO_PD = edict()
__C.HKO_PD.RAINY_TRAIN = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_rainy_train.pkl')
__C.HKO_PD.RAINY_VALID = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_rainy_valid.pkl')
__C.HKO_PD.RAINY_TEST = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_rainy_test.pkl')
__C.HKO_PD.RAINY_ALL = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_all.pkl')


__C.HKO.ITERATOR = edict()
__C.HKO.ITERATOR.HEIGHT = 120
__C.HKO.ITERATOR.WIDTH = 120
__C.HKO.ITERATOR.CHANNEL = 1


__C.HKO.ITERATOR.MODEL = 'gru'  # lstm, gru, traj
__C.HKO.ITERATOR.FRAME = 'ende'

__C.MODEL = edict()
from now.models.model import activation
__C.MODEL.CNN_ACT_TYPE = activation('leaky', negative_slope=0.2, inplace=True)  # activation("relu")  #
__C.MODEL.RNN_ACT_TYPE = activation('leaky', negative_slope=0.2, inplace=True)
