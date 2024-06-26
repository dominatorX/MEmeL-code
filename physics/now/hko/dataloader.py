import pandas as pd
import numpy as np
import bisect
from now.hko import image
from now.hko.mask import *
from now.config import cfg
from now.utils import *
import math
import json
import os
import pickle


def encode_month(month):
    """Encode the month into a vector

    Parameters
    ----------
    month : np.ndarray
        (...,) int, between 1 and 12
    Returns
    -------
    ret : np.ndarray
        (..., 2) float
    """
    angle = 2 * np.pi * month / 12.0
    ret = np.empty(shape=month.shape + (2,), dtype=np.float32)
    ret[..., 0] = np.cos(angle)
    ret[..., 1] = np.sin(angle)
    return ret


def decode_month(code):
    """Decode the month code back to the month value

    Parameters
    ----------
    code : np.ndarray
        (..., 2) float
    Returns
    -------
    month : np.ndarray
        (...,) int
    """
    assert code.shape[-1] == 2
    flag = code[..., 1] >= 0
    arccos_res = np.arccos(code[..., 0])
    angle = flag * arccos_res + (1 - flag) * (2 * np.pi - arccos_res)
    month = angle / (2.0 * np.pi) * 12.0
    month = np.round(month).astype(np.int)
    return month


def get_valid_datetime_set():
    valid_datetime_set = pickle.load(open(cfg.HKO_VALID_DATETIME_PATH, 'rb'))
    return valid_datetime_set


def get_exclude_mask():
    with np.load(os.path.join(cfg.HKO_DATA_BASE_PATH, 'mask_dat.npz')) as dat:
        exclude_mask = dat['exclude_mask'][:]
        return exclude_mask


def convert_datetime_to_filepath(date_time):
    """Convert datetime to the filepath

    Parameters
    ----------
    date_time : datetime.datetime

    Returns
    -------
    ret : str
    """
    ret = os.path.join("%04d" % date_time.year,
                       "%02d" % date_time.month,
                       "%02d" % date_time.day,
                       'RAD%02d%02d%02d%02d%02d00.png'
                       % (date_time.year - 2000, date_time.month, date_time.day,
                          date_time.hour, date_time.minute))
    ret = os.path.join(cfg.HKO_PNG_PATH, ret)
    return ret


def convert_datetime_to_maskpath(date_time):
    """Convert datetime to path of the mask

    Parameters
    ----------
    date_time : datetime.datetime

    Returns
    -------
    ret : str
    """
    ret = os.path.join("%04d" % date_time.year,
                       "%02d" % date_time.month,
                       "%02d" % date_time.day,
                       'RAD%02d%02d%02d%02d%02d00.mask'
                       % (date_time.year - 2000, date_time.month, date_time.day,
                          date_time.hour, date_time.minute))
    ret = os.path.join(cfg.HKO_MASK_PATH, ret)
    return ret


class HKOSimpleBuffer(object):
    def __init__(self, df, max_buffer_length, width, height):
        self._df = df
        self._max_buffer_length = max_buffer_length
        assert self._df.size > self._max_buffer_length
        self._width = width
        self._height = height

    def reset(self):
        self._datetime_keys = self._df.index[:self._max_buffer_length]
        self._load()

    def _load(self):
        paths = []
        for i in range(self._datetime_keys.size):
            paths.append(convert_datetime_to_filepath(self._datetime_keys[i]))
        self._frame_dat = image.quick_read_frames(path_list=paths,
                                                  im_h=self._height,
                                                  im_w=self._width,
                                                  grayscale=True)
        self._frame_dat = self._frame_dat.reshape((self._max_buffer_length, 1,
                                                   self._height, self._width))
        self._noise_mask_dat = np.zeros((self._datetime_keys.size, 1,
                                         self._height, self._width),
                                        dtype=np.uint8)

    def get(self, timestamps):
        """timestamps must be sorted

        Parameters
        ----------
        timestamps

        Returns
        -------

        """
        if not (timestamps[0] in self._datetime_keys and timestamps[-1] in self._datetime_keys):
            read_begin_ind = self._df.index[self._df.index.get_loc(timestamps[0])]
            read_end_ind = min(read_begin_ind + self._max_buffer_length, self._df.size)
            assert self._df.index[read_end_ind - 1] >= timestamps[-1]
            self._datetime_keys = self._df.index[read_begin_ind:read_end_ind]
            self._load()
        begin_ind = self._datetime_keys.get_loc(timestamps[0])
        end_ind = self._datetime_keys.get_loc(timestamps[-1]) + 1
        return self._frame_dat[begin_ind:end_ind, :, :, :], \
               self._noise_mask_dat[begin_ind:end_ind, :, :, :]


def pad_hko_dat(frame_dat, mask_dat, batch_size):
    if frame_dat.shape[1] < batch_size:
        ret_frame_dat = np.zeros(shape=(frame_dat.shape[0], batch_size,
                                        frame_dat.shape[2], frame_dat.shape[3], frame_dat.shape[4]),
                                 dtype=frame_dat.dtype)
        ret_mask_dat = np.zeros(shape=(mask_dat.shape[0], batch_size,
                                       mask_dat.shape[2], mask_dat.shape[3], mask_dat.shape[4]),
                                dtype=mask_dat.dtype)
        ret_frame_dat[:, :frame_dat.shape[1], ...] = frame_dat
        ret_mask_dat[:, :frame_dat.shape[1], ...] = mask_dat
        return ret_frame_dat, ret_mask_dat, frame_dat.shape[1]
    else:
        return frame_dat, mask_dat, batch_size


_exclude_mask = get_exclude_mask()


def precompute_mask(img):
    if img.dtype == np.uint8:
        threshold = round(cfg.HKO.ITERATOR.FILTER_RAINFALL_THRESHOLD * 255.0)
    else:
        threshold = cfg.HKO.ITERATOR.FILTER_RAINFALL_THRESHOLD
    mask = np.zeros_like(img, dtype=np.bool)
    mask[:] = np.broadcast_to((1 - _exclude_mask).astype(np.bool), shape=img.shape)
    mask[np.logical_and(img < threshold,
                        img >= 0)] = 0
    return mask


class HKOIterator(object):
    def __init__(self, pd_path, sample_mode, seq_len=30, batch_size=16,
                 max_consecutive_missing=0, begin_ind=None, end_ind=None,
                 stride=None, base_freq='6min'):
        """Random sample: sample a random clip that will not violate the max_missing frame_num criteria
        Sequent sample: sample a clip from the beginning of the time.
                        Everytime, the clips from {T_begin, T_begin + 6min, ..., T_begin + (seq_len-1) * 6min} will be used
                        The begin datetime will move forward by adding stride: T_begin += 6min * stride
                        Once the clips violates the maximum missing number criteria, the starting
                         point will be moved to the next datetime that does not violate the missing_frame criteria

        Parameters
        ----------
        pd_path : str
            path of the saved pandas dataframe
        sample_mode : str
            Can be "random" or "sequent"
        seq_len : int
        max_consecutive_missing : int
            The maximum consecutive missing frames
        begin_ind : int
            Index of the begin frame
        end_ind : int
            Index of the end frame
        stride : int or None, optional
        width : int or None, optional
        height : int or None, optional
        base_freq : str, optional
        """
        self._df = pd.read_pickle(pd_path)
        self._df_all = pd.read_pickle(cfg.HKO_PD.RAINY_ALL)
        self.set_begin_end(begin_ind=begin_ind, end_ind=end_ind)
        self._df_index_set_all = frozenset([self._df_all.index[i] for i in range(self._df_all.size)])
        if cfg.HKO.ITERATOR.HEIGHT == 480:
            self._exclude_mask = 1 - _exclude_mask[np.newaxis]
        else:
            # extract 120*120 central areas for Hong Kong, covering all the land areas
            self._exclude_mask = 1-_exclude_mask[np.newaxis, 224-cfg.HKO.ITERATOR.HEIGHT//2:224+cfg.HKO.ITERATOR.HEIGHT//2,
                                     235-cfg.HKO.ITERATOR.WIDTH//2:235+cfg.HKO.ITERATOR.WIDTH//2]# 170: 332, 102: 315]
        self._channel = cfg.HKO.ITERATOR.CHANNEL
        self._seq_len = seq_len
        self._height = cfg.HKO.ITERATOR.HEIGHT
        self._width = cfg.HKO.ITERATOR.WIDTH
        self._stride = stride
        self._max_consecutive_missing = max_consecutive_missing
        self._base_freq = base_freq
        self._base_time_delta = pd.Timedelta(base_freq)
        self._batch_size = batch_size
        self.frame_dat = np.zeros((self._seq_len, batch_size, 5 if self._channel != 1 else 1,
                                   self._height, self._width), dtype=np.uint8 if self._channel == 1 else np.float32)
        self.mask_dat = np.zeros((self._seq_len, batch_size, 1, self._height, self._width), dtype=np.bool)

        assert sample_mode in ["random", "sequent"], "Sample mode=%s is not supported" % sample_mode
        self.sample_mode = sample_mode
        if sample_mode == "sequent":
            assert self._stride is not None
            self._current_datetime = self.begin_time

    def set_begin_end(self, begin_ind=None, end_ind=None):
        self._begin_ind = 0 if begin_ind is None else begin_ind
        self._end_ind = self.total_frame_num - 1 if end_ind is None else end_ind

    @property
    def total_frame_num(self):
        return self._df.size

    @property
    def begin_time(self):
        return self._df.index[self._begin_ind]

    @property
    def end_time(self):
        return self._df.index[self._end_ind]

    @property
    def use_up(self):
        if self.sample_mode == "random":
            return False
        else:
            return self._current_datetime > self.end_time

    def _next_exist_timestamp(self, timestamp):
        next_ind = bisect.bisect_right(self._df.index, timestamp)
        if next_ind >= self._df.size:
            return None
        else:
            return self._df.index[bisect.bisect_right(self._df.index, timestamp)]

    def _is_valid_clip(self, datetime_clip):
        """Check if the given datetime_clip is valid

        Parameters
        ----------
        datetime_clip :

        Returns
        -------
        ret : bool
        """
        missing_count = 0
        if datetime_clip is None:
            return False
        for i in range(len(datetime_clip)):
            if datetime_clip[i] not in self._df_index_set_all:
                # print(datetime_clip[i])
                missing_count += 1
                if missing_count > self._max_consecutive_missing or \
                        missing_count >= len(datetime_clip):
                    return False
            else:
                missing_count = 0
        return True

    def _load_frames(self, datetime_clips):
        assert isinstance(datetime_clips, list)
        for clip in datetime_clips:
            assert len(clip) == self._seq_len
        batch_size = len(datetime_clips)

        if batch_size < self._batch_size:
            return None, None
        if self.sample_mode == "random":
            paths, hit_inds, miss_inds = [], [], []
            for i in range(batch_size):
                for j in range(self._seq_len):
                    timestamp = datetime_clips[i][j]
                    if timestamp in self._df_index_set_all:
                        paths.append(convert_datetime_to_filepath(timestamp))
                        hit_inds.append([j, i])
                    else:
                        miss_inds.append([j, i])
            all_mask_dat = self._exclude_mask[np.newaxis, ...].repeat(self._seq_len*batch_size, axis=0)
            hit_inds = np.array(hit_inds, dtype=np.int)
            all_frame_dat = image.quick_read_frames(path_list=paths,
                                                    im_h=self._height,
                                                    im_w=self._width,
                                                    channel=self._channel)
            self.frame_dat[hit_inds[:, 0], hit_inds[:, 1], :, :, :] = all_frame_dat
            self.mask_dat[hit_inds[:, 0], hit_inds[:, 1], :, :, :] = all_mask_dat

        else:
            count = 0
            pending_key = {}
            paths = []

            for i in range(self._seq_len):
                for j in range(batch_size):
                    timestamp = datetime_clips[j][i]
                    if timestamp in self._df_index_set_all and timestamp not in pending_key:
                        pending_key[timestamp] = count
                        count += 1
                        paths.append(convert_datetime_to_filepath(timestamp))

            pending_frame = image.quick_read_frames(path_list=paths,
                                                    im_h=self._height,
                                                    im_w=self._width,
                                                    channel=self._channel)
            pending_mask = self._exclude_mask
            for i in range(self._seq_len):
                for j in range(batch_size):
                    timestamp = datetime_clips[j][i]
                    if timestamp in self._df_index_set_all:
                        assert timestamp in pending_key
                        ind = pending_key[timestamp]
                        self.frame_dat[i, j, :, :, :] = pending_frame[ind, :, :, :]
                        self.mask_dat[i, j, :, :, :] = pending_mask

        if self._channel == 1:
            data_batch = self.frame_dat.astype(np.float32) / 255.0
        else:
            data_batch = self.frame_dat

            # [0.021, 22.794, 32.968, 45.279]
            # [0.004, -2.474, -56.426, -43.313]
            data_batch[:, :, 4, :, :] /= 255.0
            data_batch[:, :, 3, :, :] = (data_batch[:, :, 3, :, :] - 0.004) / 0.017
            data_batch[:, :, 2, :, :] = (data_batch[:, :, 2, :, :] + 2.474) / 25.268
            data_batch[:, :, 1, :, :] = (data_batch[:, :, 1, :, :] + 56.426) / 89.394
            data_batch[:, :, 0, :, :] = (data_batch[:, :, 0, :, :] + 43.313) / 88.592
            if self._channel == 3:
                data_batch = np.concatenate([np.sqrt(np.square(data_batch[:, :, 0:1, ...]) +
                                                     np.square(data_batch[:, :, 1:2, ...])),
                                             data_batch[:, :, 2:3, ...], data_batch[:, :, 4:5, ...]], axis=2)

        return data_batch, self.mask_dat

    def reset(self, begin_ind=None, end_ind=None):
        assert self.sample_mode == "sequent"
        self.set_begin_end(begin_ind=begin_ind, end_ind=end_ind)
        self._current_datetime = self.begin_time

    def random_reset(self):
        assert self.sample_mode == "sequent"
        self.set_begin_end(begin_ind=np.random.randint(0,
                                                       self.total_frame_num -
                                                       5 * self._seq_len),
                           end_ind=None)
        self._current_datetime = self.begin_time

    def check_new_start(self):
        assert self.sample_mode == "sequent"
        datetime_clip = pd.date_range(start=self._current_datetime,
                                      periods=self._seq_len,
                                      freq=self._base_freq)
        if self._is_valid_clip(datetime_clip):
            return self._current_datetime == self.begin_time
        else:
            return True

    def sample(self, batch_size, only_return_datetime=False):
        """
        Sample a minibatch from the hko7 dataset based on the given type and pd_file
        Parameters
        ----------
        batch_size : int
            Batch size
        only_return_datetime : bool
            Whether to only return the datetimes
        Returns
        -------
        frame_dat : np.ndarray
            Shape: (seq_len, valid_batch_size, 1, height, width)
        mask_dat : np.ndarray
            Shape: (seq_len, valid_batch_size, 1, height, width)
        datetime_clips : list
            length should be valid_batch_size
        new_start : bool
        """

        if self.sample_mode == 'sequent':
            if self.use_up:
                raise ValueError("The HKOIterator has been used up!")
            datetime_clips = []
            new_start = False
            for i in range(batch_size):
                while not self.use_up:
                    datetime_clip = pd.date_range(start=self._current_datetime,
                                                  periods=self._seq_len,
                                                  freq=self._base_freq)

                    if self._is_valid_clip(datetime_clip):

                        new_start = new_start or (self._current_datetime == self.begin_time)
                        datetime_clips.append(datetime_clip)
                        self._current_datetime += self._stride * self._base_time_delta
                        break
                    else:
                        new_start = True
                        self._current_datetime = \
                            self._next_exist_timestamp(timestamp=self._current_datetime)
                        if self._current_datetime is None:
                            # This indicates that there is no timestamp left,
                            # We point the current_datetime to be the next timestamp of self.end_time
                            self._current_datetime = self.end_time + self._base_time_delta
                            break
                        continue
            new_start = None if batch_size != 1 else new_start
            if only_return_datetime:
                return datetime_clips, new_start
        else:
            assert only_return_datetime is False
            datetime_clips = []
            new_start = None
            for i in range(batch_size):
                while True:
                    rand_ind = np.random.randint(0, self._df.size, 1)[0]
                    random_datetime = self._df.index[rand_ind]

                    datetime_clip = pd.date_range(start=random_datetime,
                                                  periods=self._seq_len,
                                                  freq=self._base_freq)

                    if self._is_valid_clip(datetime_clip):
                        datetime_clips.append(datetime_clip)
                        break
        frame_dat, mask_dat = self._load_frames(datetime_clips=datetime_clips)
        return frame_dat, mask_dat, datetime_clips, new_start
