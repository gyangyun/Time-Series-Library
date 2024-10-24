import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sktime.datasets import load_from_tsfile_to_dataframe
from torch.utils.data import DataLoader, Dataset

from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import Normalizer, interpolate_missing, subsample
from utils.augmentation import run_augmentation_single
from utils.timefeatures import time_features

warnings.filterwarnings("ignore")


class Dataset_ETT_hour(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        seasonal_patterns=None,
    ):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 - self.seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(
                lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values),
                                       freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(
                self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTm1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="t",
        seasonal_patterns=None,
    ):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 * 4 - self.seq_len,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(
                lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values),
                                       freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(
                self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        seasonal_patterns=None,
    ):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [
            0, num_train - self.seq_len,
            len(df_raw) - num_test - self.seq_len
        ]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(
                lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values),
                                       freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(
                self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="pred",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=False,
        inverse=False,
        timeenc=0,
        freq="15min",
        seasonal_patterns="Yearly",
    ):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == "train":
            dataset = M4Dataset.load(training=True,
                                     dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False,
                                     dataset_file=self.root_path)
        training_values = np.array([
            v[~np.isnan(v)]
            for v in dataset.values[dataset.groups == self.seasonal_patterns]
        ])  # split different frequencies
        self.ids = np.array(
            [i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros(
            (self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(
            low=max(1,
                    len(sampled_timeseries) - self.window_sampling_limit),
            high=len(sampled_timeseries),
            size=1,
        )[0]

        insample_window = sampled_timeseries[max(0, cut_point -
                                                 self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
            cut_point - self.label_len:min(len(sampled_timeseries), cut_point +
                                           self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, "train.csv"))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, "test.csv"))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(
            os.path.join(root_path, "test_label.csv")).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index +
                                         self.win_size]), np.float32(
                                             self.test_labels[0:self.win_size])
        elif self.flag == "val":
            return np.float32(self.val[index:index +
                                       self.win_size]), np.float32(
                                           self.test_labels[0:self.win_size])
        elif self.flag == "test":
            return np.float32(
                self.test[index:index + self.win_size]), np.float32(
                    self.test_labels[index:index + self.win_size])
        else:
            return np.float32(
                self.test[index // self.step *
                          self.win_size:index // self.step * self.win_size +
                          self.win_size]), np.float32(
                              self.test_labels[index // self.step *
                                               self.win_size:index //
                                               self.step * self.win_size +
                                               self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(
            os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index +
                                         self.win_size]), np.float32(
                                             self.test_labels[0:self.win_size])
        elif self.flag == "val":
            return np.float32(self.val[index:index +
                                       self.win_size]), np.float32(
                                           self.test_labels[0:self.win_size])
        elif self.flag == "test":
            return np.float32(
                self.test[index:index + self.win_size]), np.float32(
                    self.test_labels[index:index + self.win_size])
        else:
            return np.float32(
                self.test[index // self.step *
                          self.win_size:index // self.step * self.win_size +
                          self.win_size]), np.float32(
                              self.test_labels[index // self.step *
                                               self.win_size:index //
                                               self.step * self.win_size +
                                               self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(
            os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index +
                                         self.win_size]), np.float32(
                                             self.test_labels[0:self.win_size])
        elif self.flag == "val":
            return np.float32(self.val[index:index +
                                       self.win_size]), np.float32(
                                           self.test_labels[0:self.win_size])
        elif self.flag == "test":
            return np.float32(
                self.test[index:index + self.win_size]), np.float32(
                    self.test_labels[index:index + self.win_size])
        else:
            return np.float32(
                self.test[index // self.step *
                          self.win_size:index // self.step * self.win_size +
                          self.win_size]), np.float32(
                              self.test_labels[index // self.step *
                                               self.win_size:index //
                                               self.step * self.win_size +
                                               self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(
            os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index +
                                         self.win_size]), np.float32(
                                             self.test_labels[0:self.win_size])
        elif self.flag == "val":
            return np.float32(self.val[index:index +
                                       self.win_size]), np.float32(
                                           self.test_labels[0:self.win_size])
        elif self.flag == "test":
            return np.float32(
                self.test[index:index + self.win_size]), np.float32(
                    self.test_labels[index:index + self.win_size])
        else:
            return np.float32(
                self.test[index // self.step *
                          self.win_size:index // self.step * self.win_size +
                          self.win_size]), np.float32(
                              self.test_labels[index // self.step *
                                               self.win_size:index //
                                               self.step * self.win_size +
                                               self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, "swat_train2.csv"))
        test_data = pd.read_csv(os.path.join(root_path, "swat2.csv"))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index +
                                         self.win_size]), np.float32(
                                             self.test_labels[0:self.win_size])
        elif self.flag == "val":
            return np.float32(self.val[index:index +
                                       self.win_size]), np.float32(
                                           self.test_labels[0:self.win_size])
        elif self.flag == "test":
            return np.float32(
                self.test[index:index + self.win_size]), np.float32(
                    self.test_labels[index:index + self.win_size])
        else:
            return np.float32(
                self.test[index // self.step *
                          self.win_size:index // self.step * self.win_size +
                          self.win_size]), np.float32(
                              self.test_labels[index // self.step *
                                               self.win_size:index //
                                               self.step * self.win_size +
                                               self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """
    def __init__(self,
                 args,
                 root_path,
                 file_list=None,
                 limit_size=None,
                 flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path,
                                                    file_list=file_list,
                                                    flag=flag)
        self.all_IDs = (
            self.all_df.index.unique()
        )  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path,
                                                "*"))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception("No files found using: {}".format(
                os.path.join(root_path, "*")))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [
            p for p in data_paths if os.path.isfile(p) and p.endswith(".ts")
        ]
        if len(input_paths) == 0:
            pattern = "*.ts"
            raise Exception(
                "No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(
            input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(
            filepath,
            return_separate_X_and_y=True,
            replace_missing_vals_with="NaN")
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(
            labels.cat.codes, dtype=np.int8
        )  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)
        ).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if (np.sum(horiz_diffs) >
                0):  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if (np.sum(vert_diffs) > 0
            ):  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat(
            (pd.DataFrame({col: df.loc[row, col]
                           for col in df.columns
                           }).reset_index(drop=True).set_index(
                               pd.Series(lengths[row, 0] * [row]))
             for row in range(df.shape[0])),
            axis=0,
        )

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if (self.root_path.count("EthanolConcentration") >
                0):  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(
                torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(
                batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(
            torch.from_numpy(batch_x)), torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)


class Dataset_IE_day(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="MS",
        data_path="IEd1.csv",
        target="electricity_consumption",
        scale=False,
        timeenc=0,
        freq="d",
        seasonal_patterns=None,
    ):
        # size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 7 * 8
            self.label_len = 7 * 2
            self.pred_len = 7 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ["train", "test", "val", "pred"]
        type_map = {"train": 0, "val": 1, "test": 2, "pred": 3}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.args = args

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_pickle(os.path.join(self.root_path, self.data_path))

        if self.args.cols:
            cols = [col.strip() for col in self.args.cols.split()]
        else:
            cols = list(df_raw.columns)
        # 重新排列列的顺序
        if self.features in ["MS", "S"]:
            # 如果target为空，则默认使用cols中最后一列作为target
            if self.target == "":
                self.target = cols[-1]

            cols = [col for col in cols if col not in ["date", self.target]]
            df_raw = df_raw[["date"] + cols + [self.target]]
        else:
            # 如果target为空，则默认使用除了date外的所有列作为target
            if self.target == "":
                self.target = [col for col in cols if col != "date"]
            else:
                targets = [t.strip() for t in self.target.split()]
                self.target = targets

            cols = [col for col in cols if col not in self.target + ["date"]]
            df_raw = df_raw[["date"] + cols + self.target]

        if self.set_type != 3:
            df_raw["date"] = pd.to_datetime(df_raw["date"])
            df_raw.set_index("date", inplace=True)
            df_raw.sort_index(ascending=True, inplace=True)
            df_raw = df_raw[self.args.train_start:self.args.test_end]

            num_train = df_raw[self.args.train_start:self.args.
                               train_end].shape[0]
            num_test = df_raw[self.args.test_start:self.args.test_end].shape[0]
            num_vali = len(df_raw) - num_train - num_test

            df_raw.reset_index(drop=False, inplace=True)

            border1s = [
                0,
                num_train - self.seq_len,
                len(df_raw) - num_test - self.seq_len,
            ]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
        else:
            df_raw["date"] = pd.to_datetime(df_raw["date"])
            df_raw.sort_values(by="date", ascending=True, inplace=True)
            df_raw.reset_index(drop=True, inplace=True)

            if self.args.pred_start:
                pred_start_date = pd.to_datetime(self.args.pred_start)
            else:
                pred_start_date = df_raw[-self.args.pred_start]["date"]

            all_date_set = set(pd.to_datetime(df_raw["date"]).values)

            # 先校验数据集中是否有足够的数据预测pred_start
            pred_start_lookback_start_date = pred_start_date - pd.Timedelta(
                days=self.seq_len)
            pred_start_seq_len_set = set(
                pd.date_range(
                    start=pred_start_lookback_start_date,
                    periods=self.seq_len,
                    freq="D",
                ).values)
            if not pred_start_seq_len_set.issubset(all_date_set):
                raise ValueError(
                    f"数据集中没有{self.args.pred_start}往前{self.seq_len}个长度的历史数据以预测{self.args.pred_start}"
                )

            if not self.args.pred_end:
                pred_end_date = pred_start_date
            else:
                pred_end_date = pd.to_datetime(self.args.pred_end)

            # 如果不使用自回归
            if not self.args.use_autoregression:
                # 校验待预测的时间长度和模型能够预测的时间长度
                to_pred_len = (pred_end_date - pred_start_date).days + 1
                if to_pred_len > self.pred_len:
                    raise ValueError(
                        f"待预测长度【{to_pred_len}】，大于模型能预测长度【{self.pred_len}】，且不使用自回归的方式预测，无法支持该预测任务"
                    )

                border1 = df_raw[df_raw["date"] ==
                                 pred_start_lookback_start_date].index[0]
                # 因为不使用自回归，且模型能预测的长度大于待预测的长度
                # 所以只需要预测一次，让获取到的(x, y)中的y[0]为pred_start_date对应时间
                # 从而将x输入模型，生成的pred_y中的pred_y[0]也是pred_start_date对应时间
                # 在预测时从预测结果中截取出待预测结果就行
                border2_date = pred_start_date + pd.Timedelta(
                    days=self.pred_len)

                # 因为__len__()函数约束了每次都能获取seq_len长度的x和pred_len长度的y
                # 只有手动补全才能保障获取到的最后一组(x, y)中的y[0]为pred_start_date对应时间
                # 从而将x输入模型，生成的pred_y中的pred_y[0]也是pred_start_date对应时间
                need_dates = list(
                    pd.date_range(
                        start=pred_start_lookback_start_date,
                        end=border2_date,
                        freq="D",
                    ).values)
                for tmp_date in need_dates:
                    if tmp_date in all_date_set:
                        continue
                    df_raw.loc[len(df_raw)] = [None] * len(df_raw.columns)
                    df_raw.loc[len(df_raw) - 1, "date"] = tmp_date

                border2 = df_raw[df_raw["date"] == border2_date].index[0]
            # 如果使用自回归
            else:
                border1 = df_raw[df_raw["date"] ==
                                 pred_start_lookback_start_date].index[0]
                # 自回归的方式，无论pred_len是多长，每次只用第一个时间步长的预测结果
                # 所以需要保证最后一个样本的预测结果中第一个时间步长为为pred_end_date
                # 所以需要保证数据子集的边界能取到pred_end_date + pred_len
                border2_date = pred_end_date + pd.Timedelta(days=self.pred_len)

                # 因为__len__()函数约束了每次都能获取seq_len长度的x和pred_len长度的y
                # 只有手动补全才能保障获取到的最后一组(x, y)中的y[0]为pred_end_date对应时间
                # 从而将x输入模型，生成的pred_y中的pred_y[0]也是pred_end_date对应时间
                need_dates = list(
                    pd.date_range(
                        start=pred_start_lookback_start_date,
                        end=border2_date,
                        freq="D",
                    ).values)
                for tmp_date in need_dates:
                    if tmp_date in all_date_set:
                        continue
                    df_raw.loc[len(df_raw)] = [None] * len(df_raw.columns)
                    df_raw.loc[len(df_raw) - 1, "date"] = tmp_date

                border2 = df_raw[df_raw["date"] == border2_date].index[0]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(
                lambda row: row.weekday(), 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values),
                                       freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # 每个时间步长都增加次日天气，注意把wd_-1_shift, wd_max_-1_shift, wd_min_-1_shift放在除了date外的头三列
        # 获取需要复制的值
        values_to_append_x = seq_x[-1, :3]
        # 在seq_x的头部新增三列，并填充相同的值
        seq_x = np.hstack((np.tile(values_to_append_x,
                                   (seq_x.shape[0], 1)), seq_x))

        # 每个时间步长都增加次日天气，注意把wd_-1_shift, wd_max_-1_shift, wd_min_-1_shift放在除了date外的头三列
        # 获取需要复制的值
        values_to_append_y = seq_y[-self.pred_len, :3]
        # 在seq_y的头部新增三列，并填充相同的值
        seq_y = np.hstack((np.tile(values_to_append_y,
                                   (seq_y.shape[0], 1)), seq_y))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)