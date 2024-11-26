import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from scipy.signal import medfilt

from sklearn.preprocessing import StandardScaler
from .preprocess import z_score, get_array
from utils.timefeatures import time_features

_mean_std_dict = {
    'Lactate': (8.0, 1.0),
    'Na': (40.0, 6.0),
    'K': (5.0, 0.5),
    'Current (uAmps)': (6.0, 2.0),
    'Temperature (°C)': (32.0, 0.5),
}


def fourier_embedding(x, num_channels, max_positions=10000):
    '''
    x: N x C ndarray
    num_channels: int, number of channels to encode
    sin, cost embedding
    '''
    freqs = np.arange(0, num_channels // 2, dtype=np.float32) / (num_channels / 2)
    freqs = (1 / max_positions) ** freqs

    emb = np.outer(x, freqs)  # NxC x (num_channels // 2)
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1).reshape(x.shape[0], -1)
    return emb



def truncate(data, feat='Time (Seconds)', threshold=3600):
    '''
    data: DataFrame
    '''
    data = data[data[feat] < threshold]
    return data


def rm_power_freq(data, freq=60):
    '''
    remove the powerline frequency
    '''
    fft_results = np.fft.rfft(data)
    # freqs = [freq * i for i in range(1, fft_results.shape[0] // freq + 1)]
    # for fr in freqs:
        # fft_results[fr] = 0.0
    fft_results[freq] = 0.0
    # fft_results[freq * 2] = 0.0
    filtered_signal = np.fft.irfft(fft_results)
    return filtered_signal


class EMGDataset(Dataset):
    def __init__(self, root, subject_list, activities,
                 extra_files=None):
        self.root = root
        self.df_list = []
        self.sample_freq = 500

        self.borders = [0]
        for subject_id in subject_list:
            for act_id, act in activities.items():
                print(f'Loading Subject {subject_id} {act} EMG data...')
                file_path = os.path.join(root, f'Subject-{subject_id}-{act_id+1}-{act}-EMG.csv')
                df = pd.read_csv(file_path)
                num_seconds = int(df['Time (Seconds)'].max())
                df = truncate(df, threshold=num_seconds)
                self.df_list.append(df)
                self.borders.append(self.borders[-1] + num_seconds)
        
        if extra_files:
            for file_path in extra_files:
                print(f'Loading extra EMG data from {file_path}...')
                df = pd.read_csv(file_path)
                num_seconds = int(df['Time (Seconds)'].max())
                df = truncate(df, threshold=num_seconds)
                self.df_list.append(df)
                self.borders.append(self.borders[-1] + num_seconds)

    def __len__(self):
        return self.borders[-1]

    def __getitem__(self, idx):
        df_id = np.searchsorted(self.borders, idx, side='right') - 1
        seg_id = idx - self.borders[df_id]

        df = self.df_list[df_id]
        seg = df.loc[df['Time (Seconds)'].between(seg_id, seg_id + 1)]
        clean_seg = rm_power_freq(seg['ECG Raw Data'].values)
        if clean_seg.shape[0] < self.sample_freq:
            out = np.pad(clean_seg, (0, self.sample_freq - clean_seg.shape[0]), 'edge')
        else:
            out = clean_seg[:self.sample_freq]
        return torch.from_numpy(out).float().reshape(1, -1)

    def unnormalize(self, data):
        return data * 0.123 + 2.5
    

class PulseData(Dataset):
    def __init__(self, root, subject_list, activities, extra_files=None):
        self.root = root
        self.df_list = []
        self.mean = 0.5
        self.std = 0.5
        self.sample_freq = 140
        self.borders = [0]
        for subject_id in subject_list:
            for act_id, act in activities.items():
                print(f'Loading Subject {subject_id} {act} Pulse data...')
                file_path = os.path.join(root, f'Subject-{subject_id}-{act_id+1}-{act}-Pulse_data.csv')
                df = pd.read_csv(file_path)
                num_seconds = int(df['Time'].max()) - 1
                df = truncate(df, feat='Time', threshold=num_seconds)
                self.df_list.append(df)
                self.borders.append(self.borders[-1] + num_seconds)

        if extra_files:
            for file_path in extra_files:
                print(f'Loading extra Pulse data from {file_path}...')
                df = pd.read_csv(file_path)
                num_seconds = int(df['Time'].max()) - 1
                df = truncate(df, feat='Time', threshold=num_seconds)
                self.df_list.append(df)
                self.borders.append(self.borders[-1] + num_seconds)
                
    def __len__(self):
        return self.borders[-1]

    def __getitem__(self, idx):
        df_id = np.searchsorted(self.borders, idx, side='right') - 1
        seg_id = idx - self.borders[df_id]
        df = self.df_list[df_id]
        seg = df.loc[df['Time'].between(seg_id, seg_id + 1)]
        zscores = z_score(seg['Filtered Data'].values, self.mean, self.std)
        clean_seg = zscores
        
        if clean_seg.shape[0] < self.sample_freq:
            out = np.pad(clean_seg, (0, self.sample_freq - clean_seg.shape[0]), 'edge')
        else:
            out = clean_seg[:self.sample_freq]

        return torch.from_numpy(out).float().reshape(1, -1)

    def unnormalize(self, data):
        return data * self.std + self.mean
    

class Dataset_CLS_encoded(Dataset):
    def __init__(self, root_path,
                 subjects=[1, 2, 3, 4], 
                 acts=['Biking', 'VR', 'Hand grip', 'Stroop'],
                 cols=None,
                 encode_dir='Encoded',
                 flag='train', 
                 size=None, 
                 scale=False, 
                 embedding=None,
                 timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.cols = cols
        self.encode_dir = encode_dir
        self.subject_list = subjects
        self.activities = acts
        self.act2label = {key: value for value, key in enumerate(self.activities)}
        self.num_classes = len(self.activities)
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        # flag = flag.lower()
        # assert flag in ['train', 'test', 'val']

        self.scale = scale
        self.embedding = embedding
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        # if scaler_path:
            # self.scaler = load(scaler_path)
            # print('Loaded scaler from', scaler_path)
        self.build_data()

    def build_data(self):
        self.feat_arr = []
        self.target_arr = []
        self.stamp_arr = []
        self.borders = [-0.5]

        for subject_id in self.subject_list:
            for act in self.activities:
                feat, stamp = self.__read_data__(subject_id=subject_id, act=act)
                self.feat_arr.append(feat)
                self.target_arr.append(self.act2label[act])
                self.stamp_arr.append(stamp)
                self.borders.append(self.borders[-1] + len(feat) - self.seq_len - self.label_len + 1)


    def __read_data__(self, subject_id, act):
        datapath = os.path.join(self.root_path, f'Subject_{subject_id}-cleaned-{act}.csv')
        emg_path = os.path.join(self.encode_dir, f'Subject_{subject_id}-cleaned-{act}-EMG.csv')
        pulse_path = os.path.join(self.encode_dir, f'Subject_{subject_id}-cleaned-{act}-Pulse.csv')
        df_raw = pd.read_csv(datapath)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # columns = ['date', 'Lactate', 'Na', 'K', 'Current (uAmps)', 'Temperature (°C)', 'Fatigue level']
        columns = self.cols if self.cols else df_raw.columns
        df_data = df_raw[columns]

        border1 = 0
        border2 = len(df_data)
        
        # cols_data = df_raw.columns[1:-1]    # remove date and target
        # apply median filter to all the columns
        df_data = df_data.apply(lambda x: medfilt(x, kernel_size=5))
        if self.scale:
            df_data = df_data.apply(lambda x: z_score(x, *_mean_std_dict[x.name]))
        if self.embedding == 'fourier':
            embeded_data = fourier_embedding(df_data.values, num_channels=8)
        else:
            embeded_data = df_data.values
        # load EMG and Pulse
        df_emg = pd.read_csv(emg_path)
        emg_arr = get_array(df_emg)
        df_pulse = pd.read_csv(pulse_path)
        pulse_arr = get_array(df_pulse)

        data = np.concatenate([embeded_data, emg_arr, pulse_arr], axis=1)
        print(f'Loaded Subject: {subject_id}, act: {act}, data shape: {data.shape}')

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(data)
            data = self.scaler.transform(data)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['second'] = df_stamp.date.apply(lambda row: row.second, 1)
            data_stamp = df_stamp.drop(columns=['date']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        data_x = data[border1:border2]
        return data_x, data_stamp

    def __getitem__(self, index):
        series_idx = np.searchsorted(self.borders, index, side="right") - 1
        idx = index - int(self.borders[series_idx] + 0.5)

        data_x = self.feat_arr[series_idx]
        label = self.target_arr[series_idx]
        data_stamp = self.stamp_arr[series_idx]

        s_begin = idx + self.label_len
        s_end = s_begin + self.seq_len

        seq_x = data_x[s_begin:s_end]

        seq_x_mark = data_stamp[s_begin:s_end]

        return seq_x, label, seq_x_mark

    def __len__(self):
        return int(self.borders[-1]+0.5)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_CLS_manual(Dataset):
    def __init__(self, root_path,
                 subjects=[1, 2, 3, 4], 
                 cols=None,
                 encode_dir=None,   # path to EMG and Pulse clean data
                 flag='train', 
                 size=None, 
                 scale=False, 
                 embedding=None,
                 timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.cols = cols
        self.encode_dir = encode_dir
        self.subject_list = subjects
        self.activities = ['Biking', 'VR', 'Hand grip', 'Stroop']
        self.act2label = {key: value for value, key in enumerate(self.activities)}
        self.num_classes = len(self.activities)
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        flag = flag.lower()
        assert flag in ['train', 'test', 'val']

        self.scale = scale
        self.embedding = embedding
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.build_data()

    def build_data(self):
        self.feat_arr = []
        self.target_arr = []
        self.stamp_arr = []
        self.borders = [-0.5]

        for subject_id in self.subject_list:
            for act in self.activities:
                feat, stamp = self.__read_data__(subject_id=subject_id, act=act)
                self.feat_arr.append(feat)
                self.target_arr.append(self.act2label[act])
                self.stamp_arr.append(stamp)
                self.borders.append(self.borders[-1] + len(feat) - self.seq_len - self.label_len + 1)


    def __read_data__(self, subject_id, act):
        datapath = os.path.join(self.root_path, f'Subject_{subject_id}-cleaned-{act}.csv')
        emg_path = os.path.join(self.root_path, self.encode_dir, f'Subject_{subject_id}-cleaned-{act}-EMG.csv')
        pulse_path = os.path.join(self.root_path, self.encode_dir, f'Subject_{subject_id}-cleaned-{act}-Pulse.csv')
        df_raw = pd.read_csv(datapath)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # columns = ['date', 'Lactate', 'Na', 'K', 'Current (uAmps)', 'Temperature (°C)', 'Fatigue level']
        columns = self.cols if self.cols else df_raw.columns
        df_data = df_raw[columns]

        border1 = 0
        border2 = len(df_data)
        
        # cols_data = df_raw.columns[1:-1]    # remove date and target
        # apply median filter to all the columns
        df_data = df_data.apply(lambda x: medfilt(x, kernel_size=5))
        if self.scale:
            df_data = df_data.apply(lambda x: z_score(x, *_mean_std_dict[x.name]))
        if self.embedding == 'fourier':
            embeded_data = fourier_embedding(df_data.values, num_channels=8)
        else:
            embeded_data = df_data.values
        # load EMG and Pulse
        df_emg = pd.read_csv(emg_path)
        emg_arr = get_array(df_emg)
        df_pulse = pd.read_csv(pulse_path)
        pulse_arr = get_array(df_pulse)

        data = np.concatenate([embeded_data, emg_arr, pulse_arr], axis=1)
        print(f'Loaded Subject: {subject_id}, act: {act}, data shape: {data.shape}')

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(data)
            data = self.scaler.transform(data)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            # df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['second'] = df_stamp.date.apply(lambda row: row.second, 1)
            data_stamp = df_stamp.drop(columns=['date']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        data_x = data[border1:border2]
        return data_x, data_stamp

    def __getitem__(self, index):
        series_idx = np.searchsorted(self.borders, index, side="right") - 1
        idx = index - int(self.borders[series_idx] + 0.5)

        data_x = self.feat_arr[series_idx]
        label = self.target_arr[series_idx]
        data_stamp = self.stamp_arr[series_idx]

        s_begin = idx + self.label_len
        s_end = s_begin + self.seq_len

        seq_x = data_x[s_begin:s_end]

        seq_x_mark = data_stamp[s_begin:s_end]

        return seq_x, label, seq_x_mark

    def __len__(self):
        return int(self.borders[-1]+0.5)
    


class Dataset_IMP_encoded(Dataset):
    def __init__(self, root_path,
                 subjects=[1, 2, 3, 4], 
                 acts=['Biking', 'VR', 'Hand grip', 'Stroop'],
                 cols=None,
                 encode_dir=None,   # path to EMG and Pulse clean data
                 flag='train', size=None,
                 scale=False, 
                 embedding=None,
                 timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.cols = cols
        self.encode_dir = encode_dir
        self.subject_list = subjects
        self.activities = acts

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        assert self.seq_len == self.pred_len

        self.target = 'Fatigue level'
        self.scale = scale
        self.embedding = embedding
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.build_data()

    def build_data(self):
        self.feat_arr = []
        self.target_arr = []
        self.stamp_arr = []
        self.borders = [-0.5]

        for subject_id in self.subject_list:
            for act in self.activities:
                feat, target, stamp = self.__read_data__(subject_id=subject_id, act=act)
                self.feat_arr.append(feat)
                self.target_arr.append(target)
                self.stamp_arr.append(stamp)
                self.borders.append(self.borders[-1] + len(feat) - self.seq_len - self.label_len + 1)

    def __read_data__(self, subject_id, act):
        datapath = os.path.join(self.root_path, f'Subject_{subject_id}-cleaned-{act}.csv')
        emg_path = os.path.join(self.encode_dir, f'Subject_{subject_id}-cleaned-{act}-EMG.csv')
        pulse_path = os.path.join(self.encode_dir, f'Subject_{subject_id}-cleaned-{act}-Pulse.csv')
        df_raw = pd.read_csv(datapath)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        columns = self.cols if self.cols else df_raw.columns 
        # ['date', 'Current (uAmps)', 'Temperature (°C)', 'Fatigue level']
        df_data = df_raw[columns]
        
        border1 = 0
        border2 = len(df_raw)
        # apply median filter to all the columns
        df_data = df_data.apply(lambda x: medfilt(x, kernel_size=5))
        if self.embedding == 'fourier':
            embeded_data = fourier_embedding(df_data.values, num_channels=8)
        else:
            embeded_data = df_data.values

        # load EMG and Pulse
        df_emg = pd.read_csv(emg_path)
        emg_arr = get_array(df_emg)
        df_pulse = pd.read_csv(pulse_path)
        pulse_arr = get_array(df_pulse)

        data = np.concatenate([embeded_data, emg_arr, pulse_arr], axis=1)
        print(f'Loaded Subject: {subject_id}, act: {act}, data shape: {data.shape}')

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(data)
            data = self.scaler.transform(data)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['second'] = df_stamp.date.apply(lambda row: row.second, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        data_x = data[border1:border2]
        data_y = df_raw[[self.target]].values[border1:border2]
        return data_x, data_y, data_stamp

    def __getitem__(self, index):
        series_idx = np.searchsorted(self.borders, index, side="right") - 1
        idx = index - int(self.borders[series_idx] + 0.5)

        data_x = self.feat_arr[series_idx]
        data_y = self.target_arr[series_idx]
        data_stamp = self.stamp_arr[series_idx]

        s_begin = idx + self.label_len
        s_end = s_begin + self.seq_len

        r_begin = idx
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data_x[s_begin:s_end]
        # seq_y = self.data_x[r_begin:r_end]
        target_y = data_y[r_begin:r_end]

        seq_x_mark = data_stamp[s_begin:s_end]
        seq_y_mark = data_stamp[r_begin:r_end]

        return seq_x, target_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return int(self.borders[-1]+0.5)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    


class Dataset_IMP_Pred_encoded(Dataset):
    def __init__(self, root_path, 
                 encode_dir=None,
                 size=None, flag='pred',
                 subject=5,
                 cols=None, 
                 act='Biking',
                 scale=False,
                 embedding=None, 
                 inverse=False, 
                 timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
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

        self.target = 'Fatigue level'
        self.scale = scale
        self.embedding = embedding
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.encode_dir = encode_dir

        self.__read_data__(subject, act)

    def __read_data__(self, subject, act):
        datapath = os.path.join(self.root_path, f'Subject_{subject}-cleaned-{act}.csv')
        df_raw = pd.read_csv(datapath)

        emg_path = os.path.join(self.encode_dir, f'Subject_{subject}-cleaned-{act}-EMG.csv')
        pulse_path = os.path.join(self.encode_dir, f'Subject_{subject}-cleaned-{act}-Pulse.csv')
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        columns = self.cols if self.cols else df_raw.columns 
        # ['date', 'Current (uAmps)', 'Temperature (°C)', 'Fatigue level']
        df_data = df_raw[columns]

        border1 = 0              # len(df_raw) - self.seq_len
        border2 = len(df_raw)

        df_data = df_data.apply(lambda x: medfilt(x, kernel_size=5))

        if self.embedding == 'fourier':
            embeded_data = fourier_embedding(df_data.values, num_channels=8)
        else:
            embeded_data = df_data.values

        # load EMG and Pulse
        df_emg = pd.read_csv(emg_path)
        emg_arr = get_array(df_emg)
        df_pulse = pd.read_csv(pulse_path)
        pulse_arr = get_array(df_pulse)

        data = np.concatenate([embeded_data, emg_arr, pulse_arr], axis=1)

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(data)
            data = self.scaler.transform(data)

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        # pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns=['date'])
        
        df_stamp.date = list(tmp_stamp.date.values)# + list(pred_dates[1:])
        # self.future_dates = list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['second'] = df_stamp.date.apply(lambda row: row.second, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = df_raw[[self.target]].values[border1:border2]
        # if self.inverse:
        #     self.data_y = df_data.values[border1:border2]
        # else:
        #     self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = self.label_len + index * self.pred_len
        s_end = s_begin + self.seq_len

        r_begin = s_begin - self.label_len
        r_end = s_end

        seq_x = self.data_x[s_begin:s_end]

        target_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, target_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x - self.label_len) // self.pred_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)