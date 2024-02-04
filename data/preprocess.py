import os
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter

##------------------ Helper functions for preprocessing ----------------- ##

def get_array(df, length=3600):
    cols = list(df.columns)
    df = df[cols[1:]]
    arr = df.values
    if arr.shape[0] < length:
        arr = np.pad(arr, ((0, length - arr.shape[0]), (0, 0)), 'edge')
    else:
        arr = arr[:length]
    return arr


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


def z_score(arr, mean=None, std=None):
    if mean is None or std is None:
        mean = arr.mean()
        std = arr.std()
    return arr / std - mean / std


def sub_mean(arr, mean=None):
    if mean is None:
        mean = arr.mean()
    return arr - mean


def read_data(data_path):
    df_raw = pd.read_csv(data_path)
    num_train = int(len(df_raw) * 0.9)

    cols_data = df_raw.columns[1:-1]    # remove date and target
    df_data = df_raw[cols_data]

    return df_data[0:num_train].values

## ----------------- For testing purpose ----------------- ##
def train_scaler(outdir):
    subject_list = [1, 2, 3, 4]
    activities = ['Biking', 'VR', 'Hand grip', 'Stroop']
    data_root = '../fatigue/data'
    scaler = StandardScaler()
    feat_list = []
    for subject_id in subject_list:
        for act in activities:
            data_path = os.path.join(data_root, f'Subject_{subject_id}-cleaned-{act}.csv')
            data = read_data(data_path)
            feat_list.append(data)
    feats = np.concatenate(feat_list, axis=0)
    scaler.fit(feats)
    scaler_path = os.path.join(outdir, f'train_scaler.joblib')    
    dump(scaler, scaler_path)
    print(scaler.mean_, scaler.scale_)


if __name__ == '__main__':
    outdir = 'assets'
    os.makedirs(outdir, exist_ok=True)
    train_scaler(outdir)
    scaler = load(os.path.join(outdir, 'train_scaler.joblib'))
    print(scaler.mean_, scaler.scale_)

