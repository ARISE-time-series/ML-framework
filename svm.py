import os
import random
from omegaconf import OmegaConf

import numpy as np
import pandas as pd
from scipy.signal import medfilt
import torch
import hydra

from data.preprocess import get_array
from sklearn.svm import SVC
from typing import List


def construct_data(root_path,           # Root path of the dataset
                   subjects: List[int], # List of subject ids
                   acts: List[str],     # List of activity names
                   encode_dir: str,     # Directory containing encoded data
                   seq_len: int,        # Sequence length
                   cols: List[str]):    # List of column names    
    act2label = {key: value for value,key in enumerate(acts)}

    X = []
    y = []
    for subject_id in subjects:
        for act in acts:
            other_path = os.path.join(root_path, f'Subject_{subject_id}-cleaned-{act}.csv')
            emg_path = os.path.join(encode_dir, f'Subject_{subject_id}-cleaned-{act}-EMG.csv')
            pulse_path = os.path.join(encode_dir, f'Subject_{subject_id}-cleaned-{act}-Pulse.csv')

            df_raw = pd.read_csv(other_path)
            df_data = df_raw[cols]
            # apply median filter to all the columns
            df_data = df_data.apply(lambda x: medfilt(x, kernel_size=5))

            other_data = df_data.values
            df_emg = pd.read_csv(emg_path)
            emg_arr = get_array(df_emg)
            df_pulse = pd.read_csv(pulse_path)
            pulse_arr = get_array(df_pulse)
            data = np.concatenate([other_data, emg_arr, pulse_arr], axis=1)
            data = np.reshape(data, (data.shape[0]//seq_len, seq_len * data.shape[1]))
            X.append(data)
            label = np.array([act2label[act]] * data.shape[0])
            y.append(label)
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config):
    torch.backends.cudnn.benchmark = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.task == 'forecast':
        cfg = config.forecast
    else:
        cfg = config.classification
    
    train_X, train_y = construct_data(cfg.data.root_path, subjects=cfg.data.train_subjects, acts=cfg.data.train_acts, 
                                      encode_dir=cfg.data.encode_dir, seq_len=cfg.model.seq_len, cols=cfg.data.cols)
    test_X, test_y = construct_data(cfg.data.root_path, subjects=cfg.data.test_subjects, acts=cfg.data.train_acts,
                                    encode_dir=cfg.data.encode_dir, seq_len=cfg.model.seq_len, cols=cfg.data.cols)
    root_dir = os.path.join(config.exp_dir, config.task)
    os.makedirs(root_dir, exist_ok=True)
    exp_dir = os.path.join(root_dir, f'{cfg.model.name}_'
                           f'seq{cfg.model.seq_len}_{config.tag}')
    os.makedirs(exp_dir, exist_ok=True)
    save_dir = os.path.join(exp_dir, 'results')
    os.makedirs(save_dir, exist_ok=True)

    model = SVC(C=cfg.model.reg, probability=True)
    model.fit(train_X, train_y)
    acc = model.score(test_X, test_y)
    print(f'Accuracy: {acc}')

    if config.test:
        # save probability predictions for ROC curve
        proba = model.predict_proba(test_X)
        activities = ['Biking', 'VR', 'Hand grip', 'Stroop']
        data_dict = {'labels': test_y}
        for i, act in enumerate(activities):
            data_dict[act] = proba[:, i]
        df = pd.DataFrame(data_dict)
        df.to_csv(os.path.join(save_dir, 'probsnlabel.csv'), index=False)
        log_file = os.path.join(save_dir, 'log.txt')
        with open(log_file, 'w') as f:
            f.write(f'Accuracy: {acc}')
    

if __name__ == '__main__':
    main()