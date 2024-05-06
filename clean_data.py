#%%
import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd

#%%
def clean_pulse(data_dir='../data/extra/pulse'):
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, file) for file in files if file.endswith('.xlsx')]

    for i, file_path in enumerate(files):
        df = pd.read_excel(file_path, sheet_name=None)
        df = pd.concat(df.values(), ignore_index=True)
        print(f'Loading {file_path}...')
        num_seconds = int(df['Time'].max())
        # for seg_id in range(num_seconds):
        #     seg = df.loc[df['Time'].between(seg_id, seg_id + 1)]
        #     if seg['Filtered Data'].sum() < 0.001:
        #         print(f'Found zero data in {file_path} at segment {seg_id}')
        #         # remove the segment
        #         df.drop(seg.index, inplace=True)
        df.to_csv(os.path.join(data_dir, f'cleaned_{i}.csv'), index=False)

#%%
def clean_emg(data_dir='../data/extra/emg', mean=1.47, std=0.123):
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, file) for file in files if file.endswith('.xlsx')]
    for i, file_path in enumerate(files):
        df = pd.read_excel(file_path, sheet_name=None)
        df = pd.concat(df.values(), ignore_index=True)
        print(f'Mean of {file_path}: {df["ECG Raw Data"].mean()}')
        print(f'Std of {file_path}: {df["ECG Raw Data"].std()}')
        # normalize the data
        df['ECG Raw Data'] = (df['ECG Raw Data'] - mean) / std
        df.to_csv(os.path.join(data_dir, f'cleaned_{i}.csv'), index=False)
        
# %%
def normalize_emg(root, outdir, mean=2.49, std=0.123):
    os.makedirs(outdir, exist_ok=True)
    subject_list = range(1, 10)
    activities = {0: 'Stroop', 1: 'VR', 2: 'Hand grip', 3: 'Biking'}
    for subject_id in subject_list:
        for act_id, act in activities.items():
            file_path = os.path.join(root, f'Subject {subject_id}_cleaned', f'{act_id+1}_{act} EMG.xlsx')
            df = pd.read_excel(file_path, sheet_name=None)
            df = pd.concat(df.values(), ignore_index=True)
            print(f'Mean of {file_path}: {df["ECG Raw Data"].mean()}')
            print(f'Std of {file_path}: {df["ECG Raw Data"].std()}')
            # normalize the data
            df['ECG Raw Data'] = (df['ECG Raw Data'] - mean) / std
            df.to_csv(os.path.join(outdir, f'Subject-{subject_id}-{act_id+1}-{act}-EMG.csv'), index=False)
            print(f'Saved to {outdir}/Subject-{subject_id}-{act_id+1}-{act}-EMG.csv')
#%%
def clean_pulse2(root, outdir):
    os.makedirs(outdir, exist_ok=True)
    subject_list = range(1, 10)
    activities = {0: 'Stroop', 1: 'VR', 2: 'Hand grip', 3: 'Biking'}

    for subject_id in subject_list:
        for act_id, act in activities.items():
            file_path = os.path.join(root, f'Subject {subject_id}_cleaned', f'{act_id+1}_{act} Pulse data.xlsx')
            df = pd.read_excel(file_path, sheet_name=None)
            df = pd.concat(df.values(), ignore_index=True)

            num_seconds = int(df['Time'].max())
            # for seg_id in range(num_seconds):
                # seg = df.loc[df['Time'].between(seg_id, seg_id + 1)]
                # if seg['Filtered Data'].sum() < 0.001:
                #     print(f'Found zero data in {file_path} at segment {seg_id}')
                #     # remove the segment
                #     df.drop(seg.index, inplace=True)
            save_path = os.path.join(outdir, f'Subject-{subject_id}-{act_id+1}-{act}-Pulse_data.csv')
            df.to_csv(save_path, index=False)
# %%
normalize_emg('../data/fatigue', '../data/emg-pulse')
# %%
clean_pulse2('../data/fatigue', '../data/emg-pulse')
# %%
clean_pulse()
# %%
