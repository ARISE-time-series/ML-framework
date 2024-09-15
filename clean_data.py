#%%
import os
import re
from argparse import ArgumentParser
import numpy as np
import pandas as pd


#%%
def clean_pulse(outdir, data_dir='../data/extra/pulse'):
    os.makedirs(outdir, exist_ok=True)
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, file) for file in files if file.endswith('.xlsx')]

    for i, file_path in enumerate(files):
        df = pd.read_excel(file_path, sheet_name=None)
        df = pd.concat(df.values(), ignore_index=True)
        print(f'Loading {file_path}...')
        num_seconds = int(df['Time'].max())
        print(f'Mean of {file_path}: {df["Filtered Data"].mean()}')
        print(f'Std of {file_path}: {df["Filtered Data"].std()}')
        # for seg_id in range(num_seconds):
        #     seg = df.loc[df['Time'].between(seg_id, seg_id + 1)]
        #     if seg['Filtered Data'].sum() < 0.001:
        #         print(f'Found zero data in {file_path} at segment {seg_id}')
        #         # remove the segment
        #         df.drop(seg.index, inplace=True)
        df.to_csv(os.path.join(outdir, f'Pulse-cleaned_{i}.csv'), index=False)

#%%
def clean_emg(outdir, data_dir='../data/extra/emg', mean=2.5, std=0.123):
    os.makedirs(outdir, exist_ok=True)
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, file) for file in files if file.endswith('.xlsx')]
    print(f'Found {len(files)} files')
    for i, file_path in enumerate(files):
        df = pd.read_excel(file_path, sheet_name=None)
        df = pd.concat(df.values(), ignore_index=True)
        print(f'Mean of {file_path}: {df["ECG Raw Data"].mean()}')
        print(f'Std of {file_path}: {df["ECG Raw Data"].std()}')
        # normalize the data
        df['ECG Raw Data'] = (df['ECG Raw Data'] - mean) / std
        df.to_csv(os.path.join(outdir, f'EMG-cleaned_{i}.csv'), index=False)
        
# %%
def normalize_emg(root, outdir, mean=2.5, std=0.123):
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
clean_emg(outdir='../fft-data/extra', data_dir='../unlabel/EMG-unlabeled')
#%%
clean_pulse(outdir='../fft-data/extra', data_dir='../unlabel/Pulse-unlabeled')
#%%
normalize_emg('../data', '../fft-data/emg-pulse')
#%%
clean_pulse2('../data', '../fft-data/emg-pulse')

# %%

def extract_info(file_name):
    match = re.match(r'(\d+)_([a-zA-Z]+)\s([a-zA-Z\s]+)\.xlsx', file_name)
    if match:
        number = int(match.group(1))
        name = match.group(2)
        return number, name
    else:
        raise ValueError(f'Invalid file name: {file_name}')

def clean_emg_test(data_dir, outdir, type='EMG'):
    os.makedirs(outdir, exist_ok=True)
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, file) for file in files if file.endswith(f'{type}.xlsx')]
    for file_path in files:
        filename = os.path.basename(file_path)
        number, name = extract_info(filename)
        print(f'Extracted info: {number}, {name}')
        df = pd.read_excel(file_path, sheet_name=None)
        df = pd.concat(df.values(), ignore_index=True)
        print(df.columns)
        print(f'Mean of {file_path}: {df["ECG Raw Data"].mean()}')
        print(f'Std of {file_path}: {df["ECG Raw Data"].std()}')
        # normalize the data
        df['ECG Raw Data'] = (df['ECG Raw Data'] - 2.5) / 0.123
        df.to_csv(os.path.join(outdir, f'Subject-10-{number}-{name}-EMG.csv'), index=False)
        print(f'Saved to {outdir}/Subject-10-{number}-{name}-EMG.csv')

def clean_pulse_test(data_dir, outdir):
    os.makedirs(outdir, exist_ok=True)
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, file) for file in files if file.endswith('Pulse data.xlsx')]
    for file_path in files:
        filename = os.path.basename(file_path)
        number, name = extract_info(filename)
        print(f'Extracted info: {number}, {name}')
        df = pd.read_excel(file_path, sheet_name=None)
        df = pd.concat(df.values(), ignore_index=True)
        print(df.columns)
        print(f'Mean of {file_path}: {df["Filtered Data"].mean()}')
        print(f'Std of {file_path}: {df["Filtered Data"].std()}')
        df.to_csv(os.path.join(outdir, f'Subject-10-{number}-{name}-Pulse_data.csv'), index=False)
        print(f'Saved to {outdir}/Subject-10-{number}-{name}-Pulse_data.csv')

# %%

clean_emg_test('../fft-data/fig3test', outdir='../fft-data/vae-test', type='EMG')
# %%
clean_pulse_test('../fft-data/fig3test', outdir='../fft-data/vae-test')
# %%
