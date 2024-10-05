#%%
import os
import numpy as np
import pandas as pd
#%%
data_dir = '../data'
out_dir = '../data/manual'
os.makedirs(out_dir, exist_ok=True)
act_dict = {
    'Stroop': 1, 
    'VR': 2,
    'Hand grip': 3,
    'Biking': 4,
}
subject_id = 3
act = 'Stroop'
#%%
act_id = act_dict[act]
emg_path = os.path.join(data_dir, f'Subject {subject_id}_cleaned', f'{act_id}_{act} EMG feature list.xlsx')
emg_df = pd.read_excel(emg_path, sheet_name=None)
emg_df = emg_df['Sheet1']
# %%
pulse_path = os.path.join(data_dir, f'Subject {subject_id}_cleaned', f'{act_id}_{act} Pulse feature list.xlsx')
pulse_df = pd.read_excel(pulse_path, sheet_name=None)
pulse_df = pulse_df['Sheet1']
# %%
num_seconds = 3600
emg_columns = ['Moving_Average', 'Maximum_Data']
print(emg_df.columns[1:])
pulse_columns = pulse_df.columns[1:]
print(pulse_columns)
# %%
for i in range(1, 10):
    for act, act_id in act_dict.items():
        emg_path = os.path.join(data_dir, f'Subject {i}_cleaned', f'{act_id}_{act} EMG feature list.xlsx')
        emg_df = pd.read_excel(emg_path, sheet_name=None)
        emg_df = emg_df['Sheet1']
        pulse_path = os.path.join(data_dir, f'Subject {i}_cleaned', f'{act_id}_{act} Pulse feature list.xlsx')
        pulse_df = pd.read_excel(pulse_path, sheet_name=None)
        pulse_df = pulse_df['Sheet1']
        emg_np = emg_df[emg_columns].values
        pulse_np = pulse_df[pulse_columns].values
        print(f'Processing Subject {i} - {act}')
        print(f'Before: {emg_np.shape}, {pulse_np.shape}')
        if len(emg_np) < num_seconds:
            emg_np = np.pad(emg_np, ((0, num_seconds - emg_np.shape[0]), (0, 0)), mode='edge')
        else:
            emg_np = emg_np[:num_seconds]
        if len(pulse_np) < num_seconds:
            pulse_np = np.pad(pulse_np, ((0, num_seconds - pulse_np.shape[0]), (0, 0)), mode='edge')
        else:
            pulse_np = pulse_np[:num_seconds]
        print(f'After: {emg_np.shape}, {pulse_np.shape}')
        # save to csv
        emg_df_out = pd.DataFrame(emg_np)
        pulse_df_out = pd.DataFrame(pulse_np)
        emg_df_out.to_csv(os.path.join(out_dir, f'Subject_{i}-cleaned-{act}-EMG.csv'))
        pulse_df_out.to_csv(os.path.join(out_dir, f'Subject_{i}-cleaned-{act}-Pulse.csv'))

# %%
