'''
This file merges the xlsx files from the fatigue dataset 
into a single csv file for each subject and each activity.
'''
import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np


_feat_list = ['GSR', 'Temp']


def main(args):
    for i in range(args.num_subjects):
        subject_id = i + 1
        subject_dir = os.path.join(args.data_dir, f'Subject {subject_id}_cleaned')
        for act in ['1_Stroop', '2_VR', '3_Hand grip', '4_Biking']:
            df = pd.DataFrame(
                {
                    'date': pd.period_range(start='2024-01-01', periods=3600, freq='S'), 
                }
            )
            for feat in _feat_list:
                datapath = os.path.join(subject_dir, f'{act} {feat}.xlsx')
                print(f'Loading {datapath}...')
                df_raw = pd.read_excel(datapath)
                target_cols = [col for col in df_raw.columns if 'Time' not in col and 'Time ' not in col]
                df[target_cols] = df_raw[target_cols]

            # load label 
            label_path = os.path.join(subject_dir, f'{act} Label.xlsx')
            df_label = pd.read_excel(label_path)
            # linear interpolation
            labels = df_label['Fatigue level']
            interpolated_labels = np.interp(np.arange(0, 3600), df_label['Time (min)'] * 60, labels)
            df['Fatigue level'] = interpolated_labels
            # save to csv
            act_name = act.split('_')[1]
            out_path = os.path.join(args.outdir, f'Subject_{subject_id}-cleaned-{act_name}.csv')
            df.to_csv(out_path, index=False)
            print(f'Saved to {out_path}')
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/fatigue')
    parser.add_argument('--outdir', type=str, default='../data/fatigue')
    parser.add_argument('--num_subjects', type=int, default=9)
    args = parser.parse_args()
    main(args)