import os
from argparse import ArgumentParser
from omegaconf import OmegaConf

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from models.vae import VAE
from data.datasets import EMGDataset, PulseData

from torch.utils.data import DataLoader

from tqdm import tqdm


@torch.no_grad()
def encode_data(encoder, data_loader, device):
    encoder.eval()
    encoded_data = []
    for batch in tqdm(data_loader):
        batch = batch.to(device)
        encoded_batch = encoder(batch)
        encoded_data.append(encoded_batch)
    return torch.cat(encoded_data, dim=0)



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = OmegaConf.load(args.config)
    subject_list = config.data.encode_list
    activities = config.data.activities
    
    outdir = os.path.join(args.outdir, 'Encoded')
    os.makedirs(outdir, exist_ok=True)

    model = VAE(num_modes=config.model.num_modes, layers=config.model.layers)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt)
    encoder = model.encoder
    encoder = encoder.to(device)
    encoder.eval()

    for subject_id in subject_list:
        for act_id, act in activities.items():
            sub_id_list = [subject_id]
            act_list = {act_id: act}

            if config.data.feature == 'EMG':
                dataset = EMGDataset(config.data.root, subject_list=sub_id_list, activities=act_list)
            elif config.data.feature == 'Pulse':
                dataset = PulseData(config.data.root, subject_list=sub_id_list, activities=act_list)
            print(f'Loaded: {len(dataset)} samples.')
            batch_size = config.train.batch_size
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
            encoded_data = encode_data(encoder, data_loader, device)
            encoded_data = encoded_data.cpu().numpy()
            print(encoded_data.shape)
            # convert to DataFrame
            df = pd.DataFrame(encoded_data[:, 0, :])
            # save to csv
            csv_path = os.path.join(outdir, f'Subject_{subject_id}-cleaned-{act}-{config.data.feature}.csv')
            df.to_csv(csv_path)
            print(f'Saved: {csv_path}')



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/encoder.yaml')
    parser.add_argument('--ckpt', type=str, default='path to checkpoint')
    parser.add_argument('--outdir', type=str, default='../data/fatigue')
    args = parser.parse_args()
    main(args)