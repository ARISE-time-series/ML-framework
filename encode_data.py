import os
from argparse import ArgumentParser
from omegaconf import OmegaConf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from models.vae import VAE, AE
from data.datasets import EMGDataset, PulseData

from torch.utils.data import DataLoader

from tqdm import tqdm


@torch.no_grad()
def encode_data(model, data_loader, device):
    model.eval()
    encoded_data = []
    recon_data = []
    gt_data = []
    for batch in tqdm(data_loader):
        batch = batch.to(device)
        encoded_batch = model.encode(batch) # encode to latent space
        recon_batch = model.decode(encoded_batch, batch.shape[-1])  # decode to original space

        gt_data.append(batch)
        recon_data.append(recon_batch)
        encoded_data.append(encoded_batch)
    return torch.cat(encoded_data, dim=0), torch.cat(recon_data, dim=0), torch.cat(gt_data, dim=0)


def plot_recon(recon_data, gt_data, outdir, subject_id, act, feature):
    '''
    Args:
        - recon_data: reconstructed data, shape (T,)
        - gt_data: ground truth data, shape (T,)
        - outdir: output directory
    '''
    # plot reconstructions

    recon = recon_data
    gt = gt_data
    plt.plot(recon, label='Reconstruction')
    plt.plot(gt, label='Ground Truth')
    plt.legend()
    plt.title(f'Subject {subject_id} - {act} - {feature}')
    plt.savefig(os.path.join(outdir, f'Subject_{subject_id}-cleaned-{act}-{feature}.png'))
    plt.clf()
    plt.close()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = OmegaConf.load(args.config)
    subject_list = config.data.encode_list
    activities = config.data.activities
    
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    figdir = os.path.join('figs', config.data.feature, f'{config.model.name}-{config.model.num_modes}')
    os.makedirs(figdir, exist_ok=True)

    if config.model.name == 'vae':
        model = VAE(num_modes=config.model.num_modes, layers=config.model.layers)
    elif config.model.name == 'ae':
        model = AE(num_modes=config.model.num_modes, layers=config.model.layers)
    else:
        raise ValueError(f'Invalid model name: {config.model.name}')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt)

    model = model.to(device)
    # start_id = 0
    # end_id = 1000
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
            latents, recon_data, gt_data = encode_data(model, data_loader, device)
            # averaged construction error
            mse = torch.nn.functional.mse_loss(recon_data, gt_data)
            print(f'MSE: {mse.item()}')

            latents = latents.cpu().numpy()
            recon_data = recon_data.cpu().numpy()
            gt_data = gt_data.cpu().numpy()
            
            if args.reconstruct:
                recon_data = recon_data.reshape(-1)
                # unnormalize data
                recon_data = dataset.unnormalize(recon_data)
                gt_data = gt_data.reshape(-1)
                gt_data = dataset.unnormalize(gt_data)
                # save to csv
                df = pd.DataFrame({'reconstruction': recon_data, 
                                   'ground truth': gt_data})
                csv_path = os.path.join(outdir, f'Subject_{subject_id}-reconstructed-{act}-{config.data.feature}.csv')
                df.to_csv(csv_path)
                print(f'Reconstructed signal saved at {csv_path}')
            else:
                # convert to DataFrame
                df = pd.DataFrame(latents[:, 0, :])
                # save to csv
                csv_path = os.path.join(outdir, f'Subject_{subject_id}-cleaned-{act}-{config.data.feature}.csv')
                df.to_csv(csv_path)
                print(f'Latent code saved at {csv_path}')



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/encoder.yaml')
    parser.add_argument('--ckpt', type=str, default='path to checkpoint')
    parser.add_argument('--outdir', type=str, default='../data/fatigue')
    parser.add_argument('--reconstruct', action='store_true')
    args = parser.parse_args()
    main(args)