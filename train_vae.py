"""
This file is used to train the encoder model for EMG and Pulse data. 
"""

import os
from argparse import ArgumentParser
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F

from models.vae import VAE, AE

from data.datasets import EMGDataset, PulseData

from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm


def recon_loss_fn(x, x_recon):
    '''
    Args:
        x: input data
        x_recon: reconstructed data
        weight: weight for the fourier loss
    '''
    # mse loss
    mse = F.mse_loss(x_recon, x)
    # L2 loss in fourier space
    x_fft = torch.fft.rfft(x, dim=-1)
    x_recon_fft = torch.fft.rfft(x_recon, dim=-1)
    diff = x_fft - x_recon_fft
    batch_size = x.shape[0]
    freq_loss = torch.mean(torch.linalg.norm(diff.reshape(batch_size, -1), dim=1))
    return mse, freq_loss


def kl_loss_fn(mean, logvar):
    return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())


def main(args):
    config = OmegaConf.load(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject_list = config.data.subject_list
    activities = config.data.activities
    print(activities)
    # set up directory
    exp_dir = os.path.join(
        args.exp_dir, f"{config.data.feature}-encoder-{config.model.num_modes}modes-{config.log.tag}"
    )
    os.makedirs(exp_dir, exist_ok=True)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    prefix = config.data.feature
    if config.data.extra_dir != 'None':
        # load extra data
        files = os.listdir(config.data.extra_dir)
        extra_files = [os.path.join(config.data.extra_dir, file) for file in files if file.startswith(prefix)]
    else:
        extra_files = None
    
    # load data
    if config.data.feature == "EMG":
        train_dataset = EMGDataset(
            config.data.root, subject_list=subject_list, activities=activities,
            extra_files=extra_files
        )
    elif config.data.feature == "Pulse":
        train_dataset = PulseData(
            config.data.root, subject_list=subject_list, activities=activities, 
            extra_files=extra_files
        )

    print(f"Loaded: {len(train_dataset)} training samples.")
    batch_size = config.train.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True
    )
    num_batches = len(train_loader)
    # build model
    if config.model.name == "vae":
        model = VAE(num_modes=config.model.num_modes, layers=config.model.layers)
    else:
        model = AE(num_modes=config.model.num_modes, layers=config.model.layers)
    model = model.to(device)

    num_epochs = config.train.epochs
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * num_batches, eta_min=config.train.lr_min)
    # set up wandb
    if args.use_wandb:
        wandb.init(
            project=config.log.project, entity=config.log.entity, group=config.log.group, 
            config=OmegaConf.to_container(config)
        )

    # training loop
    model.train()
    for epoch in tqdm(range(config.train.epochs)):
        avg_loss = 0.0
        avg_recon_loss = 0.0
        avg_kl_loss = 0.0
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()

            if config.model.name == "vae":
                rec_batch, mean, logvar = model(batch)
            else:
                rec_batch = model(batch)
                mean, logvar = None, None

            recon_loss, freq_loss = recon_loss_fn(batch, rec_batch)
            if mean is not None and logvar is not None:
                kl_loss = kl_loss_fn(mean, logvar)
            else:
                kl_loss = torch.zeros(1, device=device)            
            
            loss = recon_loss + kl_loss * config.train.kl_weight + freq_loss * config.train.freq_weight

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            avg_loss += loss.item()
            avg_recon_loss += recon_loss.item()
            avg_kl_loss += kl_loss.item()
            if i % (num_batches // 10) == 0:
                print(f"Epoch {epoch}, batch {i}/{num_batches}, loss {loss.item()}; " 
                        f"recon_loss {recon_loss.item()}, kl_loss {kl_loss.item()}, freq_loss {freq_loss.item()}")
                if args.use_wandb:
                    wandb.log({"train/loss": loss.item(), 
                                'train/recon_loss': recon_loss.item(),
                                'train/kl_loss': kl_loss.item(),
                                'train/freq_loss': freq_loss.item()})
        
        if epoch % config.train.save_freq == 0 and epoch > 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_{epoch}.pt")
            print(f"Saving model to {ckpt_path}")
            torch.save(model.state_dict(), ckpt_path)
        
        avg_loss /= num_batches
        avg_recon_loss /= num_batches
        avg_kl_loss /= num_batches
        if args.use_wandb:
            wandb.log({"train/avg loss": avg_loss, 
                        'train/avg recon_loss': avg_recon_loss,
                        'train/avg kl_loss': avg_kl_loss})

        print(f'Epoch {epoch}, loss {avg_loss}; ' 
                f'recon_loss {avg_recon_loss}, kl_loss {avg_kl_loss}')


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/pretrain/EMG-vae.yaml"
    )
    parser.add_argument("--exp_dir", type=str, default="exps/encoder")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb to log")
    args = parser.parse_args()
    main(args)
