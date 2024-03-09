"""
This file is used to train the encoder model for EMG and Pulse data. 
"""

import os
from argparse import ArgumentParser
from omegaconf import OmegaConf

import torch
import torch.nn as nn

from models.vae import VAE
from data.datasets import EMGDataset, PulseData

from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm


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

    # load data
    if config.data.feature == "EMG":
        train_dataset = EMGDataset(
            config.data.root, subject_list=subject_list, activities=activities
        )
    elif config.data.feature == "Pulse":
        train_dataset = PulseData(
            config.data.root, subject_list=subject_list, activities=activities
        )

    print(f"Loaded: {len(train_dataset)} training samples.")
    batch_size = config.train.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    # build model
    model = VAE(num_modes=config.model.num_modes, layers=config.model.layers)
    model = model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)

    criterion = nn.MSELoss()
    # set up wandb
    if args.use_wandb:
        wandb.init(
            project=config.log.project, entity=config.log.entity, group=config.log.group
        )

    # training loop
    model.train()
    train_steps = 0
    for epoch in tqdm(range(config.train.epochs)):
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch)
            loss.backward()
            optimizer.step()
            if args.use_wandb:
                wandb.log({"loss": loss.item()})

            if train_steps % config.train.log_freq == 0:
                print(f"Epoch {epoch}, total step {train_steps}, loss {loss.item()}")
            if train_steps % config.train.save_freq == 0 and train_steps > 0:
                ckpt_path = os.path.join(ckpt_dir, f"ckpt_{train_steps}.pt")
                print(f"Saving model to {ckpt_path}")
                torch.save(model.state_dict(), ckpt_path)
            train_steps += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/pretrain/Pulse_encoder.yaml"
    )
    parser.add_argument("--exp_dir", type=str, default="exps/encoder")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb to log")
    args = parser.parse_args()
    main(args)
