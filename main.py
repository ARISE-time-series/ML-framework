import os
import random
from argparse import ArgumentParser
from omegaconf import OmegaConf

import numpy as np
import torch

from exp.classification import Exp_Classification
from exp.imputation import Exp_Imputation
from exp.forecast import Exp_Forecast


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/classificaiton/vae-feat.yaml')
    parser.add_argument('--exp_dir', type=str, default='exps')
    parser.add_argument('--test', action='store_true', help='Test the model.')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb to log the experiment.')
    parser.add_argument('--use_encoder', action='store_true', help='Use encoder to encode raw data to features.')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed.')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = OmegaConf.load(args.config)
    if config.model.task_name == 'imputation' or config.model.task_name == 'forecast':
        exp = Exp_Imputation(config)
    elif config.model.task_name == 'classification':
        exp = Exp_Classification(config)
    elif config.model.task_name == 'forecast':
        exp = Exp_Forecast(config)
    else:
        raise ValueError(f'Invalid task name: {config.model.task_name}')

    root_dir = os.path.join(args.exp_dir, config.model.task_name)
    os.makedirs(root_dir, exist_ok=True)
    exp_dir = os.path.join(root_dir, f'{config.model.name}_'
                           f'seq{config.model.seq_len}_label{config.model.label_len}_pred{config.model.pred_len}_'
                           f'bs{config.train.batch_size}_dm{config.model.d_model}_{config.log.tag}')
    os.makedirs(exp_dir, exist_ok=True)
    
    if args.test:
        if config.model.task_name == 'classification':     
            exp.eval(exp_dir)
        else:
            for subject in config.data.test_subjects:
                for act in config.data.test_acts:
                    print(act)
                    exp.eval(exp_dir, subject, act)
    else:
        config_path = os.path.join(exp_dir, 'config.yaml')
        OmegaConf.save(config, config_path)
        exp.train(exp_dir, eval=True, use_wandb=args.use_wandb)

