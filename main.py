import os
import random
from omegaconf import OmegaConf

import numpy as np
import torch
import hydra

from exp.classification import Exp_Classification
from exp.imputation import Exp_Imputation
from exp.forecast import Exp_Forecast
import wandb


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
    
    if config.wandb: 
        wandb.init(project='fatigue-project', group=config.task, 
                   config=OmegaConf.to_container(cfg), 
                   reinit=True, settings=wandb.Settings(start_method='fork'))
        cfg = OmegaConf.create(dict(wandb.config))
    
    if config.task == 'forecast':
        exp = Exp_Forecast(cfg)
    elif config.task == 'classification':
        exp = Exp_Classification(cfg)
    elif config.task == 'imputation':
        exp = Exp_Imputation(cfg)
    else:
        raise ValueError(f'Invalid task name: {config.task}')

    root_dir = os.path.join(config.exp_dir, config.task)
    os.makedirs(root_dir, exist_ok=True)
    exp_dir = os.path.join(root_dir, f'{cfg.model.name}_'
                           f'seq{cfg.model.seq_len}_label{cfg.model.label_len}_pred{cfg.model.pred_len}_'
                           f'bs{cfg.train.batch_size}_dm{cfg.model.d_model}_{config.tag}')
    os.makedirs(exp_dir, exist_ok=True)

    if config.test:
        if config.task == 'classification':     
            exp.eval(exp_dir)
            if config.explain:
                exp.explain(exp_dir)
        else:
            avg_acc = []
            for subject in cfg.data.test_subjects:
                for act in cfg.data.test_acts:
                    print(act)
                    acc = exp.eval(exp_dir, subject, act)
                    avg_acc.append(acc)
            print(f'Average accuracy: {np.mean(avg_acc)}')
    else:
        config_path = os.path.join(exp_dir, 'config.yaml')
        OmegaConf.save(config, config_path)
        exp.train(exp_dir, eval=True, 
                  mixup=cfg.train.mixup, 
                  bandwidth=cfg.train.bandwidth)
        if config.task == 'forecast':
            avg_acc = []
            for subject in cfg.data.test_subjects:
                for act in cfg.data.test_acts:
                    print(act)
                    acc = exp.eval(exp_dir, subject, act)
                    avg_acc.append(acc)
            print(f'Average accuracy: {np.mean(avg_acc)}')
            if wandb.run is not None:
                wandb.log({'avg_acc': np.mean(avg_acc)})
                wandb.finish()
    

if __name__ == '__main__':
    main()