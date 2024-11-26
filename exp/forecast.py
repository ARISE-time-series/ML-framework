from data.dataloaders import get_loader
from .basic import Exp_Basic
from models import Transformer
from utils.helper import EarlyStopping, adjust_learning_rate, visual, count_parameters
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time
from omegaconf import OmegaConf

import warnings
import matplotlib.pyplot as plt
import numpy as np

import wandb


warnings.filterwarnings('ignore')


def kernel_density(feats, bandwidth=0.5):
    '''
    calculating the sampling probability for each sample using Gaussian kernel

    Args:
        feats: torch.Tensor, (N, d)
    Returns:
        probs: torch.Tensor, (N, N)
    '''
    logprob = -0.5 * (feats.unsqueeze(1) - feats.unsqueeze(0)).pow(2).sum(2) / (2 * bandwidth ** 2)
    logprob = logprob - logprob.logsumexp(1, keepdim=True)
    probs = logprob.exp()
    return probs


def mixup_data(x, x_mark, y, y_mark, alpha=0.0, bandwidth=0.5):
    batch_size = x.shape[0]
    if alpha > 0.0:
        feat = y[..., 0]
        probs = kernel_density(feat, bandwidth=bandwidth)
        probs = probs.cpu().numpy()
        lam = np.random.beta(alpha, alpha)

        index = np.array([np.random.choice(np.arange(batch_size), p=prob) for prob in probs])
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_x_mark = lam * x_mark + (1 - lam) * x_mark[index, :]
        mixed_y = lam * y + (1 - lam) * y[index, :]
        mixed_y_mark = lam * y_mark + (1 - lam) * y_mark[index, :]
    else:
        mixed_x = x
        mixed_x_mark = x_mark
        mixed_y = y
        mixed_y_mark = y_mark
    return mixed_x, mixed_x_mark, mixed_y, mixed_y_mark



class Exp_Forecast(Exp_Basic):
    def __init__(self, config):
        super(Exp_Forecast, self).__init__(config)

    def _build_model(self):
        model_dict = {
            'Transformer': Transformer,
        }
        model = model_dict[self.config.model.name].Model(self.config.model).float()
        print('Model parameters:', count_parameters(model))
        return model

    def _get_data(self, flag, subject=None, act=None):
        data_loader = get_loader(self.config, flag, subject, act)
        return data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), 
                                 lr=self.config.train.lr, 
                                 weight_decay=self.config.train.weight_decay)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        # criterion = nn.L1Loss()
        return criterion

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, target, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                target = target.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(target[:, -self.config.model.pred_len:, :]).float()
                dec_inp = torch.cat([target[:, :self.config.model.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1
                outputs = outputs[:, -self.config.model.pred_len:, f_dim:]

                pred = outputs.detach().cpu()
                true = target[:, -self.config.model.pred_len:].detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, path, eval=False, mixup=0.0, bandwidth=1.0):
        train_loader = self._get_data(flag='train')
        vali_loader = self._get_data(flag='test')

        os.makedirs(path, exist_ok=True)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.config.train.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.config.train.epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, target, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                target = target.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x, batch_x_mark, target, batch_y_mark = mixup_data(batch_x, batch_x_mark, target, batch_y_mark, alpha=mixup, bandwidth=bandwidth)

                # decoder input
                dec_inp = torch.zeros_like(target[:, -self.config.model.pred_len:, :]).float()
                dec_inp = torch.cat([target[:, :self.config.model.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # print(outputs.shape,batch_y.shape)
                f_dim = -1
                outputs = outputs[:, -self.config.model.pred_len:, f_dim:]

                loss = criterion(outputs, target[:, -self.config.model.pred_len:])
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.config.train.epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            
            vali_loss = self.vali(vali_loader, criterion)
                # test_loss = self.vali(test_data, test_loader, criterion)
            if wandb.run is not None:
                wandb.log({'train_loss': train_loss, 'vali_loss': vali_loss}, step=epoch)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, (epoch + 1) // 5, self.config.train)
        
        best_model_path = os.path.join(path, 'checkpoint.pt')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    @torch.no_grad()
    def test(self, exp_dir, subject, act):
        test_loader =self._get_data(flag='pred', subject=subject, act=act)
        self.model.eval()

        preds = []
        ground_truth = []
        start_tokens = None
        for i, (batch_x, target, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            if start_tokens is None:
                start_tokens = torch.zeros([target.shape[0], self.config.model.label_len, target.shape[2]]).float().to(self.device)
            batch_x, target = batch_x.float().to(self.device), target.float()
            batch_x_mark, batch_y_mark = batch_x_mark.float().to(self.device), batch_y_mark.float().to(self.device)

            # decoder input
        pass



    def eval(self, exp_dir, subject, act):
        test_loader = self._get_data(flag='pred', subject=subject, act=act)
        ckpt_path = os.path.join(exp_dir, 'checkpoint.pt')
        
        print('loading model')
        self.model.load_state_dict(torch.load(ckpt_path))
        print(f'Model loaded from {ckpt_path}')

        preds = []
        ground_truth = []

        self.model.eval()
        start_tokens = None
        with torch.no_grad():
            for i, (batch_x, target, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if start_tokens is None:
                    start_tokens = torch.zeros([target.shape[0], self.config.model.label_len, target.shape[2]]).float().to(self.device)
                
                batch_x = batch_x.float().to(self.device)
                target = target.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # decoder input
                dec_inp = torch.zeros([target.shape[0], batch_x_mark.shape[1], target.shape[2]]).float().to(self.device)
                dec_inp = torch.cat([start_tokens, dec_inp], dim=1).float().to(self.device)
                # encoder - decoder

                if self.config.model.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                pred = outputs[:, -self.config.model.pred_len:, -1:].detach().cpu().numpy()  # .squeeze()
                start_tokens = outputs[:, -self.config.model.label_len:, :]
                preds.append(pred)
                ground_truth.append(target[:, -self.config.model.pred_len:, :].numpy())

        # preds = np.array(preds)
        preds = np.concatenate(preds, axis=0).reshape(-1, 1)
        ground_truths = np.concatenate(ground_truth, axis=0).reshape(-1, 1)
        # compute metrics
        # mae, mse, rmse, mape, mspe, rse, corr = metric(preds, ground_truths)
        # print(f'MAE: {mae:.4f}, MSE: {mse:.4f}'
        #       f'RMSE: {rmse:.4f}, MAPE: {mape:.4f}, MSPE: {mspe:.4f}, RSE: {rse:.4f}, CORR: {corr:.4f}')
        # if (pred_data.scale):
            # preds = pred_data.inverse_transform(preds)

        save_data = {
            'preds': preds,
            'ground_truths': ground_truths,
        }
        
        title = f'{subject}-{act}'
        # result save
        # if preds <= (ground_truths + 1.2) and preds >= (ground_truths - 1.2), we consider it as correct prediction
        acc = np.sum((preds <= (ground_truths + 1.4)) & (preds >= (ground_truths - 1.4))) / len(preds)
        print(f'{subject}-{act} | accuracy: {acc}')

        pred_dir = os.path.join(exp_dir, 'pred')
        os.makedirs(pred_dir, exist_ok=True)   
        np.savez(os.path.join(pred_dir, f'{title}.npz'), **save_data)
        # pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1), columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)
        plt.plot(preds.reshape(-1), label='Prediction')
        plt.plot(ground_truths.reshape(-1), label='Ground truth')
        plt.legend()

        plt.title(f'{title}-acc-{acc:.4f}')
        plt.savefig(os.path.join(pred_dir, f'{title}.png'))
        plt.clf()
        print(f'Prediction saved at {pred_dir}/{title}.png')
        return acc
