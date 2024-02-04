from data.dataloaders import get_loader
from .basic import Exp_Basic
from models import Transformer
from utils.helper import EarlyStopping, adjust_learning_rate, visual, test_params_flop
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

class Exp_Imputation(Exp_Basic):
    def __init__(self, config):
        super(Exp_Imputation, self).__init__(config)

    def _build_model(self):
        model_dict = {
            'Transformer': Transformer,
        }
        model = model_dict[self.config.model.name].Model(self.config.model).float()

        return model

    def _get_data(self, flag):
        data_loader = get_loader(self.config, flag)
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
                # encoder - decoder

                if self.config.model.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
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

    def train(self, path, eval=False, use_wandb=False):
        train_loader = self._get_data(flag='train')
        vali_loader = self._get_data(flag='test')

        os.makedirs(path, exist_ok=True)
        if use_wandb:
            wandb.init(project=self.config.log.project,
                       group=self.config.log.group,
                       config=OmegaConf.to_container(self.config))
        

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

                # decoder input
                dec_inp = torch.zeros_like(target[:, -self.config.model.pred_len:, :]).float()
                dec_inp = torch.cat([target[:, :self.config.model.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder

                if self.config.model.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    
                else:
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
            if use_wandb:
                wandb.log({'train_loss': train_loss, 'vali_loss': vali_loss}, step=epoch)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, (epoch + 1) // 5, self.config.train)
        if use_wandb:
            wandb.finish()
        
        best_model_path = os.path.join(path, 'checkpoint.pt')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def ieval(self, exp_dir):
        test_loader = self._get_data(flag='test')
        ckpt_path = os.path.join(exp_dir, 'checkpoint.pt')
        
        print('loading model')
        self.model.load_state_dict(torch.load(ckpt_path))

        preds = []
        trues = []
        inputx = []
        folder_path = os.path.join(exp_dir, 'results')

        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, target, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                target = target.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # decoder input
                dec_inp = torch.zeros_like(target[:, -self.config.model.pred_len:, :]).float()
                dec_inp = torch.cat([target[:, :self.config.model.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder

                if self.config.model.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.config.model.pred_len:, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                target = target[:, -self.config.model.pred_len:].detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = target  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

            
        preds = np.concatenate(preds, axis=0).reshape(-1, 1)
        trues = np.concatenate(trues, axis=0).reshape(-1, 1)

        visual(trues[:, -1], preds[:, -1], os.path.join(folder_path, 'test.png'))
        inputx = np.concatenate(inputx, axis=0)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()


        np.save(folder_path + 'pred.npy', preds)
        return

    def eval(self, exp_dir):
        test_loader = self._get_data(flag='pred')
        ckpt_path = os.path.join(exp_dir, 'checkpoint.pt')
        
        print('loading model')
        self.model.load_state_dict(torch.load(ckpt_path))

        preds = []
        ground_truth = []

        self.model.eval()
        start_tokens = None
        with torch.no_grad():
            for i, (batch_x, target, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if start_tokens is None:
                    start_tokens = torch.ones([target.shape[0], self.config.model.label_len, target.shape[2]]).float().to(self.device)
                
                batch_x = batch_x.float().to(self.device)
                
                target = target.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([target.shape[0], self.config.model.pred_len, target.shape[2]]).float().to(self.device)
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
        preds = np.concatenate(preds, axis=0)
        ground_truths = np.concatenate(ground_truth, axis=0)
        # if (pred_data.scale):
            # preds = pred_data.inverse_transform(preds)

        save_data = {
            'preds': preds,
            'ground_truths': ground_truths,
        }
        
        subject = self.config.data.test_subjects[0]
        act = self.config.data.test_act
        title = f'{subject}-{act}'
        # result save

        pred_dir = os.path.join(exp_dir, 'pred')
        os.makedirs(pred_dir, exist_ok=True)   
        np.savez(os.path.join(pred_dir, f'{title}.npz'), **save_data)
        # pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1), columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)
        plt.plot(np.reshape(preds, (-1,)), label='Prediction')
        plt.plot(np.reshape(ground_truths, (-1,)), label='Ground truth')
        plt.legend()

        plt.title(f'{title}')
        plt.savefig(os.path.join(pred_dir, f'{title}.png'))
        return
