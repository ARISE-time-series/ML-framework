from data.dataloaders import get_loader
from .basic import Exp_Basic
from models import Transformer, iTransformer, mlp
from utils.helper import EarlyStopping, adjust_learning_rate, cal_accuracy, count_parameters

from sklearn import metrics

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import shap
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wandb

warnings.filterwarnings('ignore')


def mixup_data(x, mask_x, label, alpha=0.0, num_classes=4):
    batch_size = x.shape[0]
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        index = torch.arange(-1, batch_size - 1)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_mask_x = lam * mask_x + (1 - lam) * mask_x[index, :]
        # mixup labels
        mix_label = nn.functional.one_hot(label, num_classes=num_classes)
        mix_label = lam * mix_label + (1 - lam) * mix_label[index, :]
    else:
        mixed_x = x
        mixed_mask_x = mask_x
        mix_label = nn.functional.one_hot(label, num_classes=num_classes)
    return mixed_x, mixed_mask_x, mix_label.float()


class Exp_Classification(Exp_Basic):
    def __init__(self, config):
        super(Exp_Classification, self).__init__(config)

    def _build_model(self):
        model_dict = {
            'Transformer': Transformer,
            'iTransformer': iTransformer, 
            'MLP': mlp
        }
        model = model_dict[self.config.model.name].Model(self.config.model).float()
        print('Model parameters:', count_parameters(model))
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
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy
    
    
    def train(self, path, eval=False, mixup=0.4, bandwidth=None):
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

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                batch_x, padding_mask, label = mixup_data(batch_x, padding_mask, label, mixup)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.config.train.epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_loader, criterion)
            if wandb.run is not None:
                wandb.log({'train_loss': train_loss, 'vali_loss': vali_loss, 'vali_accuracy': val_accuracy}, step=epoch)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                best_acc = -early_stopping.val_loss_min
                print(f'Best validation accuracy: {best_acc}')
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, (epoch + 1) // 5, self.config.train)

        best_model_path = os.path.join(path, 'checkpoint.pt')
        self.model.load_state_dict(torch.load(best_model_path))
        best_acc = -early_stopping.val_loss_min
        print(f'Best validation accuracy: {best_acc}')
        return self.model

    def eval(self, exp_dir):
        ckpt_path = os.path.join(exp_dir, 'checkpoint.pt')

        test_loader = self._get_data(flag='test')
        
        print('loading model')
        self.model.load_state_dict(torch.load(ckpt_path))

        preds = []
        trues = []
        folder_path = os.path.join(exp_dir, 'results')

        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)
        # save to file 
        activities = ['Biking', 'VR', 'Hand grip', 'Stroop']
        data_dict = {'labels': trues}
        for i, act in enumerate(activities):
            data_dict[act] = probs[:, i].cpu().numpy()
        df = pd.DataFrame(data_dict)
        df.to_csv(os.path.join(folder_path, 'probsnlabel.csv'), index=False)

        print('accuracy:{}'.format(accuracy))
        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')

        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.close()
        
        subject = self.config.data.test_subjects[0]
        confusion_mat = metrics.confusion_matrix(trues, predictions)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=activities)
        cm_display.plot()
        plt.savefig(os.path.join(folder_path, f'subject_{subject}-confusion_matrix.png'))
        return

    def explain(self, exp_dir, num_background_batch=10, num_test_batch=20):
        ckpt_path = os.path.join(exp_dir, 'checkpoint.pt')

        test_loader = self._get_data(flag='explain')
        
        print('loading model')
        self.model.load_state_dict(torch.load(ckpt_path))

        folder_path = os.path.join(exp_dir, 'results')

        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        background = []
        back_mask = []
        x_test = []
        x_test_mask = []
        test_preds = []
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                if i < num_background_batch:
                    background.append(batch_x)
                    back_mask.append(padding_mask)
                    print(f'Background label: {label}')
                elif i >= num_background_batch and i < num_background_batch + num_test_batch:
                    x_test.append(batch_x)
                    x_test_mask.append(padding_mask)
                    outputs = self.model(batch_x, padding_mask, None, None)
                    preds = torch.argmax(outputs, dim=1)
                    test_preds.append(preds)
                    # calculate accuracy
                    acc = torch.sum(preds == label).float() / label.shape[0]
                    print(f'Accuracy: {acc}')

        background = torch.cat(background, 0)
        back_mask = torch.cat(back_mask, 0)
        x_test = torch.cat(x_test, 0)
        x_test_mask = torch.cat(x_test_mask, 0)
        test_preds = torch.cat(test_preds, 0)
        exp = shap.GradientExplainer(self.model, [background, back_mask])
        # exp = shap.DeepExplainer(self.model, [background, back_mask])
        # shap_values = exp.shap_values([x_test, x_test_mask], ranked_outputs=1, check_additivity=False)
        shap_values = exp.shap_values([x_test, x_test_mask], ranked_outputs=1, nsamples=x_test.shape[0])
        shap_val = shap_values[0][0]
        np.save(os.path.join(folder_path, 'x_tests.npy'), x_test.cpu().numpy())
        np.save(os.path.join(folder_path, 'shap_values-grad.npy'), shap_val)
        np.save(os.path.join(folder_path, 'test_preds.npy'), test_preds.cpu().numpy())
    
    def kernel_explain(self, exp_dir, num_background_batch=5, num_test_batch=2):
        ckpt_path = os.path.join(exp_dir, 'checkpoint.pt')

        test_loader = self._get_data(flag='explain')
        
        print('loading model')
        self.model.load_state_dict(torch.load(ckpt_path))

        folder_path = os.path.join(exp_dir, 'results')

        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        background = []
        back_mask = []
        x_test = []
        x_test_mask = []
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                if i < num_background_batch:
                    background.append(batch_x)
                    back_mask.append(padding_mask)
                elif i >= num_background_batch and i < num_background_batch + num_test_batch:
                    x_test.append(batch_x)
                    x_test_mask.append(padding_mask)

        background = torch.cat(background, 0)
        back_mask = torch.cat(back_mask, 0)
        x_test = torch.cat(x_test, 0)
        x_test_mask = torch.cat(x_test_mask, 0)

        exp = shap.KernelExplainer(self.model, [background, back_mask])
        shap_values = exp.shap_values([x_test, x_test_mask], check_additivity=False)
        shap_val = shap_values[0]
        print(shap_val)
        np.save(os.path.join(folder_path, 'shap_values.npy'), shap_val)
