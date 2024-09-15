import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.act = F.leaky_relu
        self.layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.layers = configs.layers

        self.fc = MLP(self.layers)
    
    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        x_input = x_enc.reshape(x_enc.shape[0], -1)
        pred = self.fc(x_input)
        return pred