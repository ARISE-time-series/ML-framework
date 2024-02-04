import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, config):
        self.config = config
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None
    
    def _acquire_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_data(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass
    
