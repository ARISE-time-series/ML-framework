import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        in_dim = configs.in_dim
        hidden_dim = configs.hidden_dim
        out_dim = configs.num_classes
        num_layers = configs.num_layers

        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):   
        # x_enc: (batch_size, seq_len, in_dim)
        lstm_out, _ = self.lstm(x_enc)
        out = self.fc(lstm_out[:, -1, :])
        return out          # out: (batch_size, out_dim)