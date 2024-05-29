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

class Encoder(nn.Module):
    def __init__(self, in_chan, layers=[256, 128, 32]):
        super().__init__()
        self.in_chan = in_chan
        self.mlp = MLP([in_chan] + layers)
    
    def forward(self, x):
        # x: (batch_size, in_channels, seq_len)
        # x: (batch_size, in_channels)
        out = self.mlp(x)
        # out: (batch_size, in_chan, 16)
        return out

class Decoder(nn.Module):
    def __init__(self, out_chan, layers=[32, 128, 256]):
        super().__init__()
        self.out_chan = out_chan
        self.mlp = MLP(layers + [out_chan])
    
    def forward(self, x):
        pass
    

class FourierEncoder(nn.Module):
    def __init__(self, num_modes=192, layers=[256, 128, 32]):
        super().__init__()
        self.num_modes = num_modes
        self.mlp = MLP([num_modes * 2] + layers)
    
    def forward(self, x):
        # x: (batch_size, in_channels, seq_len)

        fft_results = torch.fft.rfft(x, dim=-1)
        # fft_results: (batch_size, in_channels, seq_len // 2 + 1) complex 32
        fft_results = torch.view_as_real(fft_results)
        # fft_results: (batch_size, in_channels, seq_len // 2 + 1, 2)
        freq_feat = fft_results[:, :, :self.num_modes, :].reshape(*fft_results.shape[:-2], -1)
        # freq_feat: (batch_size, in_channels, num_modes * 2)
        out = self.mlp(freq_feat)
        # out: (batch_size, in_channels, 16)
        return out


class FourierDecoder(nn.Module):
    def __init__(self, num_modes=192, layers=[32, 128, 256]):
        super().__init__()
        self.num_modes = num_modes
        self.mlp = MLP(layers + [num_modes * 2])
    
    def forward(self, x, seq_len=None):
        # x: (batch_size, in_channels, 16)
        out = self.mlp(x)
        # out: (batch_size, in_channels, num_modes * 2)
        out = out.reshape(*out.shape[:-1], self.num_modes, 2)
        # out: (batch_size, in_channels, num_modes, 2)
        out = torch.view_as_complex(out)
        # out: (batch_size, in_channels, num_modes)
        out = torch.fft.irfft(out, n=seq_len, dim=-1)
        # out: (batch_size, in_channels, seq_len)
        return out
    

class AE(nn.Module):
    def __init__(self, num_modes=192, layers=[256, 128, 32], 
                 model_type='fourier'):
        super().__init__()
        self.num_modes = num_modes
        if model_type == 'fourier':
            self.encoder = FourierEncoder(num_modes, layers)
            self.decoder = FourierDecoder(num_modes, list(reversed(layers)))
        else:
            self.encoder = Encoder(num_modes, layers)
            self.decoder = Decoder(num_modes, list(reversed(layers)))

    def forward(self, x):
        # x: (batch_size, in_channels, seq_len)
        seq_len = x.shape[-1]
        x = self.encoder(x)
        # x: (batch_size, in_channels, 16)
        x = self.decoder(x, seq_len)
        # x: (batch_size, in_channels, seq_len)
        return x
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x, seq_len):
        return self.decoder(x, seq_len)


class VAE(nn.Module):
    def __init__(self, num_modes=192, layers=[256, 128, 32], 
                 model_type='fourier'):
        super().__init__()
        self.num_modes = num_modes
        enc_layers = layers[:-1] + [layers[-1] * 2]
        dec_layers = list(reversed(layers))
        if model_type == 'fourier':
            self.encoder = FourierEncoder(num_modes, layers=enc_layers)
            self.decoder = FourierDecoder(num_modes, layers=dec_layers)
        else:
            self.encoder = Encoder(num_modes, layers=enc_layers)
            self.decoder = Decoder(num_modes, layers=dec_layers)
            
    def reprameterization(self, mean, logvar):
        noise = torch.randn_like(mean)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        return mean + torch.exp(logvar / 2) * noise
    
    def forward(self, x):
        seq_len = x.shape[-1]

        enc_out = self.encoder(x)
        mean, logvar = enc_out.chunk(2, dim=-1)
        z = self.reprameterization(mean, logvar)

        x = self.decoder(z, seq_len)
        return x, mean, logvar
    
    def encode(self, x):
        mean, logvar = self.encoder(x).chunk(2, dim=-1)
        z = self.reprameterization(mean, logvar)
        return z
    
    def decode(self, z, seq_len):
        return self.decoder(z, seq_len)