import torch
from torch import nn


class LinearDecoder(nn.Module):
    def __init__(self, latent_dim, obs_dim):
        super(LinearDecoder, self).__init__()
        self.bn_in = nn.BatchNorm1d(latent_dim)
        self.fc = nn.Linear(latent_dim, obs_dim)

    def forward(self, z):
        return self.fc(z)


class LinearLSTMDecoder(nn.Module):
    def __init__(self, latent_dim, obs_dim):
        super(LinearLSTMDecoder, self).__init__()
        self.bn_in = nn.BatchNorm1d(latent_dim)
        self.fc = nn.Linear(latent_dim, obs_dim)

    def forward(self, z):
        return self.fc(z)


class VideoDecoder(nn.Module):
    def __init__(self, latent_dim, obs_dim, nhidden, d):
        super(VideoDecoder, self).__init__()
        self.tanh = torch.tanh
        self.bn0 = nn.BatchNorm1d(latent_dim)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.bn1 = nn.BatchNorm1d(nhidden)
        self.fcs = nn.ModuleList(
            [nn.Linear(nhidden, nhidden) for _ in range(d)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(nhidden) for _ in self.fcs])

        self.ct1 = nn.ConvTranspose2d(
            in_channels=nhidden,
            out_channels=obs_dim[2],
            kernel_size=(
                obs_dim[0],
                obs_dim[1],
            ),
        )

    def forward(self, z):
        out = self.bn0(z)
        out = self.fc1(out)
        out = self.bn1(out)
        for fc, bn in zip(self.fcs, self.bns):
            t = self.tanh(fc(out))
            out = t + out
            out = bn(out)

        out = torch.reshape(
            out,
            (
                out.shape[0],
                out.shape[1],
                1,
                1,
            ),
        )
        out = self.ct1(out)

        out = torch.reshape(
            out,
            (
                out.shape[0],
                out.shape[1],
                out.shape[2],
                out.shape[3],
            ),
        )
        out = out.permute(
            0,
            2,
            3,
            1,
        )

        return out


class TimeDistributedDec(nn.Module):
    def __init__(self, module):
        super(TimeDistributedDec, self).__init__()
        self.module = module

    def forward(self, x):
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)
        y = y.contiguous().view(x.size(0), x.size(1), y.size(-1))

        return y


class IdDec(nn.Module):
    def forward(self, x):
        return x
    
    def set_freeze(self, freeze):
        pass
