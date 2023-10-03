import torch
from torch import nn


class LTSFLinear(nn.Module):
    def __init__(self, latent_dim, lookback):
        super(LTSFLinear, self).__init__()
        self.latent_dim = latent_dim
        self.lookback = lookback

        self.ll = nn.Linear(latent_dim * lookback, latent_dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.ll(x[:, -self.lookback * self.latent_dim :])


class LTSFNLinear(nn.Module):
    def __init__(self, latent_dim, lookback):
        super(LTSFNLinear, self).__init__()
        self.latent_dim = latent_dim
        self.lookback = lookback

        self.ll = nn.Linear(latent_dim * lookback, latent_dim)

    def forward(self, x):
        last_l = x[:, 0].unsqueeze(1).repeat(1, x.shape[1], 1)
        x = x - last_l
        x = x.view(x.shape[0], -1)
        return self.ll(x[:, -self.lookback * self.latent_dim :]) + last_l[:, 0]


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=stride, padding=0
        )

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class LTSFDLinear(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, latent_dim, lookback):
        super(LTSFDLinear, self).__init__()

        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)

        self.latent_dim = latent_dim
        self.lookback = lookback

        self.llt = nn.Linear(latent_dim * lookback, latent_dim)
        self.lls = nn.Linear(latent_dim * lookback, latent_dim)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        trend_init = trend_init.view(trend_init.shape[0], -1)
        seasonal_init = seasonal_init.view(seasonal_init.shape[0], -1)
        trend_output = self.llt(
            trend_init[:, -self.latent_dim * self.lookback :]
        )
        seasonal_output = self.lls(
            seasonal_init[:, -self.latent_dim * self.lookback :]
        )

        x = seasonal_output + trend_output
        return x


class MLP(nn.Module):
    def __init__(self, layers, activation, use_batchnorm, add_last=False):
        super(MLP, self).__init__()

        if use_batchnorm:
            raise NotImplementedError

        if activation == "ELU":
            self.activation = nn.ELU()
        elif activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "unknown activation: {}".format(activation)
            )

        self.activation = nn.Tanh()

        self.fcs = nn.ModuleList(
            [
                nn.Linear(
                    in_dim,
                    out_dim,
                )
                for (
                    in_dim,
                    out_dim,
                ) in layers
            ]
        )
        self.fcs_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(in_dim)
                for (
                    in_dim,
                    _,
                ) in layers
            ]
        )

        self.add_last = add_last
        print(add_last)

        self.bias = nn.Parameter(torch.Tensor(layers[-1][1]))

    def set_freeze(self, freeze):
        for param in self.parameters():
            param.requires_grad = not freeze

    def forward(self, x):
        orig_dim = len(x.shape)
        orig_shape = x.shape

        dx = x

        if orig_dim == 3:
            dx = dx.view(-1, orig_shape[2])

        # dx = self.fcs_bn[0](dx)
        for fc, bn in zip(self.fcs[:-1], self.fcs_bn[1:]):
            dx = fc(dx)
            # dx = bn(dx)
            if dx.size(-1) == fc.out_features:
                ddx = self.activation(dx)
                dx = dx + ddx
            else:
                dx = self.activation(dx)

        dx = self.fcs[-1](dx)

        if orig_dim == 3:
            dx = dx.view(orig_shape[0], orig_shape[1], -1)

        if self.add_last:
            return x + dx / 100
        else:
            return dx
