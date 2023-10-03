import torch
from torch import nn

import problem_specs
import utils
from model.solvers import ODEINTWEvents


class RecognitionRNN(nn.Module):
    def __init__(self, latent_dim, obs_dim, nhidden, d):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.i2hs = nn.ModuleList(
            [nn.Linear(nhidden, nhidden) for _ in range(d)]
        )
        self.h2o = nn.Linear(nhidden, latent_dim)

    def forward(self, x, h):
        h = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(h))
        for i2 in self.i2hs:
            t = torch.tanh(i2(h))
            h = t + h
        out = self.h2o(h)
        return out, h

    def initHidden(self, nbatch):
        return torch.zeros(nbatch, self.nhidden)

    def recognize(self, device, dataset, rec_num):
        h = self.initHidden(dataset.size(0)).to(device)

        for t in range(rec_num):
            obs = dataset[:, rec_num - t - 1, :]
            out, h = self.forward(obs, h)

        return out


class RecognitionODERNN(nn.Module):
    def __init__(self, latent_ode, rnn, latent_dim, rec_len):
        super(RecognitionODERNN, self).__init__()

        self.latent_ode = latent_ode
        self.rnn = rnn
        self.rec_len = rec_len
        self.latent_dim = latent_dim
        self.solver = ODEINTWEvents()

    def forward(self, tsdata):
        timestamps = tsdata.timestamps
        timestamps_len = timestamps.shape[1]

        state = torch.zeros((timestamps.shape[0], self.latent_dim)).to(
            utils.DEFAULT_DEVICE
        )
        x = tsdata.dataset[:, timestamps_len - 1]
        x = torch.cat([state, x], dim=-1)

        state = state + self.rnn(x)

        prev_ts = timestamps[:, timestamps_len - 1]

        for i in reversed(range(timestamps_len - 1)):
            next_ts = timestamps[:, i]
            x = tsdata.dataset[:, i]

            state, _ = self.solver(
                latent_ode=self.latent_ode,
                init_state=state,
                first_t=next_ts,  # reverse timestamps, same length
                last_t=prev_ts,  # reverse timestamps, same length
                constant_step=0.01,  # TODO: use constant
                discrete_updates=False,
                state_updater_c=None,
                state_updater=None,
                method="rk4",
                bypass_state_update=True,
            )
            x = torch.cat([state, x], dim=-1)
            state = state + self.rnn(x)
            prev_ts = next_ts

        return state

    def recognize(self, tsdata):
        tsdata = tsdata.truncated_by_obs(
            self.rec_len
        )  # TODO: maybe truncate by time
        return self(tsdata)


class MultIdEnc(nn.Module):
    def __init__(self, encoded_len):
        super(MultIdEnc, self).__init__()
        self.encoded_len = encoded_len
        self._rec_len = encoded_len

    def recognize(self, tsdata, needed_last):
        means = tsdata.dataset[
            :, self.encoded_len - needed_last : self.encoded_len
        ]
        return means, None, None, None

    def set_freeze(self, freeze):
        pass


class ResNetBlock1d(nn.Module):
    def __init__(self, nfilters, activation):
        super(ResNetBlock1d, self).__init__()


        self.conv1 = nn.Conv1d(nfilters, nfilters, 1)
        self.conv2 = nn.Conv1d(nfilters, nfilters, 1)
        self.activation = activation

    def forward(self, x):
        old_x = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)

        return self.activation(old_x + x)


class RecognitionWaveNet(nn.Module):
    def __init__(
        self,
        latent_dim,
        obs_dim,
        nfilters,
        depth,
        additional_convs_per_layer,
        encoded_len,
        activation,
    ):
        super(RecognitionWaveNet, self).__init__()

        assert (
            encoded_len <= 2**depth
        ), "TCN encoder must have its receptive field >= encoded_len"

        self.latent_dim = latent_dim
        self.depth = depth
        self._rec_len = encoded_len

        if activation == "ELU":
            self.activation = nn.ELU()
        else:
            raise NotImplementedError(
                "unknown activation: {}".format(activation)
            )

        self.conv1d_first = nn.Sequential(
            # nn.BatchNorm1d(obs_dim),
            nn.Conv1d(
                in_channels=obs_dim,
                out_channels=nfilters,
                kernel_size=2,
                dilation=1,
            ),
            # nn.BatchNorm1d(nfilters),
            self.activation,
            *[
                nn.Sequential(ResNetBlock1d(nfilters, self.activation))
                for _ in range(additional_convs_per_layer)
            ]
        )
        # self.bn_first = nn.BatchNorm1d(nfilters)
        self.conv1ds_middle = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=nfilters,
                            out_channels=nfilters,
                            kernel_size=2,
                            dilation=2**x,
                        ),
                        self.activation,
                    ),
                    *[
                        nn.Sequential(ResNetBlock1d(nfilters, self.activation))
                        for _ in range(additional_convs_per_layer)
                    ]
                )
                for x in range(1, depth - 1)
            ]
        )
        self.conv1d_last = nn.Conv1d(
            in_channels=nfilters,
            out_channels=latent_dim
            * 2,  # + latent_dim * latent_dim + latent_dim,
            kernel_size=2,
            dilation=2 ** (depth - 1),
        )

        self.out_param = nn.Parameter(
            torch.Tensor(encoded_len, latent_dim * latent_dim + latent_dim, 1)
        )
        torch.nn.init.xavier_normal_(self.out_param)

    def set_freeze(self, freeze):
        for param in self.conv1d_first.parameters():
            param.requires_grad = not freeze
        for param in self.conv1ds_middle.parameters():
            param.requires_grad = not freeze
        for param in self.conv1d_last.parameters():
            param.requires_grad = not freeze

    def forward(self, x, needed_last):
        if needed_last > 1:
            additional_pad = torch.zeros_like(x[:, :, : needed_last - 1]) + utils.DEFAULT_MISSING_VALUE
            x = torch.cat([additional_pad, x], dim=2)
        x = self.conv1d_first(x)

        for conv1d in self.conv1ds_middle:
            x = conv1d(x)

        seq_state = self.conv1d_last(x)

        return seq_state, self.out_param[-needed_last:]

    def _check_input(self, tsdata):
        assert isinstance(tsdata, problem_specs.TSData)

        timestamps = tsdata.timestamps[:, : self.rec_len]
        timestamp_firsts = timestamps[:, 0].view(-1, 1).repeat(1, self.rec_len)
        timestamps_norm = timestamps - timestamp_firsts
        timestamps_example = timestamps_norm[0]

        assert torch.all(timestamps_example > timestamps_norm - 0.00001)
        assert torch.all(timestamps_example < timestamps_norm + 0.00001)

        diffs = timestamps_norm[:, 1:] - timestamps_norm[:, :-1]
        diffs = diffs.view(-1)

        assert torch.all(diffs > 0.01 - 0.00001)
        assert torch.all(diffs < 0.01 + 0.00001)

    def recognize(self, tsdata, needed_last):
        recognizable_dataset = tsdata.dataset[:, : self.rec_len].permute(
            0, 2, 1
        )
        if self._rec_len < 2**self.depth:
            pad = torch.zeros_like(recognizable_dataset)[
                :, :, : 2**self.depth - self._rec_len
            ]
            pad += utils.DEFAULT_MISSING_VALUE
            recognizable_dataset = torch.cat(
                [pad, recognizable_dataset], dim=2
            )

        z0, out_param = self.forward(recognizable_dataset, needed_last)
        latent_dim = self.latent_dim
        # return z0[:, :latent_dim],
        # torch.nn.functional.elu((z0[:, latent_dim:])) + 1.0001
        mean = z0[:, :latent_dim]
        std = z0[:, latent_dim : 2 * latent_dim] + 0.0001
        weight = out_param[:, : latent_dim * latent_dim].view(
            needed_last, latent_dim, latent_dim
        )
        bias = out_param[:, latent_dim * latent_dim :]
        return mean, std, weight, bias

    @property
    def rec_len(self):
        return self._rec_len


class MLPEnc(nn.Module):
    def __init__(
        self, latent_dim, obs_dim, hidden_dim, depth, activation, encoded_len
    ):
        super(MLPEnc, self).__init__()

        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

        self._rec_len = encoded_len

        if activation == "ELU":
            self.activation = nn.ELU()
        else:
            raise NotImplementedError(
                "unknown activation: {}".format(activation)
            )

        if depth == 0:
            layers = [(obs_dim * self._rec_len, latent_dim * 2)]
        else:
            layers = [(obs_dim * self._rec_len, hidden_dim)]
            layers += [(hidden_dim, hidden_dim) for _ in range(1, depth)]
            layers += [(hidden_dim, latent_dim * 2)]

        self.lls = nn.ModuleList(
            [nn.Linear(in_dim, out_dim) for in_dim, out_dim in layers]
        )

        self.out_param = nn.Parameter(
            torch.Tensor(encoded_len, latent_dim * latent_dim + latent_dim, 1)
        )
        torch.nn.init.xavier_normal_(self.out_param)

    def set_freeze(self, freeze):
        for param in self.lls.parameters():
            param.requires_grad = not freeze

    def forward(self, x_, needed_last):
        if needed_last > 1:
            pad = torch.zeros_like(x_[:, :, :needed_last]) + utils.DEFAULT_MISSING_VALUE
            x_ = torch.cat([pad, x_], dim=2)

        def run_forward(x):
            x = x.reshape(-1, self._rec_len* self.obs_dim)

            for layer in self.lls[:-1]:
                x = layer(x)
                x = self.activation(x)

            x = self.lls[-1](x)
            return x

        x_l = [run_forward(x_[:, :, i : i + self._rec_len]) for i in range(needed_last)]
        zs = torch.cat([x.unsqueeze(1) for x in x_l], dim=1)

        return zs, self.out_param[-needed_last:]

    def recognize(self, tsdata, needed_last):
        recognizable_dataset = tsdata.dataset[:, : self.rec_len].permute(
            0, 2, 1
        )

        z0, out_param = self.forward(recognizable_dataset, needed_last)
        latent_dim = self.latent_dim
        mean = z0[:, :, :latent_dim]
        std = z0[:, :, latent_dim : 2 * latent_dim] + 0.0001
        weight = out_param[:, : latent_dim * latent_dim].view(
            needed_last, latent_dim, latent_dim
        )
        bias = out_param[:, latent_dim * latent_dim :]
        return mean, std, weight, bias

    @property
    def rec_len(self):
        return self._rec_len


class LSTMEnc(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim, depth, encoded_len):
        super(LSTMEnc, self).__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_dim, depth, batch_first=True)
        self.lstm_proj = nn.Linear(hidden_dim, latent_dim)
        self._rec_len = encoded_len
        self.encoded_len = encoded_len
        self.latent_dim = latent_dim

    def set_freeze(self, freeze):
        for param in self.lstm.parameters():
            param.requires_grad = not freeze

    def forward(self, x):
        bs = x.shape[0]
        x = self.lstm(x)[0]
        x = x.reshape(-1, x.shape[2])
        x = self.lstm_proj(x)
        x = x.view(bs, -1, x.shape[1])

        return x

    def recognize(self, tsdata, needed_last):
        dataset = tsdata.dataset[:, :self._rec_len]
        z0 = self(dataset)
        z0 = z0[:, -needed_last:]
        latent_dim = self.latent_dim
        mean = z0[:, :, :latent_dim]
        return mean, None, None, None
        