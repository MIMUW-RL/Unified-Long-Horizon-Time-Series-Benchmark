import torch
from torch import nn
from model.components.encoders import MultIdEnc


class LSTMModel(nn.Module):
    def __init__(
        self, obs_dim, hidden_dim, depth, separate_lstm_encoder, encoded_len
    ):
        super(LSTMModel, self).__init__()

        self.separate_lstm_encoder = separate_lstm_encoder

        if separate_lstm_encoder:
            self.enc_lstm = nn.LSTM(
                input_size=obs_dim,
                hidden_size=hidden_dim,
                num_layers=depth,
                batch_first=True,
            )

        self.main_lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=depth,
            batch_first=True,
        )

        self.ll_out = nn.Linear(hidden_dim, obs_dim)
        self.model_type = "lstm"
        self.obs_dim = obs_dim

        self.rec = MultIdEnc(encoded_len)

    def forward(self, tsdata, **kwargs):
        dataset = tsdata.dataset
        first_x = dataset[:, : self.rec.encoded_len]

        if self.separate_lstm_encoder:
            h = self.enc_lstm(first_x)[1]

            next_in = dataset[:, self.rec.encoded_len :, :1] * 0
            next_in = next_in.repeat(1, 1, self.obs_dim)

            next_x = self.main_lstm(next_in, h)[0]
            next_x = next_x.reshape(-1, next_x.shape[2])
            next_x = self.ll_out(next_x)
            next_x = next_x.view(first_x.shape[0], -1, next_x.shape[-1])

            pred_x = torch.cat([first_x, next_x], dim=1)

        else:

            if self.rec.encoded_len > 1:
                _, h = self.main_lstm(first_x[:, : self.rec.encoded_len - 1])
            else:
                h = None

            z = first_x[:, self.rec.encoded_len - 1 : self.rec.encoded_len]
            pred_x = [first_x]

            for _ in range(self.rec.encoded_len, dataset.shape[1]):
                z, h = self.main_lstm(z, h)
                z = self.ll_out(z.reshape(-1, z.shape[2])).unsqueeze(1)
                pred_x.append(z)

            pred_x = torch.cat(pred_x, dim=1)

        return {
            "pred_x": pred_x,
            "z0_mean": None,
            "z0_std": None,
        }

    def predict(self, *args, **kwargs):
        self.eval()
        with torch.no_grad():
            return self.forward(*args, **kwargs)
