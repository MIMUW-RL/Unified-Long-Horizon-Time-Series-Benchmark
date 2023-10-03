import torch
from torch import nn

import problem_specs
import utils


class LTSF(nn.Module):
    def __init__(self, rec, dec, latent_dim, out_values, ltsf_type, N):
        super(LTSF, self).__init__()

        self.ltsf_type = ltsf_type

        if ltsf_type == "linear":
            self.model_type = "latent_ltsf_linear"
        elif ltsf_type == "nlinear":
            self.model_type = "latent_ltsf_nlinear"
        else:
            raise NotImplementedError

        self.rec = rec
        self.dec = dec

        self.N = N

        self.latent_dim = latent_dim
        self.out_values = out_values

        self.ll = nn.Linear(N * latent_dim, out_values * latent_dim)

    def forward(self, tsdata, sample_vae, analytical=True, **kwargs):
        assert isinstance(tsdata, problem_specs.TSData)
        z0_mean, z0_std, _, _ = self.rec.recognize(tsdata, needed_last=self.N)
        if sample_vae:
            zeros = torch.zeros_like(z0_mean)
            ones = torch.ones_like(z0_mean)
            enc_zs = z0_mean + torch.normal(zeros, ones) * z0_std.exp()
        else:
            enc_zs = z0_mean

        if self.ltsf_type == "linear":
            z = enc_zs[:, -self.N :].reshape(-1, self.latent_dim * self.N)
            pred_z = self.ll(z).view(-1, self.out_values, self.latent_dim)
        elif self.ltsf_type == "nlinear":
            z = enc_zs[:, -self.N :]
            z = z - enc_zs[:, -1:]
            z = z.reshape(-1, self.latent_dim * self.N)
            pred_z = enc_zs[:, -1:] + self.ll(z).view(
                -1, self.out_values, self.latent_dim
            )
        else:
            raise NotImplementedError
        pred_x = self.dec(pred_z)

        pred_x = torch.cat(
            [tsdata.dataset[:, : self.rec.encoded_len], pred_x], dim=1
        )
        obs_flag = tsdata.obs_flag
        pred_x[obs_flag == 0.0] = utils.DEFAULT_MISSING_VALUE

        return {
            "pred_x": pred_x,
            "z0_mean": z0_mean,
            "z0_std": z0_std,
        }

    def predict(self, *args, **kwargs):
        self.eval()
        with torch.no_grad():
            return self.forward(*args, **kwargs)
