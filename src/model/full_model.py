import torch
from torch import nn

import problem_specs
import utils


class FullModel(nn.Module):
    def __init__(
        self,
        model_type,
        obs_dim,
        latent_dim,
        rec,
        dec,
        latent_ode,
        T,
        solver,
    ):
        super(FullModel, self).__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        if T is not None:
            self.T = T

        self.latent_ode = latent_ode
        self.rec = rec
        self.dec = dec

        self.model_type = model_type
        self.solver = solver

    @property
    def rec_len(self):
        return self.rec.rec_len

    def forward(
        self,
        tsdata,
        method,
        atol,
        rtol,
        min_dt,
        sample_vae,
        discrete_updates,
    ):
        assert isinstance(tsdata, problem_specs.TSData)
        z0_mean, z0_std, _, _ = self.rec.recognize(tsdata, needed_last=1)
        encoded_observations = tsdata.dataset[:, :self.rec.encoded_len]
        pred_z, su_probs, steps = self.forward_solver(
            samp_ts=tsdata.timestamps,
            z0_mean=z0_mean,
            z0_std=z0_std,
            method=method,
            atol=atol,
            rtol=rtol,
            min_dt=min_dt,
            sample_vae=sample_vae,
            discrete_updates=discrete_updates,
        )

        pred_x = torch.cat([encoded_observations, self.dec(pred_z)], dim=1)
        obs_flag = tsdata.obs_flag
        pred_x[obs_flag == 0.0] = utils.DEFAULT_MISSING_VALUE

        return {
            "pred_x": pred_x,
            "su_probs": su_probs,
            "z0_mean": z0_mean,
            "z0_std": z0_std,
            "steps": steps,
        }

    def predict(
        self,
        tsdata,
        discrete_updates,
        method,
        atol,
        rtol,
        min_dt,
        sample_vae,
    ):
        self.eval()
        with torch.no_grad():
            ret_val = self.forward(
                tsdata=tsdata,
                discrete_updates=discrete_updates,
                method=method,
                atol=atol,
                rtol=rtol,
                min_dt=min_dt,
                sample_vae=sample_vae,
            )
            return ret_val

    def forward_solver(
        self,
        samp_ts,
        z0_mean,
        z0_std,
        method,
        atol,
        rtol,
        min_dt,
        sample_vae,
        discrete_updates,
        adjoint=False,
    ):
        zeros = torch.zeros_like(z0_mean)
        ones = torch.ones_like(z0_mean)
        if sample_vae:
            z0 = z0_mean + torch.normal(zeros, ones) * z0_std.exp()
        else:
            z0 = z0_mean
        prev_samp = samp_ts[:, self.rec.encoded_len - 1]

        next_step_size = self.solver.default_step_size

        su_probs = []
        steps = []

        results = []

        for samp_id in range(self.rec.encoded_len, samp_ts.shape[1]):
            samp = samp_ts[:, samp_id]

            pred, su_probs_, _, steps_ = self.solver(
                    latent_ode=self.latent_ode,
                    init_state=z0.squeeze(1),
                    first_t=prev_samp,
                    last_t=samp,
                    constant_step=next_step_size,
                    adjoint=adjoint,
                    method=method,
                    atol=atol,
                    rtol=rtol,
                    min_dt=min_dt,
                    discrete_updates=discrete_updates,
                )

            su_probs.append(su_probs_)
            steps.append(steps_)
            z0 = pred
            if torch.is_tensor(z0):
                results.append(z0.view(z0.shape[0], 1, -1))
            else:
                results.append(z0[0].view(z0[0].shape[0], 1, -1))

            prev_samp = samp

        results = torch.cat(results, axis=1)

        su_probs = torch.cat(su_probs, dim=1)
        steps = torch.cat(steps, dim=1)

        return results, su_probs, steps
