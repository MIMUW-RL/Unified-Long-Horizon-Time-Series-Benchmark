import torch

import utils


def mse(y_pred, y_act, obs_flag=None, **kwargs):
    res = y_pred - y_act

    if obs_flag is not None:
        res = res[obs_flag == 1.0]

    res = res**2

    return res.mean()


def mse_far(y_pred, y_act, obs_flag, first):
    if obs_flag is None:
        return mse(y_pred[:, first:], y_act[:, first:], try_omit_first=None)
    else:
        return mse(
            y_pred[:, first:],
            y_act[:, first:],
            try_omit_first=None,
            obs_flag=obs_flag[:, first:],
        )


def rmse(y_pred, y_act):
    abs_res = torch.abs(y_pred - y_act)
    return ((abs_res**2).mean()) ** (1 / 2)


def far_mape_loss(y_pred, y_act):
    far_seq_len_start = y_act.size(1) // 4
    y_act = y_act[:, far_seq_len_start:, :]
    y_pred = y_pred[:, far_seq_len_start:, :]
    diff = torch.abs(y_act - y_pred)
    loss = (diff / (torch.abs(y_act) + 0.0001)).mean()
    return loss


def mape_loss(y_pred, y_act):
    diff = torch.abs(y_act - y_pred)
    loss = (diff / (torch.abs(y_act) + 0.0001)).mean()
    return loss


def smape_loss(y_pred, y_act):
    diff = torch.abs(y_act - y_pred)
    loss = (diff / (torch.abs(y_act) + torch.abs(y_pred) + 0.0001)).mean()
    return loss * 2.0


def avg_far_mape_loss(y_pred, y_act):
    return (far_mape_loss(y_pred, y_act) + mape_loss(y_pred, y_act)) / 2.0


def wmape_loss(y_pred, y_act, obs_flag, **kwargs):
    diff = torch.abs(y_act - y_pred)[obs_flag == 1.0]
    loss = diff.sum() / (torch.abs(y_act[obs_flag == 1.0]) + 0.0001).sum()
    return loss


z0_prior = torch.distributions.Normal(
    torch.Tensor([0.0]).to(utils.DEFAULT_DEVICE),
    torch.Tensor([1.0]).to(utils.DEFAULT_DEVICE),
)


def kl_divergence(mean, std):
    std = std.exp()
    fp_distr = torch.distributions.Normal(mean, std)
    kl = torch.distributions.kl_divergence(fp_distr, z0_prior)
    return kl.mean()


def elbo_loss(y_pred, y_test, z0_stats, try_omit_first, obs_flag=None):
    z0_mean, z0_std = z0_stats
    if obs_flag:
        raise NotImplementedError
    recon_loss = mse(
        y_pred, y_test, try_omit_first=try_omit_first
    )  # for N(0, 1) we can use MSE
    kl = kl_divergence(z0_mean, z0_std)
    # TODO: maybe torch.losumexp like in the original repo
    loss = recon_loss + 0.01 * kl
    return loss


def mae(y_pred, y_test, obs_flag, **kwargs):
    res = y_pred - y_test

    if obs_flag is not None:
        res = res[obs_flag == 1.0]

    res = torch.abs(res)

    return res.mean()


def mae_far(y_pred, y_act, obs_flag, first):
    if obs_flag is None:
        return mae(y_pred[:, first:], y_act[:, first:], try_omit_first=None)
    else:
        return mae(
            y_pred[:, first:],
            y_act[:, first:],
            try_omit_first=None,
            obs_flag=obs_flag[:, first:],
        )
