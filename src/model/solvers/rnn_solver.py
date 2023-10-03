from torch import nn

import utils


class RNNSolver(nn.Module):
    def __init__(
        self,
    ):
        super(RNNSolver, self).__init__()
        self.device = utils.DEFAULT_DEVICE

    def forward(
        self,
        latent_ode,
        init_state,
        *args,
        **kwargs,
    ):
        new_x = latent_ode(init_state)

        su_probs = new_x[:, :1].detach() * 0
        steps = su_probs + 1

        return new_x, su_probs, None, steps
