import torch
from torch import nn

import utils


# main solver interface
class ODEINTWEvents(nn.Module):
    def __init__(self, default_step_size):
        super(ODEINTWEvents, self).__init__()
        self.device = utils.DEFAULT_DEVICE
        self.default_step_size = default_step_size

    def forward(
        self,
        latent_ode,
        init_state,
        first_t,
        last_t,
        constant_step,
        discrete_updates,
        method,
        atol,
        rtol,
        min_dt,
        adjoint=False,
    ):
        return self._my_odeint(
            latent_ode=latent_ode,
            init_state=init_state,
            first_t=first_t,
            last_t=last_t,
            step_size=constant_step,
            discrete_updates=discrete_updates,
            adjoint=adjoint,
            method=method,
            atol=atol,
            rtol=rtol,
            min_dt=min_dt,
        )

    def _my_odeint(
        self,
        latent_ode,
        init_state,
        first_t,
        last_t,
        step_size,
        min_dt,
        discrete_updates,
        adjoint,
        method,
        atol,
        rtol,
    ):
        state = init_state
        su_probs = []

        min_dt = torch.zeros(state.shape[0]).to(utils.DEFAULT_DEVICE) + min_dt

        if not torch.is_tensor(step_size):
            step_size = (
                torch.zeros((state.shape[0],)).to(utils.DEFAULT_DEVICE)
                + step_size
            )

        next_step_size = step_size

        steps = []

        if adjoint:
            raise NotImplementedError("adjoint with variable ts")
        while torch.any(first_t + 0.00001 < last_t):
            should_change_state = (first_t + 0.00001 < last_t).float()
            should_change_state = should_change_state.detach().view(-1, 1)
            steps.append(should_change_state)
            if method == "rk4":
                after_step_state, step_size = rk4_step_func(
                    latent_ode, state, first_t, step_size
                )
                next_step_size = step_size
            elif method == "euler":
                after_step_state, step_size = euler_step_func(
                    latent_ode, state, first_t, step_size
                )
                next_step_size = step_size
            elif method == "dopri5":
                max_dt = last_t - first_t
                after_step_state, step_size, next_step_size = dopri5_step_func(
                    func=latent_ode,
                    y0=state,
                    t0=first_t,
                    dt=next_step_size,
                    max_dt=max_dt,
                    min_dt=min_dt,
                    atol=atol,
                    rtol=rtol,
                )
            else:
                raise NotImplementedError

            possible_next_state = after_step_state
            su_probs.append(state.detach()[:, :1] * 0)

            state = (
                should_change_state * possible_next_state
                + (1 - should_change_state) * state
            )
            assert len(state.shape) == 2, "state: {}".format(state.shape)

            first_t = first_t + step_size

        su_probs = torch.cat(su_probs, dim=1)
        steps = torch.cat(steps, dim=1)

        return state, su_probs, next_step_size, steps

"""
based on https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/rk_common.py
"""

def rk4_step_func(func, y0, t0, dt):
    dt = dt.detach()
    y0 = y0.detach()
    k1 = func(y0.detach())
    k2 = func((y0 + dt.view(-1, 1) * k1 / 3))
    k3 = func((y0 + dt.view(-1, 1) * (k2 - k1 / 3)))
    k4 = func((y0 + dt.view(-1, 1) * (k1 - k2 + k3)))
    return y0 + (k1 + 3 * (k2 + k3) + k4) * dt.view(-1, 1) / 8, dt


def euler_step_func(func, y0, history, t0, dt):
    dt = dt.detach()
    return y0 + dt.view(-1, 1) * func(y0), dt


c21 = 1 / 5
c31 = 3 / 40
c32 = 9 / 40
c41 = 44 / 45
c42 = -56 / 15
c43 = 32 / 9
c51 = 19372 / 6561
c52 = -25360 / 2187
c53 = 64448 / 6561
c54 = -212 / 729
c61 = 9017 / 3168
c62 = -355 / 33
c63 = -46732 / 5247
c64 = 49 / 176
c65 = -5103 / 18656
c71 = 35 / 384
c73 = 500 / 1113
c74 = 125 / 192
c75 = -2187 / 6784
c76 = 11 / 84

y1_1 = 35 / 384
y1_3 = 500 / 1113
y1_4 = 125 / 192
y1_5 = -2187 / 6784
y1_6 = 11 / 84

z1_1 = 5179 / 57600
z1_3 = 7571 / 16695
z1_4 = 393 / 640
z1_5 = -92097 / 339200
z1_6 = 187 / 2100
z1_7 = 1 / 40


def dopri5_step_func(func, y0, t0, dt, max_dt, min_dt, atol, rtol, attempts=1):
    i = 0
    d_stop = 0.0
    next_dt = dt
    while i < attempts:
        dt = next_dt
        dt = torch.maximum(dt, min_dt)
        dt = torch.minimum(dt, max_dt)
        dt = dt.detach()
        k1 = dt.view(-1, 1) * func(y0)
        k2 = dt.view(-1, 1) * func((y0 + c21 * k1))
        k3 = dt.view(-1, 1) * func((y0 + c31 * k1 + c32 * k2))
        k4 = dt.view(-1, 1) * func((y0 + c41 * k1 + c42 * k2 + c43 * k3))
        k5 = dt.view(-1, 1) * func(
            (y0 + c51 * k1 + c52 * k2 + c53 * k3 + c54 * k4)
        )
        k6 = dt.view(-1, 1) * func(
            (y0 + c61 * k1 + c63 * k3 + c64 * k4 + c65 * k5)
        )
        k7 = dt.view(-1, 1) * func(
            (y0 + c71 * k1 + c73 * k3 + c74 * k4 + c75 * k5 + c76 * k6)
        )

        y1 = y0 + (y1_1 * k1 + y1_3 * k3 + y1_4 * k4 + y1_5 * k5 + y1_6 * k6)
        z1 = y0 + (
            z1_1 * k1
            + z1_3 * k3
            + z1_4 * k4
            + z1_5 * k5
            + z1_6 * k6
            + z1_7 * k7
        )
        z1 = z1.detach()

        d = torch.max(torch.abs(y1 - z1), dim=-1).values

        v_rtol = (
            rtol
            * torch.max(
                torch.max(y0.detach().abs(), y1.detach().abs()), dim=-1
            ).values
        )
        error_ratio = atol + v_rtol

        i += 1

        s = (error_ratio / (d + 0.0000001)) ** (1 / 5)

        next_dt = d_stop * dt + (1 - d_stop) * dt * s
        next_dt = torch.maximum(next_dt, min_dt)
        next_dt = torch.minimum(next_dt, max_dt)
        
        d_stop = d < error_ratio.float()

        if d_stop.min() > 0.9:
            break

    return y1, dt.detach(), next_dt.detach()
