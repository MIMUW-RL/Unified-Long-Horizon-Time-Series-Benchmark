"""
modified, based on https://github.com/HazyResearch/spacetime
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
import opt_einsum as oe
import math

class OurModule(nn.Module):
    def __init__(self): 
        super().__init__()

    def register(self, name, tensor, trainable=False, lr=None, wd=None):
        """Utility method: register a tensor as a buffer or trainable parameter"""
        if trainable:
            try:
                self.register_parameter(name, nn.Parameter(tensor))
            except KeyError:
                delattr(self, name)
                self.register_parameter(name, nn.Parameter(tensor))
        else:
            
            try:
                self.register_buffer(name, tensor)
            except KeyError:
                delattr(self, name)
                self.register_buffer(name, tensor)

        optim = {}
        if trainable and lr is not None: optim["lr"] = lr
        if trainable and wd is not None: optim["weight_decay"] = wd
        if len(optim) > 0: setattr(getattr(self, name), "_optim", optim)

class SSM(OurModule):
    def __init__(self, 
                 model_dim: int, 
                 n_kernels: int,     # Number of kernels / scales
                 kernel_dim: int,
                 kernel_repeat: int,
                 n_heads: int=None,  # Number of heads per kernel
                 head_dim: int=1,    # Dimension of each head
                 kernel_weights: torch.float=None,
                 kernel_init: str='normal',
                 kernel_train: bool=True,
                 skip_connection: bool=False,
                 seed: int=42):
        super().__init__()
        # At least one of these should be int
        assert not (n_heads is None and head_dim is None)
                 
        self.model_dim = model_dim
        self.n_kernels = n_kernels
        self.kernel_dim = kernel_dim
        self.kernel_repeat = kernel_repeat
        self.head_dim, self.n_heads = self.init_heads(n_heads, head_dim)
        self.kernel_weights  = kernel_weights
        self.kernel_init     = kernel_init
        self.kernel_train    = kernel_train
        self.skip_connection = skip_connection
        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        
        self.init_weights()
        
    def init_heads(self, n_heads: int, head_dim: int):
        if head_dim is None:
            self.head_dim = self.model_dim // (self.kernel_repeat * 
                                               self.n_kernels * n_heads)
            self.n_heads  = n_heads
        else:
            self.head_dim = head_dim
            self.n_heads  = self.model_dim // (self.kernel_repeat * 
                                               self.n_kernels * head_dim)
        if self.n_heads == 5:
            print(self.n_heads, self.head_dim, n_heads, head_dim, self.kernel_repeat, self.n_kernels, self.model_dim)
            raise Exception
        return self.head_dim, self.n_heads
        
    def fft_conv(self, u_input: torch.tensor, v_kernel: torch.tensor):
        # Convolve u with v in O(n log n) time with FFT (n = len(u))
        L   = u_input.shape[-1]  # Assume u is input
        u_f = torch.fft.rfft(u_input, n=2*L) # (B H L)
        v_f = torch.fft.rfft(v_kernel[:, :L], n=2*L) # (H L)

        y_f = oe.contract('b h l, h l -> b h l', u_f, v_f) 
        y   = torch.fft.irfft(y_f, n=2*L)[..., :L]  # (B H L)
        return y
    
    def init_weights(self):
        if self.kernel_weights is not None:  
            # lr and wd as None sets them to be same as model lr and weight_decay
            register('k', self.kernel_weights, trainable=True, lr=None, wd=None)
        
        skip = torch.randn(self.model_dim)
        self.register('skip', skip, trainable=True, lr=None, wd=None)
    
    def get_kernel(self):
        raise NotImplementedError
        
    def forward(self, u):
        u = rearrange(u, 'b l d -> b d l')  # Assume u is B x L x D
        # Repeat kernels across heads
        if self.kernel_weights is None:
            k = self.get_kernel(u)
            k = repeat(k, 'nk kd -> (kr nk nh hd) kd', 
                   kr=self.kernel_repeat, nh=self.n_heads, hd=self.head_dim)
        else:
            k = self.k
        y = self.fft_conv(u, k)
        if self.skip_connection:
            y = y + oe.contract('b d l, d -> b d l', u, self.skip)
        y = rearrange(y, 'b d l -> b l d')
        return y

class CompanionSSM(SSM):
    """
    Open-loop implementation of Companion SSM:
    -> y_t = C \sum_{i = 0}^{k - 1 - i} A^k B u_i
       where A is companion matrix
    """
    def __init__(self, norm_order, **kwargs):
        self.norm_order = norm_order
        kwargs['kernel_repeat'] = 1
        kwargs['kernel_weights'] = None
        kwargs['kernel_train'] = True
        # Set kwargs['n_heads'] as n_kernels for preprocessing kernels
        # Set kwargs['head_dim'] to be original sample input dim
        super().__init__(**kwargs)
        
    def init_kernel_weights(self, kernel_init):
        if kernel_init == 'normal':
            kernel = torch.randn(self.n_kernels, self.kernel_dim)
        elif kernel_init == 'xavier':
            # Xavier-ish initialization
            stdv = 1. / math.sqrt(self.kernel_dim)
            kernel = torch.FloatTensor(self.n_kernels, 
                                       self.kernel_dim).uniform_(-stdv, stdv)
        else:
            raise NotImplementedError
        return kernel
        
    def init_weights(self):
        super().init_weights()  # Initializes skip connection
        self._fp = (self.n_kernels, self.kernel_dim)
        
        # Shift matrix initialization
        self.shift_matrix = torch.zeros(self.n_kernels, 
                                        self.kernel_dim, 
                                        self.kernel_dim)
        self.shift_matrix[:, 1:, :-1] = torch.eye(self.kernel_dim - 1)
        self.p_padding = torch.zeros(*self._fp)
        self.p_padding[:, -1] = 1.
        
        # A matrix
        a = self.init_kernel_weights(self.kernel_init)
        self.register("a", a, trainable=True, lr=None, wd=None)
        
        # B matrix
        b = self.init_kernel_weights(self.kernel_init) 
        self.register("b", b, trainable=True, lr=None, wd=None)
        
        # C matrix
        c = self.init_kernel_weights(self.kernel_init)
        self.register("c", c, trainable=True, lr=None, wd=None)
    
    def norm(self, x, ord=1):
        # x.shape is either (H x D) or (H x D x D)
        x_norm = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        # If norm(x) in batch close to 0, don't normalize 
        # (heuristicky, but we norm for stability)
        try:
            x = x / x_norm if torch.abs(x_norm).mean().item() > 1e-4 else x  
        except Exception as e:
            print(e)
            breakpoint()
        # x = F.normalize(x, dim=1, p=ord, eps=1)
        return x
    
    def matrix_power(self, l, c, b, p):
        # Construct companion matrix
        A = self.shift_matrix.to(p.device) + (
            oe.contract('h i, h j -> h j i', 
                        self.p_padding.to(p.device), p)
        )
        # Use repeated squares to power A
        g = krylov(l, A, b, c)
        return g
    
    def get_kernel(self, u, c=None, l=None):
        l = u.shape[-1] if l is None else l
        c = self.c if c is None else c
        a = (self.norm(self.a, ord=self.norm_order) 
             if self.norm_order > 0 else self.a)
        f = self.matrix_power(l, c, self.b, a).to(u.device)
        return f
    
    def forward(self, u):
        return super().forward(u)
        

class ClosedLoopCompanionSSM(CompanionSSM):
    """
    Closed-loop implementation of Companion SSM:
    - Instantiate A, B, C; so we can compute both:
    - Open-loop inference:   
      -> y_{n + h} = \sum_{i = 0}^{n + h - 1} CA^{n + h - 1 - i} B u_i
    - Closed-loop inference: 
      -> y_{n + h} = C(A + BK)^{h} x_n
                   = C(A + BK)^{h} \sum_{j = 0}^{n - 1} A^{n - 1 - j} B u_j
                   = C(A + BK)^{n + h - 1} x_1
                   = C(A + BK)^{n + h - 1} B u_0
                   = \sum_{i = 0}^{n + h - 1} C(A + BK)^{n + h - 1 - i} B u_i, u_j = 0 for j > 0
    """
    def __init__(self, 
                 lag: int=1,
                 horizon: int=1,
                 use_initial: bool=False,
                 **kwargs):
        self.lag     = lag
        self.horizon = horizon
        self.use_initial = use_initial  # When False, assumes initial hidden_state x_0 = 0. True not implemented
        self.closed_loop = True         # Toggle closed or open-loop forward pass, see self.forward
        self.inference_only = False     # Toggle different behavior during training and test
        kwargs['kernel_repeat'] = 1
        kwargs['kernel_weights'] = None
        kwargs['kernel_train'] = True
        kwargs['skip_connection'] = False
        super().__init__(**kwargs)
        
    def init_kernel_weights(self, kernel_init):
        if kernel_init == 'normal':
            kernel = torch.randn(self.n_kernels, self.kernel_dim)
        elif kernel_init == 'xavier':
            # Xavier-ish initialization
            stdv = 1. / math.sqrt(self.kernel_dim)
            kernel = torch.FloatTensor(
                self.n_kernels, self.kernel_dim).uniform_(-stdv, stdv)
        else:
            raise NotImplementedError
        return kernel
        
    def init_weights(self):
        super().init_weights()  # Initializes skip connection, A, B, C
        # K matrix
        k = self.init_kernel_weights(self.kernel_init)
        self.register("k", k, trainable=True, lr=None, wd=None)
    
    def get_companion_matrix(self, p):
        # Construct companion matrix
        return self.shift_matrix.to(p.device) + (
            oe.contract('h i, h j -> h j i', 
                        self.p_padding.to(p.device), p)
        )
    
    def fft_conv_d(self, u, v):
        L   = u.shape[-1]
        u_f = torch.fft.rfft(u, n=2*L, dim=2) # (B H L)
        v_f = torch.fft.rfft(v, n=2*L, dim=2) # (H D L)

        y_f = oe.contract('b h l, h d l -> b h l d', u_f, v_f) 
        y   = torch.fft.irfft(y_f, n=2*L, dim=2)[:, :, :L, :] # (B H L D)
        return y
    
    def forward(self, u):
        """
        During training, call this function twice to compute closed-loop and open-loop
        -> minimize the closed-loop?
        """
        u = rearrange(u, 'b l d -> b d l')
        b, d, l = u.shape
        l_horizon = self.horizon
        
        # Normalize just the non-shift column, 
        # alternatively could normalize A + BK below 
        a = (self.norm(self.a, ord=self.norm_order) 
             if self.norm_order > 0 else self.a)
        A = self.get_companion_matrix(a)
        
        if self.closed_loop:  # Compute closed-loop forecast
            # Compute hidden state 
            # -> x_lag = \sum_{i = 0}^{lag - 1} A^{lag - 1 - i}B u_i
            k_x = krylov(l, A, self.b, c=None).to(u.device)
            x = self.fft_conv_d(u, k_x)  # shape: B x H x L x D
            
            # Compute A + BK matrix
            b = (self.norm(self.b, ord=self.norm_order) 
                 if self.norm_order > 0 else self.b)
            k = (self.norm(self.k, ord=self.norm_order) 
                 if self.norm_order > 0 else self.b)
            A_BK = A + oe.contract('h i, h j -> h i j', b, k)
            
            # Rollout: Compute C(A + BK)^{h} * x_lag and K(A + BK)^{h} * x_lag
            # First compute hidden state
            x = krylov(l_horizon, A_BK, x[:, :, -1, :], c=None)
            
            # Compute predictions for layer output
            c = self.norm(self.c, ord=self.norm_order) if self.norm_order > 0 else self.c
            y = torch.einsum('...nl, ...n -> ...l', x, c).contiguous()
            y = rearrange(y, 'b d l -> b l d')
            
            # Compute predictions for layer next-time-step input (prior layer next-time-step output)
            if not self.inference_only and self.closed_loop:
                u = torch.einsum('...nl, ...n -> ...l', x, self.k).contiguous()
                u = rearrange(u, 'b d l -> b l d')
            else:
                u = None
            # Layer outputs, and next-time-step layer inputs
            return y, u
        
        else:  # Compute open-loop forecast up to L
            # A = self.norm(A, ord=self.norm_order)
            # Return CA^{n}B where A = a is computed companion matrix from self.a
            b = (self.norm(self.b, ord=self.norm_order) 
                 if self.norm_order > 0 else self.b)
            c = self.norm(self.c, ord=self.norm_order) if self.norm_order > 0 else self.c
            k = krylov(l, A, b, c).to(u.device)
            k = repeat(k, 'nk kd -> (kr nk nh hd) kd', 
                       kr=self.kernel_repeat, nh=self.n_heads, hd=self.head_dim)
            y = rearrange(self.fft_conv(u, k), 'b d l -> b l d')
            
            if not self.inference_only:
                _k  = self.norm(self.k, ord=self.norm_order)
                k_u = krylov(l, A, b, _k).to(u.device)
                k_u = repeat(k_u, 'nk kd -> (kr nk nh hd) kd', 
                             kr=self.kernel_repeat, nh=self.n_heads, hd=self.head_dim)
                y_u = rearrange(self.fft_conv(u, k_u), 'b d l -> b l d')
            else:
                y_u = None
            return y, y_u

class ClosedLoopShiftSSM(ClosedLoopCompanionSSM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def init_weights(self):
        super().init_weights()  # Initializes skip connection, A, B, C
        # A Matrix
        a = torch.zeros(self.n_kernels, self.kernel_dim)
        self.register("a", a, trainable=False, lr=None, wd=None)
        
        # B Matrix - make it not learnable
        b = torch.zeros(self.n_kernels, self.kernel_dim)
        b[:, 0] = 1
        self.register("b", b, trainable=False, lr=None, wd=None)
        
        # C matrix
        c = self.init_kernel_weights(self.kernel_init)
        self.register("c", c, trainable=True, lr=None, wd=None)
        
        # K matrix
        k = self.init_kernel_weights(self.kernel_init)
        self.register("k", k, trainable=True, lr=None, wd=None)
        raise Exception(k.shape)
        
    def get_companion_matrix(self, p):
        # Construct "companion" matrix
        return self.shift_matrix.to(p.device)


class ShiftSSM(CompanionSSM):
    """
    Open-loop implementation of Shift SSM:
    -> y_t = C \sum_{i = 0}^{k - 1 - i} S^k B u_i
       where S is shift matrix
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def init_weights(self):
        super().init_weights()  # Initializes skip connection, B, C matrices
        # A column initialized in super().init_weights(), but now we zero-out
        a = torch.zeros(self.n_kernels, self.kernel_dim)
        self.register("a", a, trainable=False, lr=None, wd=None)
        
        # B Matrix - make it not learnable by default
        b = torch.zeros(self.n_kernels, self.kernel_dim)
        b[:, 0] = 1.
        self.register("b", b, trainable=False, lr=None, wd=None)
        
        # C matrix
        c = self.init_kernel_weights(self.kernel_init)
        self.register("c", c, trainable=True, lr=None, wd=None)
    
    def forward(self, u):
        return super().forward(u)


def krylov(L, A, b, c=None, return_power=False):
    """
    Compute the Krylov matrix (b, Ab, A^2b, ...) using the squaring trick.

    If return_power=True, return A^{L-1} as well
    """
    # TODO There is an edge case if L=1 where output doesn't get broadcasted, which might be an issue if caller is expecting broadcasting semantics... can deal with it if it arises

    x = b.unsqueeze(-1) # (..., N, 1)
    A_ = A

    AL = None
    if return_power:
        AL = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        _L = L-1

    done = L == 1
    # loop invariant: _L represents how many indices left to compute
    while not done:
        if return_power:
            if _L % 2 == 1: AL = A_ @ AL
            _L //= 2

        # Save memory on last iteration
        l = x.shape[-1]
        if L - l <= l:
            done = True
            _x = x[..., :L-l]
        else: _x = x

        try:
            _x = A_ @ _x
        except Exception as e:
            print(e)
            breakpoint()
        x = torch.cat([x, _x], dim=-1) # there might be a more efficient way of ordering axes
        if not done: A_ = A_ @ A_

    try:
        assert x.shape[-1] == L
    except:
        print('x.shape', x.shape)
        print('L', L)
        breakpoint()

    if c is not None:
        x = torch.einsum('...nl, ...n -> ...l', x, c)
    x = x.contiguous() # WOW!!
    if return_power:
        return x, AL
    else:
        return x


def krylov_sequential(L, A, b, c=None):
    """ Constant matrix A

    A : (..., N, N)
    b : (..., N)
    c : (..., N)

    Returns
    if c:
    x : (..., L)
    x[i, l] = c[i] @ A^l @ b[i]

    else:
    x : (..., N, L)
    x[i, l] = A^l @ b[i]
    """

    # Check which of dim b and c is smaller to save memory
    if c is not None and c.numel() < b.numel():
        return krylov_sequential(L, A.transpose(-1, -2), c, b)

    b_ = b
    x = []
    for _ in range(L):
        if c is not None:
            x_ = torch.sum(c*b_, dim=-1) # (...) # could be faster with matmul or einsum?
        else:
            x_ = b_
        x.append(x_)
        b_ = (A @ b_.unsqueeze(-1)).squeeze(-1)

    x = torch.stack(x, dim=-1)
    return x


class ResidualSSM(SSM):
    """
    Computes both order-N differencing and moving average residuals over input sequence
    """
    def __init__(self, 
                 max_diff_order: int=4, 
                 min_avg_window: int=4, 
                 max_avg_window: int=720,
                 n_kernels: int=8,
                 kernel_repeat: int=16,
                 **kwargs):
        self.max_diff_order = max_diff_order
        self.min_avg_window = min_avg_window
        self.max_avg_window = max_avg_window
        self.n_ma_kernels = (n_kernels - self.max_diff_order) * kernel_repeat
        kwargs['n_heads'] = 1
        kwargs['kernel_weights'] = None
        kwargs['kernel_train'] = False
        kwargs['skip_connection'] = False
        # Set kwargs['kernel_repeat'] to number of model n_kernels
        super().__init__(n_kernels=n_kernels, kernel_repeat=kernel_repeat, **kwargs)
        
    def init_weights(self):
        diff_kernel    = repeat(self.init_differencing_weights(), 'nk kd -> (kr nk) kd',
                                kr=self.kernel_repeat)
        ma_r_kernel = self.init_moving_average_weights()  # Shape: (kr x nk) x hd
        self.register('diff_kernel', diff_kernel, trainable=False, lr=None, wd=None)
        self.register('ma_r_kernel', ma_r_kernel, trainable=False, lr=None, wd=None)
        
    def init_differencing_weights(self):
        kernel = torch.zeros(self.max_diff_order, self.max_diff_order).float()
        diff_coeffs = get_pascal(self.max_diff_order, self.max_diff_order).float()
        kernel[:, :self.max_diff_order] += diff_coeffs
        return kernel
    
    def init_moving_average_weights(self):
        ma_window = torch.randint(low=self.min_avg_window,
                                  high=self.max_avg_window,
                                  size=(1, self.n_ma_kernels))
        # Compute moving average kernel 
        max_window = self.max_avg_window
        kernel = torch.zeros(self.n_ma_kernels, max_window)
        kernel[:, 0] = 1.
        
        moving_avg = (1. / torch.clamp(ma_window, min=self.min_avg_window, max=max_window))
        for ix, window in enumerate(ma_window[0]):
            kernel[ix, :window] -= moving_avg[:1, ix]
        return kernel

    def get_kernel(self, u):
        """
        Initialize weights for differencing kernel
        - Assume u is shape B x D x L
        """
        b, d, l = u.shape
        l = max(l, self.diff_kernel.shape[1])
        # Pad kernels to input length
        diff_kernel = F.pad(self.diff_kernel, (0, l - self.diff_kernel.shape[1]), 'constant', 0)
        ma_r_kernel = F.pad(self.ma_r_kernel, (0, l - self.ma_r_kernel.shape[1]), 'constant', 0)
        
        # Combine kernels
        diff_kernel = rearrange(diff_kernel, '(kr nk) kd -> kr nk kd', 
                                kr=self.kernel_repeat)
        ma_r_kernel = rearrange(ma_r_kernel, '(kr nk) kd -> kr nk kd', 
                                kr=self.kernel_repeat)
        
        kernel = torch.cat([diff_kernel, ma_r_kernel], dim=1)
        kernel = repeat(kernel, 'kr nk kd -> (kr nk hd) kd', hd=self.head_dim)
        return kernel
    
    def forward(self, u):
        # Same as base SSM forward, but kernel repeating already taken care of
        u = rearrange(u, 'b l d -> b d l')
        k = self.get_kernel(u)
        y = self.fft_conv(u, k)
        return rearrange(y, 'b d l -> b l d')


class MovingAvgResidualSSM(SSM):
    """
    Computes moving average residuals over input sequence
    """
    def __init__(self, min_avg_window=4, max_avg_window=720, **kwargs):
        self.min_avg_window = min_avg_window
        self.max_avg_window = max_avg_window
        kwargs['n_heads'] = 1
        kwargs['kernel_weights'] = None
        kwargs['kernel_train'] = False
        kwargs['skip_connection'] = False
        # Set kwargs['kernel_repeat'] to number of model n_kernels
        super().__init__(**kwargs)
        
    def init_weights(self):
        # Moving average window kernels
        kernel = torch.zeros(self.n_kernels, self.kernel_dim).float()
        kernel[:, 0] = 1.
        
        # Low is a heuristic for now
        ma_window = torch.randint(low=self.min_avg_window, 
                                  high=self.max_avg_window,  # self.kernel_dim
                                  size=(1, self.n_kernels)).float()
        
        self.register('ma_window', ma_window, trainable=True, lr=None, wd=None)
        self.register('kernel', kernel, trainable=False, lr=None, wd=None)
        
    def get_kernel(self, u):
        """
        Initialize weights for differencing kernel
        - Assume u is shape B x D x L
        """
        b, d, l = u.shape
        # Set kernel values s.t. convolution computes residuals
        # from moving average, i.e., y[t] - mean(y[t:t - m])
        max_window = min(self.max_avg_window, l)
        kernel = self.kernel - (1. / torch.clamp(torch.round(self.ma_window), 
                                                 min=self.min_avg_window, 
                                                 max=max_window).T)
        return F.pad(self.kernel, (0, l-self.kernel_dim, 0, 0), 'constant', 0)


def get_pascal(n, total_rows=None):
    total_rows = n if total_rows is None else total_rows
    # Compute binomial coeffs for all rows up to n
    line = torch.zeros(total_rows, n).float()
    line[:, 0] = 1.
    for j in range(1, n):      # For all rows,
        for k in range(0, j):  # Compute C(j, k)
            # Coefficients are binomial coeffs, 
            # C(n, k + 1) = C(n, k) * (n - k) / (k + 1)
            negate = 2 * k % 2 - 1  # Negate even elements
            line[j][k+1] += (line[j][k] * (j - k) / (k + 1)) * negate
    return line


class DifferencingSSM(SSM):
    """
    Computes order-N differencing over input sequence
    """
    def __init__(self, max_diff_order=4, **kwargs):
        self.max_diff_order = max_diff_order
        kwargs['n_heads'] = 1
        kwargs['kernel_weights'] = None
        kwargs['kernel_train'] = False
        kwargs['skip_connection'] = False
        # Set kwargs['kernel_repeat'] to number of model n_kernels
        super().__init__(**kwargs)
        
    def init_weights(self):
        kernel = torch.zeros(self.n_kernels, self.kernel_dim).float()
        # Hard-coded up to 4 orders, but just the binomial coeffs / Pascal's triangle (with negatives)
        diff_coeffs = get_pascal(self.max_diff_order)
        # Could be slow, but just done once at initialization
        for ix in range(self.n_kernels):
            try:
                kernel[ix, :self.max_diff_order] += diff_coeffs[ix % len(diff_coeffs)].float()
            except:
                breakpoint()
        self.register('kernel', kernel, trainable=False, lr=None, wd=None)
    
    def get_kernel(self, u):
        """
        Initialize weights for differencing kernel
        - Assume u is shape B x D x L
        """
        b, d, l = u.shape
        return F.pad(self.kernel, (0, l-self.kernel_dim, 0, 0), 'constant', 0)


def construct_toeplitz(v, f=0.0):
    """Explicit construction of Krylov matrix [v  A @ v  A^2 @ v  ...  A^{n-1} @ v]
    where A = Z_f. This uses vectorized indexing and cumprod so it's much
    faster than using the Krylov function.
    Parameters:
        v: the starting vector of size n or (rank, n).
        f: real number
    Returns:
        K: Krylov matrix of size (n, n) or (rank, n, n).
    """
    n  = v.shape[-1]
    a = torch.arange(n, device=v.device)
    b = -a
    indices = a[:, None] + b[None]
    K = v[..., indices]
    K[..., indices < 0] *= f
    return K

def triangular_toeplitz_multiply_(u, v, sum=None):
    n = u.shape[-1]
    u_expand = F.pad(u, (0, n))
    v_expand = F.pad(v, (0, n))
    u_f = torch.fft.rfft(u_expand, n=2*n, dim=-1)
    v_f = torch.fft.rfft(v_expand, n=2*n, dim=-1)
    uv_f = u_f * v_f
    if sum is not None:
        uv_f = uv_f.sum(dim=sum)
    output = torch.fft.irfft(uv_f, n=2*n, dim=-1)[..., :n]
    return output

def triangular_toeplitz_multiply_padded_(u, v):
    """ Same as triangular_toeplitz_multiply but inputs and output assume to be 0-padded already. """
    n = u.shape[-1]
    assert n % 2 == 0
    u_f = torch.fft.rfft(u, n=n, dim=-1)
    v_f = torch.fft.rfft(v, n=n, dim=-1)
    uv_f = u_f * v_f
    output = torch.fft.irfft(uv_f, n=n, dim=-1)
    output[..., n:] = 0
    return output

class TriangularToeplitzMult(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v):
        ctx.save_for_backward(u, v)
        return triangular_toeplitz_multiply_(u, v)

    @staticmethod
    def backward(ctx, grad):
        u, v = ctx.saved_tensors
        d_u = triangular_toeplitz_multiply_(grad.flip(-1), v).flip(-1)
        d_v = triangular_toeplitz_multiply_(grad.flip(-1), u).flip(-1)
        return d_u, d_v

class TriangularToeplitzMultFast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v):
        n = u.shape[-1]
        u_expand = F.pad(u, (0, n))
        v_expand = F.pad(v, (0, n))
        u_f = torch.fft.rfft(u_expand, n=2*n, dim=-1)
        v_f = torch.fft.rfft(v_expand, n=2*n, dim=-1)

        ctx.save_for_backward(u_f, v_f)

        uv_f = u_f * v_f
        output = torch.fft.irfft(uv_f, n=2*n, dim=-1)[..., :n]
        return output

    @staticmethod
    def backward(ctx, grad):
        u_f, v_f = ctx.saved_tensors
        n = grad.shape[-1]
        g_expand = F.pad(grad.flip(-1), (0, n))
        g_f = torch.fft.rfft(g_expand, n=2*n, dim=-1)
        gu_f = g_f * u_f
        gv_f = g_f * v_f
        d_u = torch.fft.irfft(gv_f, n=2*n, dim=-1)[..., :n]
        d_v = torch.fft.irfft(gu_f, n=2*n, dim=-1)[..., :n]
        d_u = d_u.flip(-1)
        d_v = d_v.flip(-1)
        return d_u, d_v

class TriangularToeplitzMultPadded(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v):
        ctx.save_for_backward(u, v)
        output = triangular_toeplitz_multiply_(u, v)
        return output

    @staticmethod
    def backward(ctx, grad):
        u, v = ctx.saved_tensors
        d_u = triangular_toeplitz_multiply_padded_(grad.flip(-1), v).flip(-1)
        d_v = triangular_toeplitz_multiply_padded_(grad.flip(-1), u).flip(-1)
        return d_u, d_v

class TriangularToeplitzMultPaddedFast(torch.autograd.Function):
    """ Trade off speed (20-25% faster) for more memory (20-25%) """

    @staticmethod
    def forward(ctx, u, v):
        n = u.shape[-1]
        u_f = torch.fft.rfft(u, n=n, dim=-1)
        v_f = torch.fft.rfft(v, n=n, dim=-1)

        ctx.save_for_backward(u_f, v_f)

        uv_f = u_f * v_f
        output = torch.fft.irfft(uv_f, n=n, dim=-1)
        output[..., n//2:].zero_()
        return output

    @staticmethod
    def backward(ctx, grad):
        u_f, v_f = ctx.saved_tensors
        n = grad.shape[-1]
        g_expand = F.pad(grad[..., :n//2].flip(-1), (0, n//2))
        g_f = torch.fft.rfft(g_expand, n=n, dim=-1)
        gu_f = g_f * u_f
        gv_f = g_f * v_f
        d_u = torch.fft.irfft(gv_f, n=n, dim=-1)
        d_v = torch.fft.irfft(gu_f, n=n, dim=-1)
        d_u[..., n//2:].zero_()
        d_v[..., n//2:].zero_()
        d_u[..., :n//2] = d_u[..., :n//2].flip(-1) # TODO
        d_v[..., :n//2] = d_v[..., :n//2].flip(-1) # TODO
        return d_u, d_v

# triangular_toeplitz_multiply = triangular_toeplitz_multiply_
triangular_toeplitz_multiply = TriangularToeplitzMult.apply
triangular_toeplitz_multiply_fast = TriangularToeplitzMultFast.apply
triangular_toeplitz_multiply_padded = TriangularToeplitzMultPadded.apply
triangular_toeplitz_multiply_padded_fast = TriangularToeplitzMultPaddedFast.apply


def causal_convolution(u, v, fast=True, pad=False):
    if not pad and not fast:
        return triangular_toeplitz_multiply(u, v)
    if not pad and fast:
        return triangular_toeplitz_multiply_fast(u, v)
    if pad and not fast:
        return triangular_toeplitz_multiply_padded(u, v)
    if pad and fast:
        return triangular_toeplitz_multiply_padded_fast(u, v)
        

def _fft(x, N): return torch.fft.rfft(F.pad(x, (0, 2*N-x.shape[-1])), n=2*N, dim=-1)
def _ifft(x, N): return torch.fft.irfft(x, n=2*N, dim=-1)[..., :N]

def causal_convolution_inverse(u):
    """ Invert the causal convolution/polynomial/triangular Toeplitz matrix represented by u.

    This is easiest in the polynomial view:
    https://www.csa.iisc.ac.in/~chandan/courses/CNT/notes/lec5.pdf
    The idea is that
    h = g^{-1} (mod x^m) => 2h - gh^2 = g^{-1} (mod x^{2m})

    # TODO this can be numerically unstable if input is "poorly conditioned",
    # for example if u[0] is magnitudes different from the rest of u
    """
    N = u.shape[-1]
    v = u[..., :1].reciprocal()
    while v.shape[-1] < N:
        M = v.shape[-1]
        v_f = _fft(v, 2*M)
        u_f = _fft(u[..., :2*M], 2*M)
        _v = -_ifft(u_f * v_f**2, 2*M)
        _v[..., :M] = _v[..., :M] + 2*v
        v = _v
    # TODO contiguous?
    v = v[..., :N]
    return v

""" Below are experimental functions for improving the stability of LSSL/S3 algorithm. Currently not used anywhere. """

def causal_convolution_inverse_wrong(u, v):
    """ Solve u * x = v. Initial attempt by inverting the multiplication algorithm, which I think doesn't work. """
    n = u.shape[-1]
    u_expand = F.pad(u, (0, n))
    v_expand = F.pad(v, (0, n))
    u_f = torch.fft.rfft(u_expand, n=2*n, dim=-1)
    v_f = torch.fft.rfft(v_expand, n=2*n, dim=-1)
    uv_f = v_f / u_f
    x = torch.fft.irfft(uv_f, n=2*n, dim=-1)[..., :n]
    return x

def construct_toeplitz_log(v):
    n  = v.shape[-1]
    a = torch.arange(n, device=v.device)
    b = -a
    indices = a[:, None] + b[None]
    K = v[..., indices]
    K[..., indices < 0] = -100.0
    return K

def _logsumexp(x, dim=-1):
    """ logsumexp for complex """
    m = torch.max(torch.real(x), dim=dim, keepdim=True)[0]
    x = x - m
    x = torch.log(torch.sum(torch.exp(x), dim=dim))
    x = x + m.squeeze(dim)
    return x

def causal_convolution_inverse_log(u, N=-1):
    """ Invert the causal convolution/polynomial/triangular Toeplitz matrix represented by u.

    This is easiest in the polynomial view:
    https://www.csa.iisc.ac.in/~chandan/courses/CNT/notes/lec5.pdf
    The idea is that
    h = g^{-1} (mod x^m) => 2h - gh^2 = g^{-1} (mod x^{2m})

    # TODO this can be numerically unstable if input is "poorly conditioned",
    # for example if u[0] is magnitudes different from the rest of u
    """
    if N < 0:
        N = u.shape[-1]
    v = - u[..., :1]
    while v.shape[-1] < N:
        M = v.shape[-1]
        _v = F.pad(v, (0, M), value=-100.0)
        _v_ = construct_toeplitz_log(_v)
        u_ = u[..., :2*M] if u.shape[-1] >= 2*M else F.pad(u, (0, 2*M-u.shape[-1]), value=-100.0)
        _u = _logsumexp(_v_ + u_, dim=-1)
        _u = _logsumexp(_v_ + _u, dim=-1)
        _u = _u + torch.log(-torch.ones_like(_u))
        _v = _v + torch.log(2.0 * torch.ones_like(_u))
        v = _logsumexp(torch.stack([_v, _u], dim=-1), dim=-1)
    # TODO contiguous?
    v = v[..., :N]

    check = _logsumexp(construct_toeplitz_log(v) + F.pad(u, (0, N-u.shape[-1]), value=-100.0))
    print("check", check, torch.exp(check))
    return v



def power(L, A, v=None):
    """ Compute A^L and the scan sum_i A^i v_i

    A: (..., N, N)
    v: (..., N, L)
    """

    I = torch.eye(A.shape[-1]).to(A) # , dtype=A.dtype, device=A.device)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1: I = powers[-1] @ I
        L //= 2
        if L == 0: break
        l *= 2
        powers.append(powers[-1] @ powers[-1])

    if v is None: return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.size(-1) - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.size(-1) > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)

def krylov_toeplitz(L, A, b, c=None):
    """ Specializes to lower triangular Toeplitz matrix A represented by its diagonals

    A : (..., N)
    b : (..., N)
    c : (..., N)

    Returns
    x : (..., N, L)
    x[i, l] = A^l @ b[i]
    """
    x = b.unsqueeze(0) # (1, ..., N)
    A_ = A
    while x.shape[0] < L:
        xx = causal_convolution(A_, x)
        x = torch.cat([x, xx], dim=0) # there might be a more efficient way of ordering axes
        A_ = causal_convolution(A_, A_)
    x = x[:L, ...] # (L, ..., N)
    if c is not None:
        x = torch.einsum('l...n, ...n -> ...l', x, c)
    else:
        x = rearrange(x, 'l ... n -> ... n l')
    x = x.contiguous()
    return x

def krylov_toeplitz_(L, A, b, c=None):
    """ Padded version of krylov_toeplitz that saves some fft's

    TODO currently not faster than original version, not sure why
    """
    N = A.shape[-1]

    x = b.unsqueeze(0) # (1, ..., N)
    x = F.pad(x, (0, N))
    A = F.pad(A, (0, N))
    done = L == 1
    while not done:
        l = x.shape[0]
        # Save memory on last iteration
        if L - l <= l:
            done = True
            _x = x[:L-l]
        else: _x = x
        Af = torch.fft.rfft(A, n=2*N, dim=-1)
        xf = torch.fft.rfft(_x, n=2*N, dim=-1)
        xf_ = Af * xf
        x_ = torch.fft.irfft(xf_, n=2*N, dim=-1)
        x_[..., N:] = 0
        x = torch.cat([x, x_], dim=0) # there might be a more efficient way of ordering axes
        if not done:
            A = torch.fft.irfft(Af*Af, n=2*N, dim=-1)
            A[..., N:] = 0
    x = x[:L, ..., :N] # (L, ..., N)
    if c is not None:
        x = torch.einsum('l...n, ...n -> ...l', x, c)
    else:
        x = rearrange(x, 'l ... n -> ... n l')
    x = x.contiguous()
    return x


def Activation(activation=None, size=None, dim=-1, inplace=False):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return nn.Identity(inplace)
    elif activation == 'tanh':
        return nn.Tanh(inplace)
    elif activation == 'relu':
        return nn.ReLU(inplace)
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU(inplace)
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid(inplace)
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))
        
        
class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, x):
        """ x: (batch, lengths..., dim) """
        if self.training:
            if self.transposed: x = rearrange(x, 'b ... d -> b d ...')
            mask_shape = x.shape[:2] + (1,)*(x.ndim-2) if self.tie else x.shape
            mask = torch.rand(*mask_shape, device=x.device) < 1.-self.p
            x = x * mask * (1.0/(1-self.p))
            if self.transposed: x = rearrange(x, 'b d ... -> b ... d')
            return x
        return x


# class IdentitySSM(SSM):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
        
#     def init_weights(self):
#         self.register('kernel', None, trainable=False)
        
#     def forward(self, u):
#         return u


def init_preprocess_ssm(config):
    if config['method'] == 'differencing':
        ssm = DifferencingSSM
    elif config['method'] == 'ma_residual':
        ssm = MovingAvgResidualSSM
    elif config['method'] == 'residual':
        ssm = ResidualSSM
    elif config['method'] in ['identity', None]:
        return nn.Identity()
    else:
        raise NotImplementedError(f"Preprocessing config method {config['method']} not implemented!")
    return ssm(**config['kwargs'])


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,                 
                 output_dim: int,
                 activation: str=None,
                 dropout: float=0.,
                 layernorm: bool=False,
                 n_layers: int=1,
                 n_activations: int=0,
                 pre_activation: bool=False,
                 input_shape: str='bld',
                 hidden_dim: int=None,
                 skip_connection: bool=False,
                 average_pool: str=None):
        """
        Fully-connected network 
        """
        super().__init__()
        self.input_dim     = input_dim
        self.hidden_dim    = hidden_dim
        self.output_dim    = output_dim
        self.input_shape   = input_shape
        
        self.activation      = activation
        self.dropout         = dropout
        self.layernorm       = nn.LayerNorm(input_dim) if layernorm else nn.Identity()
        self.n_layers        = n_layers
        self.n_activations   = n_activations
        self.pre_activation  = pre_activation
        self.skip_connection = skip_connection
        self.average_pool    = average_pool
        
        self.initialize_layers()
        
    def initialize_layers(self):
        n_layers_to_init = self.n_layers
        n_activations_to_init = self.n_activations
        
        if self.hidden_dim is None:
            if self.n_layers < 2:
                self.hidden_dim = self.output_dim
            else:
                self.hidden_dim = self.input_dim
            
        # Add layers
        if self.n_activations > self.n_layers or self.pre_activation:
            layers = [Activation(self.activation, inplace=True), self.init_dropout()]
            n_activations_to_init -= 1
        else:
            layers = []
            
        while n_layers_to_init > 0 or n_activations_to_init > 0:
            if n_layers_to_init == self.n_layers:
                layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            elif n_layers_to_init > 1:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            elif n_layers_to_init == 1:
                layers.append(nn.Linear(self.hidden_dim, self.output_dim))
            
            if n_activations_to_init > 0:
                layers.append(Activation(self.activation, inplace=True))
            
            n_layers_to_init -= 1
            n_activations_to_init -= 1
            
        self.layers = nn.Sequential(*layers)

        
    def init_dropout(self):
        if self.dropout > 1:  # Dropout hack for now, testing DropoutNd
            return DropoutNd(p=self.dropout-1.)
        elif self.dropout > 0:
            return nn.Dropout(self.dropout)
        else:
            return nn.Identity()
        
        
    def forward(self, x):
        x = self.layernorm(x)
        
        if self.input_shape == 'bdl':
            x = rearrange(x, 'b d l -> b l d')
        
        if self.skip_connection:
            # Layernorm with skip connection
            x = self.layers(x) + x  
        else: 
            x = self.layers(x)
        
        if self.average_pool == 'l':
            x = x.mean(dim=1, keepdim=True)
        return x



def init_ssm(config):
    supported_methods = ['companion', 'closed_loop_companion',
                         'shift', 'closed_loop_shift']
    if config['method'] == 'companion':
        ssm = CompanionSSM
    elif config['method'] == 'closed_loop_companion':
        ssm = ClosedLoopCompanionSSM
    elif config['method'] == 'shift':
        ssm = ShiftSSM
    elif config['method'] == 'closed_loop_shift':
        ssm = ClosedLoopShiftSSM
    else:
        raise NotImplementedError(
            f"SSM config method {config['method']} not implemented! Please choose from {supported_methods}")
    return ssm(**config['kwargs'])


def init_mlp(config):
    if config['method'] == 'mlp':
        return MLP(**config['kwargs'])
    else:
        return nn.Identity()


class Block(OurModule):
    """
    Standard encoder block
    """
    def __init__(self, 
                 input_dim: int,
                 pre_config: str=None,
                 ssm_config: str=None,
                 mlp_config: str=None,
                 skip_connection: bool=False,
                 skip_preprocess: bool=False):
        super().__init__()
        self.input_dim = input_dim
        self.skip_connection = skip_connection
        self.skip_preprocess = skip_preprocess
        
        self.pre = init_preprocess_ssm(pre_config)
        self.ssm = init_ssm(ssm_config)
        self.mlp = init_mlp(mlp_config)
            
    def forward(self, u):
        """
        Input shape: B x L x D
        """
        z = self.pre(u)
        y = self.ssm(z)
        y = self.mlp(y)
        if self.skip_connection and self.skip_preprocess:
            return y + u  # Also skip preprocessing step
        elif self.skip_connection:
            return y + z
        else:
            return y
            

class ClosedLoopBlock(Block):
    """
    Block with a closed-loop SSM. 
    
    In SpaceTime, we only consider using one ClosedLoopBlock 
    as the last-layer in a single-layer decoder. 
    
    However, other architectures can also be explored, e.g., 
    having more "open" blocks on top of the ClosedLoopBlock 
    in a multi-layer decoder.
    """
    def __init__(self, **kwargs):
        kwargs['skip_connection'] = False
        super().__init__(**kwargs)
        
    def forward(self, u):
        z = self.pre(u)
        # Computes layer outputs and next-time-step layer inputs
        y, u_next = self.ssm(z)  
        # Return both layer outputs and prediction + "ground-truth"
        # for next-time-step layer inputs
        return y, (u_next, u)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = self.init_blocks(config)
        
    def init_blocks(self, config):
        blocks = []
        for block in config['blocks']:
            blocks.append(Block(**block))
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.blocks(x)
    
    
class Decoder(nn.Module):
    """
    In SpaceTime, we only consider using one ClosedLoopBlock 
    as the last-layer in a single-layer decoder. 
    
    However, other architectures can also be explored, e.g., 
    having more "open" blocks on top of the ClosedLoopBlock 
    in a multi-layer decoder.
    
    In future, can refactor this class to be more general 
    and support multiple layers. (p easy, just weirdness with
    nn.Sequential and multiple outputs)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = self.init_blocks(config)
        
    def init_blocks(self, config):
        return ClosedLoopBlock(**config['blocks'][0])
    
    def forward(self, x):
        return self.blocks(x)  # y, (u_next, u)


class Embedding(nn.Module):
    def __init__(self,
                 input_dim: int,
                 embedding_dim: int):
        """
        Generic class for encoding 
        """
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.initialize_layers()
        
    def initialize_layers(self):
        self.layers = nn.Identity()
        
    def forward(self, x):
        return self.layers(x)
    

class LinearEmbedding(Embedding):
    def __init__(self, input_dim, embedding_dim):
        super().__init__(input_dim, embedding_dim)
        
    def initialize_layers(self):  
        self.layers = nn.Linear(self.input_dim, self.embedding_dim)


class RepeatEmbedding(Embedding):
    def __init__(self, 
                 input_dim: int, 
                 embedding_dim: int=None, 
                 n_heads: int=None,
                 n_kernels: int=None):

        if embedding_dim is None:
            try:
                embedding_dim = input_dim * n_heads * n_kernels
            except Exception as e:
                raise e('If embedding_dim not specified, must specify n_kernels and n_heads')
        else:
            assert embedding_dim % input_dim == 0, 'Embedding_dim should be multiple of input_dim'
        
        super().__init__(input_dim, embedding_dim)
        
    def repeat(self, x):
        return repeat(x, 'b l d -> b l (r d)', 
                      r=self.embedding_dim // self.input_dim)
        
    def initialize_layers(self):  
        self.layers = self.repeat
        


def init_embedding(config):
    methods = ['linear', 'identity', 'repeat']
    if config['method'] == 'linear':
        return LinearEmbedding(**config['kwargs'])
    elif config['method'] == 'repeat':
        return RepeatEmbedding(**config['kwargs'])
    elif config['method'] == 'identity' or config['method'] is None:
        return Embedding(**config['kwargs'])
    else:
        raise NotImplementedError(f"Embedding method {config['method']} not implemented. Please select among {methods}")

