"""
modified, based on https://github.com/HazyResearch/spacetime
"""
import torch
import torch.nn as nn
from einops import repeat

from .ssm import (
    init_embedding, Encoder, Decoder, init_mlp
)
from model.components.encoders import MultIdEnc


class SpaceTime(nn.Module):
    def __init__(
        self,
        obs_dim,
        embedding_dim,
        encoder_dim,
        encoded_len,
        pred_len,
        hidden_dim,
        encoder_layers,
        decoder_layers,
        inference_only: bool=False,
    ):
        super().__init__()

        self.inference_only = inference_only
        self.lag     = encoded_len
        self.horizon = encoded_len + pred_len
        self.rec = MultIdEnc(encoded_len)
        self.model_type = "spacetime"
        
        self.embedding  = init_embedding({
            "method": "linear",
            "kwargs": {
                "input_dim": obs_dim,
                "embedding_dim": embedding_dim,
            }
        })
        n_kernels=8
        self.encoder    = self.init_encoder(
            {
                'blocks': [
                    {
                        'input_dim': embedding_dim, 
                        'pre_config': {
                            'method': 'residual', 
                            'kwargs': {
                                'max_diff_order': 4, 
                                'min_avg_window': 4, 
                                'max_avg_window': encoder_dim // 2, 
                                'model_dim': encoder_dim, 
                                'n_kernels': n_kernels, 
                                'kernel_dim': 2, 
                                'kernel_repeat': encoder_dim // n_kernels, 
                                'n_heads': 1, 
                                'head_dim': 1, 
                                'kernel_weights': None, 
                                'kernel_init': None, 
                                'kernel_train': False, 
                                'skip_connection': False, 
                                'seed': 0
                            }
                        }, 
                        'ssm_config': {
                            'method': 'companion', 
                            'kwargs': {
                                'model_dim': encoder_dim, 
                                'n_kernels': encoder_dim//n_kernels, 
                                'kernel_dim': encoder_dim // 2, 
                                'kernel_repeat': 1, 
                                'n_heads': 8, 
                                'head_dim': 1, 
                                'kernel_weights': None, 
                                'kernel_init': 'normal', 'kernel_train': True, 'skip_connection': True, 'norm_order': 1}}, 'mlp_config': {
                                    'method': 'mlp', 'kwargs': {
                                        'input_dim': hidden_dim, 
                                        'output_dim': hidden_dim, 
                                        'activation': 'gelu', 
                                        'dropout': 0.25, 
                                        'layernorm': False, 
                                        'n_layers': 1, 
                                        'n_activations': 1, 
                                        'pre_activation': True, 
                                        'input_shape': 'bld', 
                                        'skip_connection': True, 
                                        'average_pool': None
                                    }
                                }, 'skip_connection': True, 'skip_preprocess': False
                    } for _ in range(encoder_layers)
                ]
            }
        )
        self.decoder    = self.init_decoder(
            {
                'blocks': [
                    {
                        'input_dim': encoder_dim, 
                        'pre_config': {
                            'method': 'identity', 
                            'kwargs': None
                        }, 
                        'ssm_config': {
                            'method': 'closed_loop_companion', 
                            'kwargs': {
                                'lag': encoded_len, 
                                'horizon': pred_len + encoded_len, 
                                'model_dim': hidden_dim, 
                                'n_kernels': hidden_dim, 
                                'kernel_dim': hidden_dim // 2, 
                                'kernel_repeat': 1, 
                                'n_heads': 1, 
                                'head_dim': 1, 
                                'kernel_weights': None, 
                                'kernel_init': 'normal', 'kernel_train': True, 'skip_connection': False, 'norm_order': 1, 'use_initial': False}}, 'mlp_config': {'method': 'identity', 'kwargs': None}, 'skip_connection': False, 'skip_preprocess': False}
                for _ in range(decoder_layers) ]
            }
        )
        self.output     = self.init_output(
            {
                'input_dim': hidden_dim, 
                'output_dim': obs_dim, 
                'method': 'mlp', 
                'kwargs': {
                    'input_dim': hidden_dim, 
                    'output_dim': obs_dim,
                    'activation': 'gelu', 
                    'dropout': 0.25, 
                    'layernorm': False, 
                    'n_layers': 1,
                    'n_activations': 1, 
                    'pre_activation': True, 
                    'input_shape': 'bld', 
                    'skip_connection': 
                    False, 
                    'average_pool': None,
                }
            }
        )
        
    def init_encoder(self, config):
        self.encoder = Encoder(config)
        # Allow access to first encoder SSM kernel_dim
        self.kernel_dim = self.encoder.blocks[0].ssm.kernel_dim
        return self.encoder
    
    def init_decoder(self, config):
        self.decoder = Decoder(config)
        self.decoder.blocks.ssm.lag = self.lag
        self.decoder.blocks.ssm.horizon = self.horizon
        return self.decoder
    
    def init_output(self, config):
        return init_mlp(config)
    
    # -------------
    # Toggle things
    # -------------
    def set_inference_only(self, mode=False):
        self.inference_only = mode
        self.decoder.blocks.ssm.inference_only = mode
        
    def set_closed_loop(self, mode=True):
        self.decoder.blocks.ssm.closed_loop = mode
        
    def set_train(self):
        self.train()
        
    def set_eval(self):
        self.eval()
        self.set_inference_only(mode=True)
        
    def set_lag(self, lag: int):
        self.decoder.blocks.ssm.lag = lag
        
    def set_horizon(self, horizon: int):
        self.decoder.blocks.ssm.horizon = horizon
        
    # ------------
    # Forward pass
    # ------------
    def forward(self, tsdata, **kwargs):
        u = tsdata.dataset[:, :self.rec.encoded_len]
        self.set_closed_loop(True)
        # Assume u.shape is (batch x len x dim), 
        # where len = lag + horizon
        z = self.embedding(u)
        z = self.encoder(z)
        y_c, _ = self.decoder(z)  
        y_c = self.output(y_c)  # y_c is closed-loop output

        if not self.inference_only:  
            # Also compute outputs via open-loop
            self.set_closed_loop(False)
            y_o, z_u = self.decoder(z)
            y_o = self.output(y_o)    # y_o is "open-loop" output
            # Prediction and "ground-truth" for next-time-step 
            # layer input (i.e., last-layer output)
            z_u_pred, z_u_true = z_u  
        else:
            y_o = None
            z_u_pred, z_u_true = None, None
        # Return (model outputs), (model last-layer next-step inputs)
        # return (y_c, y_o), (z_u_pred, z_u_true)
        # raise Exception(u.shape, y_c.shape, y_o.shape, z_u_pred.shape, z_u_true.shape)
        y_c = y_c[:, self.rec.encoded_len:]
        y_c = torch.cat([u, y_c], dim=1)
        return {
            "pred_x": y_c,
            "z0_mean": None,
            "z0_std": None,
        }

    def predict(self, *args, **kwargs):
        with torch.no_grad():
            return self(*args, **kwargs)
