import math
import torch
from torch import nn
import numpy as np
from typing import Union


class SinusoidalEmbedding(nn.Module):
    """Transformer sinusoidal position embedding"""
    def __init__(self, time_dim, out_dim, requires_grad=False):
        super().__init__()
        self.n = time_dim
        self.d = out_dim

        position = torch.arange(self.n).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d, 2) * (-math.log(10000.0) / self.d))
        self.pe = nn.Embedding(self.n, self.d)
        self.pe.weight.data[:, 0::2] = torch.sin(position * div_term)
        self.pe.weight.data[:, 1::2] = torch.cos(position * div_term)
        self.pe.requires_grad_(requires_grad)

    def forward(self, t: Union[torch.tensor, int, np.ndarray]):
        if type(t) is int:
            t = torch.tensor([t])
        elif type(t) is np.ndarray:
            t = torch.from_numpy(t)
        return self.pe(t.view(-1))


# ======================== Backbones ============================


class BaseBackbone(nn.Module):
    """Base class for noise recovering backbone (instead of U-Net in original paper)"""
    def __init__(self, var_dim, cond_input_dim=None, hid_dim=64, num_blocks=6, n_steps=200,
                 steps_dim=32, steps_out_dim=None, steps_depth=4, act=nn.SiLU(),
                 use_cond_emb=False, cond_hid_dim=None, cond_output_dim=None, device='cpu'):
        """
        Args:
            var_dim: size of target data
            cond_input_dim: input size of conditional data (None if no conditioning)
            hid_dim: hidden size of the layers
            num_blocks: number of backbone layers
            n_steps: number of time steps embeddings (input size)
            steps_dim: hidden size of time steps embeddings
            steps_out_dim: output size of time steps embeddings
            steps_depth: the depth of time steps embedding layers
            act: activation function used in model
            use_cond_emb: whether to use conditional embeddings
            cond_hid_dim: hidden size of conditional embedding layers
            cond_output_dim: output size of conditional embedding layers
            device: device to run on ('cpu', 'cuda' or torch.cuda.device)
        NOTE:
            steps_out_dim is used when concat [y, cond, time] is used
            else output size of time is size of concat [y, cond] or [y] if unconditional
        """

        super().__init__()
        self.var_dim = var_dim
        self.cond_dim = cond_input_dim
        self.cond_input_dim = cond_input_dim
        self.cond_hid_dim = cond_hid_dim
        self.cond_output_dim = cond_output_dim

        self.cond_emb = nn.Identity()

        if self.cond_input_dim is None:
            self.in_dim = self.var_dim
        else:
            if use_cond_emb:
                self.cond_hid_dim = 16 if cond_hid_dim is None else cond_hid_dim
                self.cond_output_dim = cond_input_dim if cond_output_dim is None else cond_output_dim
                self.in_dim = self.var_dim + self.cond_output_dim

                self.cond_emb = nn.Sequential(
                    nn.Linear(self.cond_input_dim, self.cond_hid_dim),
                    nn.SiLU(),
                    nn.Linear(self.cond_hid_dim, self.cond_output_dim)
                )
                # self.cond_emb.requires_grad_(False)
            else:
                self.in_dim = self.var_dim + self.cond_input_dim

        self.steps_out_dim = steps_out_dim
        if self.steps_out_dim is not None:
            self.in_dim += self.steps_out_dim

        self.hid_dim = hid_dim
        self.n_steps = n_steps
        self.steps_dim = steps_dim

        linear_out = self.in_dim if self.steps_out_dim is None else self.steps_out_dim
        time_embs = [
            nn.Linear(self.steps_dim, self.steps_dim) if i % 2 == 0 else act
            for i in range(2 * steps_depth - 4)
        ]

        self.time_embs = nn.Sequential(
            SinusoidalEmbedding(self.n_steps, self.steps_dim),
            act,
            *time_embs,
            # nn.Dropout(0.1),
            nn.Linear(self.steps_dim, linear_out)
        )

        # self.time_embs.requires_grad_(False)

        self.num_blocks = num_blocks
        self.layers = nn.Identity()
        self.device = device

    def forward(self, x, t, cond=None):
        """
        Predicts eta_theta noise
        Args:
            x: input objects tensor of shape (B, var_dim)
            t: timestamp tensor of shape (B, 1)
            cond: input condition tensor of shape (B, cond_dim)
        Returns: predicted noise
        """

        t_enc = self.time_embs(t.to(torch.int64))
        inp = torch.cat([x, cond], dim=1) if cond is not None else x

        if self.steps_out_dim is None:
            out = self.layers(inp + t_enc)
        else:
            out = self.layers( torch.cat([inp, t_enc], dim=1))
        return out

    def to(self, device):
        super().to(device)
        self.device = device
        return self


class PlainBackbone(BaseBackbone):
    """
    Plain backbone with constant width
    Input: [X, cond], var_dim + cond_dim
    Output: [X_samples] var_dim
    """
    def __init__(self, var_dim, cond_input_dim=None, hid_dim=64, num_blocks=6, n_steps=200,
                 steps_dim=32, steps_out_dim=None, steps_depth=4, act=nn.SiLU(),
                 use_cond_emb=False, cond_hid_dim=None, cond_output_dim=None, device='cpu'):
        super().__init__(var_dim, cond_input_dim, hid_dim, num_blocks, n_steps, steps_dim, steps_out_dim, steps_depth,
                         act, use_cond_emb, cond_hid_dim, cond_output_dim, device)

        input_layer = nn.Linear(self.in_dim, hid_dim)
        layers = [nn.Linear(hid_dim, hid_dim)] * (self.num_blocks - 2)
        out_layer = nn.Linear(hid_dim, self.var_dim)

        layers_list = [input_layer] + layers + [out_layer]
        with_act = [nn.Sequential(layer, act) for layer in layers_list[:-1]] + [layers_list[-1]]
        self.layers_list = with_act
        self.layers = nn.Sequential(*layers_list)


class PlainBackboneResidual(PlainBackbone):
    """
    Plain backbone with constant width and residual connections
    Input: [X, cond], var_dim + cond_dim
    Output: [X_samples] var_dim
    """
    def __init__(self, var_dim, cond_input_dim=None, hid_dim=64, num_blocks=6, n_steps=200,
                 steps_dim=32, steps_out_dim=None, steps_depth=4, act=nn.SiLU(),
                 use_cond_emb=False, cond_hid_dim=None, cond_output_dim=None, device='cpu'):
        super().__init__(var_dim, cond_input_dim, hid_dim, num_blocks, n_steps, steps_dim, steps_out_dim, steps_depth,
                         act, use_cond_emb, cond_hid_dim, cond_output_dim, device)
        self.layers = nn.ModuleList(self.layers_list)

    def forward(self, x, t, cond=None):
        """
        Predicts eta_theta noise
        Args:
            x: input objects tensor of shape (B, var_dim)
            t: timestamp tensor of shape (B, 1)
            cond: input condition tensor of shape (B, cond_dim)
        Returns: predicted noise
        """

        t_enc = self.time_embs(t.to(torch.int64))
        inp = torch.cat([x, cond], dim=1) if cond is not None else x

        if self.steps_out_dim is None:
            out = inp + t_enc
        else:
            out = torch.cat([inp, t_enc], dim=1)

        x_embed = self.layers[0](out)

        out = None
        for i, layer in enumerate(self.layers[1:]):
            if i == 0:
                out = layer(x_embed)
            else:
                out = layer(out + x_embed)
        return out
