import os
from tqdm import tqdm
from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal

from .modules import ActNorm, InvertibleResLinear


class ResidualFlowModel(nn.Module):
    '''
    Residual Flow model class
    Pass concat [X, y] if conditioning, return only y
    '''
    def __init__(self, var_dim, cond_dim=None, n_layers=6, hid_dim=32, n_block_layers=2,
                 spnorm_coeff=0.97, logdet='unbias', n_backward_iters=100):
        """
        Args:
            var_dim: target data size
            cond_dim: conditional data size (None if not used)
            n_layers: number of residual blocks in model
            hid_dim: residual block hidden size
            n_block_layers: number of layers in each residual block
            spnorm_coeff: spectral normalization coeff (Lipschitz), must be < 1
            logdet: logdet estimation strategy
            n_backward_iters: number of iterations to sample the object
        """
        super().__init__()
        self.var_dim = var_dim
        self.cond_dim = cond_dim
        self.in_dim = var_dim + cond_dim if cond_dim is not None else var_dim
        self.out_dim = self.var_dim

        self.n_layers = n_layers
        self.device = 'cpu'
        self.net = None

        assert spnorm_coeff < 1
        self.actnorm_in_dim = self.in_dim if self.cond_dim is None else self.var_dim
        self.hid_dim = hid_dim
        self.net = nn.ModuleList()
        for i in range(self.n_layers):
            self.net.append(ActNorm(self.actnorm_in_dim))
            self.net.append(
                InvertibleResLinear(self.in_dim, self.out_dim, base_filters=self.hid_dim,
                                        coeff=spnorm_coeff, n_layers=n_block_layers,
                                        logdet_estimator=logdet, n_backward_iters=n_backward_iters)
            )

    def forward_process(self, z, cond=None):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        for i, layer in enumerate(self.net):
            if cond is not None and i % 2 == 1:  # if layer is InvertibleResLinear
                z = torch.cat([z, cond], dim=1)
            z, log_df_dz = layer(z, log_df_dz)
        return z, log_df_dz

    def backward_process(self, z, cond=None):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        for i, layer in enumerate(self.net[::-1]):
            if cond is not None and i % 2 == 0:  # if layer is InvertibleResLinear
                z = torch.cat([z, cond], dim=1)
            z, log_df_dz = layer.backward(z, log_df_dz)
        return z, log_df_dz

    def to(self, device):
        super().to(device)
        self.device = device
        self.net = self.net.to(device)
        return self


# =========================== Wrappers ==========================


class BaseFlowWrapper(object):
    def __init__(self, var_dim, cond_dim=None, n_layers=6, hid_dim=32, n_block_layers=2,
                     spnorm_coeff=0.97, logdet='unbias', n_backward_iters=100,
                     optimizer=None, batch_size=64, n_epochs=100, checkpoint_dir=None, device='cpu',
                     scheduler=None, **scheduler_kwargs):
        """
        Args:
            var_dim: target data size
            cond_dim: conditional data size (None if not used)
            n_layers: number of residual blocks in model
            hid_dim: residual block hidden size
            n_block_layers: number of layers in each residual block
            spnorm_coeff: spectral normalization coeff (Lipschitz), must be < 1
            logdet: logdet estimation strategy
            n_backward_iters: number of iterations to sample the object
        """
        self.flow = ResidualFlowModel(var_dim, cond_dim, n_layers,
                                      hid_dim, n_block_layers, spnorm_coeff, logdet, n_backward_iters).to(device)
        if optimizer is not None:
            self.optim = optimizer
        else:
            self.optim = torch.optim.Adam(self.flow.parameters(), lr=1e-2, weight_decay=1e-4)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.scheduler = scheduler

        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir is not None:
            try:
                os.makedirs(self.checkpoint_dir)
                print(f'Created directory {self.checkpoint_dir}')
            except:
                print(f'Directory {self.checkpoint_dir} already exists or can not be created')
                pass

        self.device = self.flow.device

        self.min_epoch_loss = torch.inf
        self.last_epoch_loss = None

        self.var_dim = self.flow.var_dim
        self.cond_dim = self.flow.cond_dim
        self.in_dim = self.flow.in_dim
        self.out_dim = self.flow.out_dim

        self.mu = torch.zeros(self.out_dim, dtype=torch.float32, device=self.device)
        self.var = torch.eye(self.out_dim, dtype=torch.float32, device=self.device)
        self.normal = MultivariateNormal(self.mu, self.var)

    def fit(self, Y: torch.Tensor, X_cond: torch.Tensor = None):
        """
        Fits flow
        Args:
            Y: input objects tensor of shape (B, var_dim)
            X_cond: condition tensor of shape (B, cond_dim)
        Returns: epochs losses list
        """
        raise NotImplemented

    def sample(self, input: Union[torch.tensor, int], batch_size=None):
        """
        Samples objects from condition of the X_cond's shape
        Args:
            input: int N in unconditional case, torch.Tensor X_cond else
            batch_size: None if no batchification used, else int -- batch_size
        Returns: new objects
        """
        raise NotImplemented

    def loss(self, z, logdet):
        """
        Computes loss (likelihood), see slide 20:
        https://github.com/HSE-LAMBDA/DeepGenerativeModels/blob/spring-2021/lectures/8-NF.pdf
        Args:
            z: predicted data
            logdet: computed logdet Jacobian
        Returns: mean negative log-likehood loss log(p(z)) = log(p(g(z)) + logdet Jg(z)
        """
        return -(self.normal.log_prob(z) + logdet).mean()

    def checkpoint(self):
        """Save model at the best epochs (with minimal loss)"""
        if self.checkpoint_dir is None:
            return

        if self.last_epoch_loss > self.min_epoch_loss:
            return

        self.min_epoch_loss = self.last_epoch_loss
        torch.save(self.flow.state_dict(), os.path.join(self.checkpoint_dir, f'flow.pt'))
        torch.save(self.optim.state_dict(), os.path.join(self.checkpoint_dir, f'optim.pt'))
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), os.path.join(self.checkpoint_dir, f'sched.pt'))

    def load_from_checkpoint(self, strict=False):
        """Load model from checkpoint"""
        if self.checkpoint_dir is None:
            return

        self.flow.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f'flow.pt')), strict=strict)
        self.optim.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f'optim.pt')))
        if self.scheduler is not None:
            self.scheduler.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f'sched.pt')))


class ResidualFlow(BaseFlowWrapper):
    def __init__(self, var_dim, cond_dim=None, n_layers=6, hid_dim=32, n_block_layers=2,
                 spnorm_coeff=0.97, logdet='unbias', n_backward_iters=100,
                 optimizer=None, batch_size=64, n_epochs=10, checkpoint_dir=None, device='cpu',
                 scheduler=None, **scheduler_kwargs):
        super().__init__(var_dim, cond_dim, n_layers, hid_dim, n_block_layers, spnorm_coeff, logdet, n_backward_iters,
                 optimizer, batch_size, n_epochs, checkpoint_dir, device, scheduler, **scheduler_kwargs)

    def fit(self, Y: torch.Tensor, X_cond: torch.Tensor = None):
        if X_cond is not None:
            td = TensorDataset(Y, X_cond)
        else:
            td = TensorDataset(Y)

        batches = DataLoader(td, batch_size=self.batch_size, shuffle=True)

        losses = []
        self.flow.train()
        for _ in tqdm(range(self.n_epochs), desc=f"Epoch"):
            epoch_loss = 0.0
            for data in batches:
                self.optim.zero_grad()

                y = data[0].to(self.device)
                if X_cond is not None:
                    x_cond = data[1].to(self.device)
                else:
                    x_cond = None

                z, logdet = self.flow.forward_process(y, x_cond)

                loss = self.loss(z, logdet)
                loss.backward()
                self.optim.step()

                epoch_loss += loss.item() * y.shape[0] / Y.shape[0]

            if self.scheduler is not None:
                self.scheduler.step()

            losses.append(epoch_loss)
            self.last_epoch_loss = epoch_loss
            self.checkpoint()

        self.load_from_checkpoint()
        return losses

    def sample(self, input: Union[torch.tensor, int], batch_size=None):
        N = None; X_cond = None
        if isinstance(input, int):
            N = input
            assert self.flow.cond_dim is None
        elif isinstance(input, torch.Tensor):
            X_cond = input
            assert self.flow.cond_dim == X_cond.shape[1]
        else:
            raise ValueError('Undefined input type')

        if X_cond is not None:
            td = TensorDataset(X_cond)
            bs = X_cond.shape[1] if batch_size is None else batch_size
            batches = DataLoader(td, batch_size=bs, shuffle=False)
        else:
            if batch_size is not None:
                bs = batch_size
                n_batches = N // bs
                remains = N - bs * n_batches
                batches = [bs for _ in range(n_batches)]
                if remains > 0:
                    batches += [remains]
            else:
                batches = [N]

        Ys = []
        self.flow.eval()
        with torch.no_grad():
            for data in tqdm(batches, 'batch'):
                if isinstance(data, int):
                    x_cond = None
                    size = data
                else:
                    x_cond = data[0].to(self.device)
                    size = x_cond.size(0)

                z = self.normal.sample((size,)).to(self.device)
                y, _ = self.flow.backward_process(z, x_cond)
                Ys.append(y.cpu())
            Y = torch.cat(Ys, dim=0)
        return Y
