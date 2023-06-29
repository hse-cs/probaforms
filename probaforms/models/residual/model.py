import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal

from .modules import ActNorm, InvertibleResLinear


class BaseResidualFlow(nn.Module):
    def __init__(self, var_dim, cond_dim=None, out_dim=None, n_layers=6):
        super().__init__()
        '''Pass concat [X, y] if conditioning, return only y'''

        self.var_dim = var_dim
        self.cond_dim = cond_dim
        self.in_dim = var_dim + cond_dim if cond_dim is not None else var_dim
        if self.cond_dim is not None:
            self.out_dim = self.var_dim
        else:
            self.out_dim = self.in_dim if out_dim is None else out_dim

        self.n_layers = n_layers
        self.device = 'cpu'
        self.net = None

    def forward_process(self, z, cond=None):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        for layer in self.net:
            z, log_df_dz = layer(z, log_df_dz)
        return z, log_df_dz

    def backward_process(self, z, cond=None):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        for layer in self.net[::-1]:
            z, log_df_dz = layer.backward(z, log_df_dz)
        return z, log_df_dz

    def to(self, device):
        super().to(device)
        self.device = device
        self.net = self.net.to(device)
        return self


class ResidualFlowModel(BaseResidualFlow):
    def __init__(self, var_dim, cond_dim=None, out_dim=None, n_layers=6, hid_dim=32, n_block_layers=2,
                 spnorm_coeff=0.97, logdet='unbias', n_backward_iters=100):
        super().__init__(var_dim, cond_dim, out_dim, n_layers)
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


# =========================== Wrappers ==========================


class BaseFlowWrapper(object):
    def __init__(self, model: BaseResidualFlow, optimizer, batch_size=64, n_epochs=100, scheduler=None, checkpoint_dir=None):
        self.flow = model
        self.optim = optimizer
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

        self.device = model.device

        self.min_epoch_loss = torch.inf
        self.last_epoch_loss = None

        self.var_dim = self.flow.var_dim
        self.cond_dim = self.flow.cond_dim
        self.in_dim = self.flow.in_dim
        self.out_dim = self.flow.out_dim

        self.mu = torch.zeros(self.out_dim, dtype=torch.float32, device=self.device)
        self.var = torch.eye(self.out_dim, dtype=torch.float32, device=self.device)
        self.normal = MultivariateNormal(self.mu, self.var)

    def fit(self, X: torch.tensor, cond: torch.tensor = None):
        """
        Fits flow
        Args:
            X: input objects tensor of shape (B, var_dim)
            cond: condition tensor of shape (B, cond_dim)
        """
        raise NotImplemented

    def sample(self):
        """Samples objects from normal noise"""
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
        if self.checkpoint_dir is None:
            return

        self.flow.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f'flow.pt')), strict=strict)
        self.optim.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f'optim.pt')))
        if self.scheduler is not None:
            self.scheduler.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f'sched.pt')))


class ResidualUnconditional(BaseFlowWrapper):
    def __init__(self, model: BaseResidualFlow, optimizer, batch_size=64, n_epochs=10, scheduler=None, checkpoint_dir=None):
        super().__init__(model, optimizer, batch_size, n_epochs, scheduler, checkpoint_dir)

    def fit(self, X: torch.Tensor):

        td = TensorDataset(X)
        batches = DataLoader(td, batch_size=self.batch_size, shuffle=True)

        losses = []
        self.flow.train()
        for _ in tqdm(range(self.n_epochs), desc=f"Epoch"):
            epoch_loss = 0.0
            for data in batches:
                self.optim.zero_grad()
                x0 = data[0].to(self.device)
                z, logdet = self.flow.forward_process(x0)  # (n, 2), (n) shapes
                loss = self.loss(z, logdet)
                loss.backward()
                self.optim.step()

                epoch_loss += loss.item() * x0.shape[0] / X.shape[0]

            if self.scheduler is not None:
                self.scheduler.step()

            losses.append(epoch_loss)

        return losses

    def sample(self, N: int, batched=None):
        """
        Samples N objects from estimated distribution
        Args:
            N: number of objects to sample
            batched: None if no batchification, else int -- batch_size
        Returns: new objects
        """
        if batched is not None:
            batch_size = batched
            n_batches = N // batch_size
            remains = N - batch_size * n_batches
            batches = [batch_size for _ in range(n_batches)]
            if remains > 0:
                batches += [remains]
        else:
            batches = [N]

        Xs = []
        self.flow.eval()
        with torch.no_grad():
            for size in tqdm(batches, 'batch'):
                z = self.normal.sample((size,)).to(self.device)
                x, _ = self.flow.backward_process(z)
                Xs.append(x.cpu())

            X = torch.cat(Xs, dim=0)
        return X


class ResidualConditional(BaseFlowWrapper):
    def __init__(self, model: BaseResidualFlow, optimizer, batch_size=64, n_epochs=10, scheduler=None, checkpoint_dir=None):
        super().__init__(model, optimizer, batch_size, n_epochs, scheduler, checkpoint_dir)

    def fit(self, Y: torch.Tensor, X_cond: torch.Tensor):
        td = TensorDataset(Y, X_cond)
        batches = DataLoader(td, batch_size=self.batch_size, shuffle=True)

        losses = []
        self.flow.train()
        for _ in tqdm(range(self.n_epochs), desc=f"Epoch"):
            epoch_loss = 0.0
            for y, x_cond in batches:
                self.optim.zero_grad()
                y = y.to(self.device)
                x_cond = x_cond.to(self.device)
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

    def sample(self, X_cond: torch.Tensor, batched=True):
        """Samples objects from condition of the X_cond's shape"""
        td = TensorDataset(X_cond)
        if batched is not None:
            batch_size = batched
        else:
            batch_size = X_cond.size(0)
        batches = DataLoader(td, batch_size=batch_size, shuffle=False)

        Ys = []
        self.flow.eval()
        with torch.no_grad():
            for x_cond in tqdm(batches, 'batch'):
                x_cond = x_cond[0].to(self.device)
                y = self.normal.sample((x_cond.size(0),)).to(self.device)
                y, _ = self.flow.backward_process(y, x_cond)
                Ys.append(y.cpu())

            Y = torch.cat(Ys, dim=0)
        return Y
