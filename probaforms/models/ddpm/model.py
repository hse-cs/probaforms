import os
import glob
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from .modules import BaseBackbone


def get_betas(min_beta=1e-5, max_beta=1e-2, beta_grid='linear', n_steps=200):
    """
    Beta's initialization
    Args:
        min_beta: minimal beta in grid
        max_beta: maximum beta in grid
        beta_grid: grid initialization strategy one of ['linear', 'square', 'sigmoid']
        n_steps: grid size
    Returns: beta grid
    """

    if beta_grid == 'linear':
        betas = torch.linspace(min_beta, max_beta, n_steps)
    elif beta_grid == 'square':
        betas = torch.linspace(min_beta ** 0.5, max_beta ** 0.5, n_steps) ** 2
    elif beta_grid == 'sigmoid':
        betas = torch.linspace(-6, 6, n_steps)
        betas = torch.sigmoid(betas) * (max_beta - min_beta) + min_beta
    else:
        raise NotImplemented(f'Beta grid type "{beta_grid}" is not implemented')
    return betas


class BaseDiffusion(nn.Module):
    """Base DDPM class"""
    def __init__(self, backbone: BaseBackbone):
        super().__init__()
        self.device = 'cpu'
        self.backbone = backbone
        self.var_dim = self.backbone.var_dim
        self.cond_dim = self.backbone.cond_dim
        self.in_dim = self.backbone.in_dim
        self.hid_dim = self.backbone.hid_dim
        self.n_steps = self.backbone.n_steps
        self.steps_dim = self.backbone.steps_dim

        self.alphas_cumprod = None

    def reverse_process(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor = None):
        """
        Predict noise eta_theta from xt by backbone model
        See paper, eq. (11), Algorithm 1
        Args:
            x: input objects tensor of shape (B, var_dim)
            t: noising step (timestep)
            cond: condition tensor of shape (B, cond_dim)
        Returns: eta_theta of shape (B, var_dim)
        """
        eta_theta = self.backbone(x, t, cond)
        return eta_theta

    def forward_process(self, yX0: torch.Tensor, t: torch.Tensor, eta: torch.Tensor = None):
        """
        Noise the input objects by t steps, see paper, 3.2 and eq. (4)
        Args:
            yX0: input object tensor of shape (B, var_dim) or (B, var_dim + cond_dim)
            t: noising step (timestep)
            eta: normal noise of shape (B, var_dim)
        Returns: noised objects in the close form of shape (B, var_dim)
        """

        if eta is None:
            eta = torch.randn_like(yX0)

        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1)
        mean = alpha_bar_t.sqrt() * yX0
        var = (1 - alpha_bar_t).sqrt()
        x_t_noised = mean + var * eta  # reparameterization trick
        return x_t_noised[:, :self.var_dim]

    def to(self, device):
        super().to(device)
        self.device = device
        self.backbone.device = device
        return self


class DiffusionMLP(BaseDiffusion):
    """DDPM multilayer model"""
    def __init__(self, backbone: BaseBackbone, betas=(1e-4, 1e-2), beta_grid='linear', sigma_method='beta'):
        """
        Args:
            backbone: backbone model to predict noise
            betas: tuple of (min_beta, bax_beta)
            beta_grid: grid initialization strategy one of ['linear', 'square', 'sigmoid']
            sigma_method: variance computation strategy, one of ['beta', 'beta_wave']
        """
        super().__init__(backbone)

        self.backbone = backbone
        self.n_steps = backbone.n_steps

        # forward process variances \beta_t
        # paper, 3.1: fix as the constants
        self.sigma_method = sigma_method
        self.min_beta = betas[0]
        self.max_beta = betas[1]
        self.beta_grid = beta_grid
        self.betas = get_betas(self.min_beta, self.max_beta, beta_grid, self.n_steps)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=-1)

        self.device = 'cpu'

    def get_sigma(self, t):
        """Variance computation from betas, see paper, 2"""
        if self.sigma_method == 'beta':
            # sigma_t^2 = beta_t, see paper, eq. (6)
            return self.betas[t].sqrt().view(-1, 1)
        elif self.sigma_method == 'beta_wave':
            # sigma_t^2 = beta_wave_t, see paper, eq. (7)
            _ = self.betas[t] * (1 - self.alphas_cumprod[t - 1])
            _ = _ / (1 - self.alphas_cumprod[t])
            sigma = _.sqrt()
            return sigma.view(-1, 1)
        else:
            raise ValueError('Unknown method')

    def one_step_denoise(self, xt, t_val, cond=None):
        """
        Denoise xt-1 <-- xt, see paper, Algorithm 2 Sampling
        Args:
            xt:  objects to denoise of shape (B, var_dim)
            t_val: timestamp of xt
            cond: condition tensor of shape (B, cond_dim)

        Returns: denoised x_t-1
        """

        t = t_val * torch.ones(size=(xt.shape[0], 1), dtype=torch.long).to(self.device)

        # predict eta_theta
        eta_theta = self.reverse_process(xt, t, cond)

        # get x_t-1
        alpha_t = self.alphas[t].view(-1, 1)
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1)
        coef = (1 - alpha_t) / (1 - alpha_bar_t).sqrt()
        x = (xt - coef * eta_theta) / alpha_t.sqrt()

        if t_val > 1:
            z = torch.randn((xt.shape[0], self.backbone.var_dim)).to(self.device)
            sigma_t = self.get_sigma(t)
            x = x + sigma_t * z
        return x

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return super().to(device)


# =========================== Wrappers ===========================


class BaseDiffusionWrapper(object):
    def __init__(self, model: BaseDiffusion, optimizer, loss_fn, batch_size=64, n_epochs=10, scheduler=None, checkpoint_dir=None):
        self.dfm = model
        self.optim = optimizer
        self.loss_fn = loss_fn
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

    def fit(self, X: torch.tensor, cond: torch.tensor = None):
        """
        Fits diffusion
        Args:
            X: input objects tensor of shape (B, var_dim)
            cond: condition tensor of shape (B, cond_dim)
        """
        raise NotImplemented

    def sample(self):
        """Samples objects from normal noise"""
        raise NotImplemented

    def checkpoint(self):
        if self.checkpoint_dir is None:
            return

        if self.last_epoch_loss > self.min_epoch_loss:
            return

        self.min_epoch_loss = self.last_epoch_loss
        torch.save(self.dfm.state_dict(), os.path.join(self.checkpoint_dir, f'dfm.pt'))
        torch.save(self.optim.state_dict(), os.path.join(self.checkpoint_dir, f'optim.pt'))
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), os.path.join(self.checkpoint_dir, f'sched.pt'))

    def load_from_checkpoint(self, strict=True):
        if self.checkpoint_dir is None:
            return

        self.dfm.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f'dfm.pt')), strict=strict)
        self.optim.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f'optim.pt')))
        if self.scheduler is not None:
            self.scheduler.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f'sched.pt')))


class DDPMUnconditional(BaseDiffusionWrapper):
    def __init__(self, model: BaseDiffusion, optimizer, loss_fn=nn.MSELoss(), batch_size=64, n_epochs=100, scheduler=None, checkpoint_dir=None):
        super().__init__(model, optimizer, loss_fn, batch_size, n_epochs, scheduler, checkpoint_dir)

    def fit(self, X: torch.Tensor):
        td = TensorDataset(X)
        batches = DataLoader(td, batch_size=self.batch_size, shuffle=True)

        losses = []
        self.dfm.train()
        for epoch in tqdm(range(self.n_epochs), desc=f"Epoch"):
            epoch_loss = 0.0
            for data in batches:
                self.optim.zero_grad()

                x0 = data[0].to(self.device)
                eta = torch.randn_like(x0)
                t = torch.randint(high=self.dfm.n_steps, size=(x0.shape[0], 1)).to(self.device)

                noisy_objs = self.dfm.forward_process(x0, t, eta)
                eta_theta = self.dfm.reverse_process(noisy_objs, t)
                loss = self.loss_fn(eta_theta, eta)

                loss.backward()
                self.optim.step()

                epoch_loss += loss.item() * x0.shape[0] / X.shape[0]

            if self.scheduler is not None and (epoch + 1) % 10 == 0:
                self.scheduler.step()

            losses.append(epoch_loss)
            self.last_epoch_loss = epoch_loss
            self.checkpoint()

        self.load_from_checkpoint()
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
        self.dfm.eval()
        with torch.no_grad():
            for size in tqdm(batches, 'batch'):
                x = torch.randn((size, self.dfm.backbone.var_dim)).to(self.device)
                for i in range(self.dfm.n_steps):
                    t_val = self.dfm.n_steps - i - 1
                    x = self.dfm.one_step_denoise(x, t_val)
                Xs.append(x.cpu())

            X = torch.cat(Xs, dim=0)
        return X


class DDPMConditional(BaseDiffusionWrapper):
    def __init__(self, model: BaseDiffusion, optimizer, loss_fn=nn.MSELoss(), batch_size=64, n_epochs=10,
                 scheduler=None, checkpoint_dir=None):
        super().__init__(model, optimizer, loss_fn, batch_size, n_epochs, scheduler, checkpoint_dir)

        # Condition embedding are used always if conditioning
        use_cond_emb = (model.cond_dim is not None)
        if use_cond_emb:
            self.cond_emb = nn.Sequential(
                nn.LazyLinear(16),
                nn.SiLU(),
                nn.Linear(16, self.dfm.cond_dim)
            ).to(self.device)
        else:
            self.cond_emb = nn.Identity().to(self.device)
        # self.cond_emb.requires_grad_(False)

    def fit(self, Y: torch.Tensor, X_cond: torch.Tensor):
        td = TensorDataset(Y, X_cond)
        batches = DataLoader(td, batch_size=self.batch_size, shuffle=True)

        losses = []
        self.dfm.train()
        for _ in tqdm(range(self.n_epochs), desc=f"Epoch"):
            epoch_loss = 0.0
            for y0, cond0 in batches:
                self.optim.zero_grad()

                y0 = y0.to(self.device); cond0 = cond0.to(self.device)
                t = torch.randint(high=self.dfm.n_steps, size=(y0.shape[0], 1)).to(self.device)
                cond0 = self.cond_emb(cond0)

                yX0 = torch.cat([y0, cond0], dim=1)
                eta = torch.randn_like(yX0)
                noisy_objs = self.dfm.forward_process(yX0, t, eta)

                eta_theta = self.dfm.reverse_process(noisy_objs, t, cond0)
                loss = self.loss_fn(eta_theta, eta[:, :self.dfm.var_dim])

                loss.backward()
                self.optim.step()

                epoch_loss += loss.item() * y0.shape[0] / Y.shape[0]

            if self.scheduler is not None:
                self.scheduler.step()

            losses.append(epoch_loss)

            self.last_epoch_loss = epoch_loss
            self.checkpoint()

        self.load_from_checkpoint()
        return losses

    def sample(self, X_cond: torch.Tensor, batched=None):
        """Samples objects from condition of the X_cond's shape"""
        td = TensorDataset(X_cond)
        if batched is not None:
            batch_size = batched
        else:
            batch_size = X_cond.size(0)
        batches = DataLoader(td, batch_size=batch_size, shuffle=False)

        Ys = []
        self.dfm.eval()
        with torch.no_grad():
            for x_cond in tqdm(batches, 'batch'):
                x_cond = self.cond_emb(x_cond[0].to(self.device))
                y = torch.randn((x_cond.size(0), self.dfm.backbone.var_dim)).to(self.device)
                for i in range(self.dfm.n_steps):
                    t_val = self.dfm.n_steps - i - 1
                    y = self.dfm.one_step_denoise(y, t_val, x_cond)
                Ys.append(y.cpu())

            Y = torch.cat(Ys, dim=0)
        return Y
