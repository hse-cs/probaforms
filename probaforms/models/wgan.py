import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm  # type: ignore

from .interfaces import GenModel


DEVICE = torch.device("cpu")


class Generator(nn.Module):
    
    def __init__(self,
                 n_inputs: int,
                 n_outputs: int,
                 hidden: tuple[int, ...] | list[int] = (10,),
                 activation: str = 'tanh'
                 ) -> None:
        super(Generator, self).__init__()
        if activation not in ('tanh', 'relu'):
            warnings.warn("Unsupported activation function in Generator, setting to ReLU")
        
        self.model = nn.Sequential()
        for i in range(len(hidden)):
            # add layer
            if i == 0:
                alayer = nn.Linear(n_inputs, hidden[i])
            else:
                alayer = nn.Linear(hidden[i - 1], hidden[i])
            self.model.append(alayer)
            # add activation
            if activation == 'tanh':
                act = nn.Tanh()
            elif activation == 'relu':
                act = nn.ReLU()
            else:
                act = nn.ReLU()
            self.model.append(act)
        # output layer
        self.model.append(nn.Linear(hidden[-1], n_outputs))
    
    def forward(self, X: torch.Tensor, C: torch.Tensor | None = None) -> torch.Tensor:
        """
        Generator implementation.

        Parameters:
        -----------
        X: torch.Tensor of shape [batch_size, lat_size]
            Input sample to transform.
        C: torch.Tensor of shape [batch_size, cond_size] or None
            Condition values.

        Return:
        -------
        X_gen: torch.Tensor of shape [lat_size, n_outputs]
            Transformed X.
        """
        if C is None:
            Z = X
        else:
            Z = torch.cat((X, C), dim=1)
        X_gen = self.model(Z)
        return X_gen


class Discriminator(nn.Module):
    
    def __init__(self,
                 n_inputs: int,
                 hidden: tuple[int, ...] | list[int] = (10,),
                 activation: str = 'tanh'
                 ) -> None:
        super(Discriminator, self).__init__()
        if activation not in ('tanh', 'relu'):
            warnings.warn("Unsupported activation function in Discriminator, setting to ReLU")
        
        self.model = nn.Sequential()
        for i in range(len(hidden)):
            # add layer
            if i == 0:
                alayer = nn.Linear(n_inputs, hidden[i])
            else:
                alayer = nn.Linear(hidden[i - 1], hidden[i])
            self.model.append(alayer)
            # add activation
            if activation == 'tanh':
                act = nn.Tanh()
            elif activation == 'relu':
                act = nn.ReLU()
            else:
                act = nn.ReLU()
            self.model.append(act)
        # output layer
        self.model.append(nn.Linear(hidden[-1], 1))
    
    def forward(self, X: torch.Tensor, C: torch.Tensor | None = None) -> torch.Tensor:
        """
        Implementation of discriminator.

        Parameters:
        -----------
        X: torch.Tensor of shape [batch_size, feature_size]
            Input sample to transform.
        C: torch.Tensor of shape [batch_size, cond_size] or None
            Condition values.

        Return:
        -------
        pred: torch.Tensor of shape [lat_size, n_outputs]
            Transformed X.
        """
        if C is None:
            Z = X
        else:
            Z = torch.cat((X, C), dim=1)
        pred = self.model(Z)
        return pred


class ConditionalWGAN(GenModel):
    """
        Conditional Wasserstein GAN model.

        Parameters:
        -----------
        latent_dim: int
            Size of latent space.
        generator_hidden: tuple of ints
            Number of neurons in hidden layers of the generator. Example: (10, 20, 15).
        discriminator_hidden: tuple of ints
            Number of neurons in hidden layers of the discriminator. Example: (10, 20, 15).
        generator_activation: string
            Activation function of the hidden neurons of the generator. Possible values: 'tanh', 'relu'.
        discriminator_activation: string
            Activation function of the hidden neurons of the discriminator. Possible values: 'tanh', 'relu'.
        batch_size: int
            Batch size.
        n_epochs: int
            Number of epoches for fitting the model.
        lr: float
            Learning rate.
        weight_decay: float
            L2 regularization coefficient.
        n_critic: float
            The number of learning iterations of the discriminator per one iteration of the generator. n_critic > 1
        verbose: int
            Controls the verbosity: the higher, the more messages.
            - >0: a progress bar for epochs is displayed;
            - >1: loss functions values for each epoch are also displayed;
            - >2: loss functions values for each batch are also displayed.
    """
    
    def __init__(self,
                 latent_dim: int = 1,
                 generator_hidden: tuple[int, ...] = (100, 100),
                 discriminator_hidden: tuple[int, ...] = (100, 100),
                 generator_activation: str = 'relu',
                 discriminator_activation: str = 'relu',
                 batch_size: int = 32,
                 n_epochs: int = 1000,
                 lr: float = 0.00005,
                 weight_decay: float = 0,
                 n_critic: int = 5,
                 verbose: int = 0,
                 device: torch.device = DEVICE) -> None:
        super(ConditionalWGAN, self).__init__()

        self.generator_hidden = generator_hidden
        self.discriminator_hidden = discriminator_hidden
        self.generator_activation = generator_activation
        self.discriminator_activation = discriminator_activation
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.latent_dim = latent_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_critic = n_critic
        self.verbose = verbose
        self.device = device

        self.disc_loss_history: list[torch.Tensor] = []
        self.gen_loss_history: list[torch.Tensor] = []
        
        self.generator: Generator
        self.discriminator: Discriminator
        self.opt_gen: torch.optim.Optimizer
        self.opt_disc: torch.optim.Optimizer

    def _model_init(self, X: np.ndarray, C: np.ndarray | None = None) -> None:
        if C is None:
            c_len = 0
        else:
            c_len = C.shape[1]
        
        self.generator = Generator(
            n_inputs=self.latent_dim + c_len,
            n_outputs=X.shape[1],
            hidden=self.generator_hidden,
            activation=self.generator_activation
        )
        self.discriminator = Discriminator(
            n_inputs=X.shape[1] + c_len,
            hidden=self.discriminator_hidden,
            activation=self.discriminator_activation
        )
        
        self.opt_gen = torch.optim.RMSprop(
            self.generator.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.opt_disc = torch.optim.RMSprop(
            self.discriminator.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        self.to(self.device)

    def fit(self, X: np.ndarray, C: np.ndarray | None = None) -> 'ConditionalWGAN':
        """
        Fit the model.

        Parameters:
        -----------
        X: numpy.ndarray of shape [batch_size, var_size]
            Input sample of real data.
        C: numpy.ndarray of shape [batch_size, cond_size] or None
            Condition values.
        """
        
        # model init
        self._model_init(X, C)
        
        # numpy to tensor
        X_real = torch.tensor(X, dtype=torch.float, device=self.device)
        C_cond = None
        if C is None:
            dataset_real = TensorDataset(X_real)
        else:
            C_cond = torch.tensor(C, dtype=torch.float, device=self.device)
            dataset_real = TensorDataset(X_real, C_cond)

        assert isinstance(self.generator, Generator)
        assert isinstance(self.discriminator, Discriminator)
        assert isinstance(self.opt_gen, torch.optim.Optimizer)
        assert isinstance(self.opt_disc, torch.optim.Optimizer)
        
        # Turn on training
        self.generator.train(True)
        self.discriminator.train(True)
        
        self.disc_loss_history = []
        self.gen_loss_history = []
        
        iter_i = 0
        # Fit GAN
        _range = range(self.n_epochs) if self.verbose < 1 else tqdm(range(self.n_epochs), unit='epoch')
        loss_gen, loss_disc = torch.tensor(0), torch.tensor(0)
        for epoch in _range:
            for i, abatch in enumerate(DataLoader(dataset_real, batch_size=self.batch_size, shuffle=True)):
                
                # generate a batch of fake observations
                z_noise = torch.normal(0, 1, (len(abatch[0]), self.latent_dim), device=self.device)
                if C is None:
                    fake_batch = self.generator(z_noise, None)
                else:
                    fake_batch = self.generator(z_noise, abatch[1])
                
                if iter_i % self.n_critic != 0:
                    # Discriminator
                    if C is None:
                        loss_disc = -torch.mean(self.discriminator(abatch[0], None)) + torch.mean(
                            self.discriminator(fake_batch, None))
                    else:
                        loss_disc = -torch.mean(self.discriminator(abatch[0], abatch[1])) + torch.mean(
                            self.discriminator(fake_batch, abatch[1]))
                    # optimization step
                    self.opt_disc.zero_grad()
                    loss_disc.backward()
                    self.opt_disc.step()
                    
                    # Clip weights of discriminator
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)
                    
                    if self.verbose >= 2:
                        display_delta = max(1, (X.shape[0] // self.batch_size) // self.verbose)
                        if i % display_delta == 0:
                            loss_to_display = (loss_gen.detach().numpy(), loss_disc.detach().numpy())
                            _range.set_description(
                                f"G loss: {loss_to_display[0]:.4f}, D loss: {loss_to_display[1]:.4f}")
                
                else:
                    # Generator
                    if C is None:
                        loss_gen = -torch.mean(self.discriminator(fake_batch, None))
                    else:
                        loss_gen = -torch.mean(self.discriminator(fake_batch, abatch[1]))
                    # optimization step
                    self.opt_gen.zero_grad()
                    loss_gen.backward()
                    self.opt_gen.step()
                    
                    if (self.verbose >= 2) and (epoch != 0):
                        display_delta = max(1, (X.shape[0] // self.batch_size) // self.verbose)
                        if i % display_delta == 0:
                            loss_to_display = (loss_gen.detach().cpu().numpy(), loss_disc.detach().cpu().numpy())
                            _range.set_description(
                                f"G loss: {loss_to_display[0]:.4f}, D loss: {loss_to_display[1]:.4f}")

                iter_i += 1

            if self.verbose == 1:
                loss_to_display = (loss_gen.detach().numpy(), loss_disc.detach().numpy())
                _range.set_description(f"G loss: {loss_to_display[0]:.4f}, D loss: {loss_to_display[1]:.4f}")
            
            # calculate and store loss after an epoch
            Z_noise = torch.normal(0, 1, (len(X_real), self.latent_dim), device=self.device)
            if C_cond is None:
                X_fake = self.generator(Z_noise, None)
            else:
                X_fake = self.generator(Z_noise, C_cond)
            if C is None:
                gen_loss_epoch = - torch.mean(self.discriminator(X_fake, None))
                disc_loss_epoch = torch.mean(self.discriminator(X_real, None)) + gen_loss_epoch
            else:
                gen_loss_epoch = - torch.mean(self.discriminator(X_fake, C_cond))
                disc_loss_epoch = torch.mean(self.discriminator(X_real, C_cond)) + gen_loss_epoch
            self.disc_loss_history.append(disc_loss_epoch.detach().cpu())
            self.gen_loss_history.append(gen_loss_epoch.detach().cpu())
        
        # Turn off training
        self.generator.train(False)
        self.discriminator.train(False)

        return self
    
    def sample(self, C: int | np.ndarray = 10) -> np.ndarray:
        """
        Sample new objects based on the give conditions.

        Parameters:
        -----------
        C: numpy.ndarray of shape [batch_size, cond_size] or Int
            Condition values or number of samples to generate.

        Return:
        -------
        X: numpy.ndarray of shape [batch_size, var_size]
            Generated sample.
        """
        self.generator.eval()
        if isinstance(C, int):
            Z = torch.normal(0, 1, (C, self.latent_dim), device=self.device)
            X = self.generator(Z, None).detach().cpu().numpy()
        else:
            Z = torch.normal(0, 1, (len(C), self.latent_dim), device=self.device)
            C_cond = torch.tensor(C, dtype=torch.float, device=self.device)
            X = self.generator(Z, C_cond).detach().cpu().numpy()
        return X


__all__ = ['ConditionalWGAN']
