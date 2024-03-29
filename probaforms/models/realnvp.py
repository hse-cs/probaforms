import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm  # type: ignore

from probaforms.models.interfaces import GenModel
from probaforms.models.nflow import InvertibleLayer, NormalizingFlow


DEVICE = torch.device('cpu')
DistType = torch.distributions.distribution.Distribution


def gen_network(n_inputs: int,
                n_outputs: int,
                hidden: tuple[int, ...] = (10,),
                activation='tanh') -> nn.Module:
    model = nn.Sequential()
    for i in range(len(hidden)):
        
        # add layer
        if i == 0:
            alayer = nn.Linear(n_inputs, hidden[i])
        else:
            alayer = nn.Linear(hidden[i - 1], hidden[i])
        model.append(alayer)
        
        # add activation
        if activation == 'tanh':
            model.append(nn.Tanh())
        elif activation == 'relu':
            model.append(nn.ReLU())
        else:
            model.append(nn.ReLU())
    
    # output layer
    model.append(nn.Linear(hidden[-1], n_outputs))
    
    return model


class RealNVPLayer(InvertibleLayer):
    """
    Invertible RealNVP function for RealNVP normalizing flow model.

    Parameters:
    -----------
    var_size: int
        Input vector size.
    cond_size: int
        Conditional vector size.
    mask: torch.Tensor
        Tensor of {0, 1} to separate input vector components into two groups. Example: [0, 1, 0, 1].
    hidden: tuple of ints
        Number of neurons in hidden layers. Example: (10, 20, 15).
    activation: string
        Activation function of the hidden neurons. Possible values: 'tanh', 'relu'.
    """
    
    def __init__(self,
                 var_size: int,
                 cond_size: int,
                 mask: torch.Tensor,
                 hidden: tuple[int, ...] = (10,),
                 activation: str = 'tanh',
                 device: torch.device = DEVICE) -> None:
        super(RealNVPLayer, self).__init__(var_size=var_size)
        
        if activation not in ('tanh', 'relu'):
            warnings.warn("Unsupported activation function in Discriminator, setting to ReLU")

        self.device = device
        self.mask = mask.to(DEVICE)
        self.nn_t = gen_network(var_size + cond_size, var_size, hidden, activation)
        self.nn_s = gen_network(var_size + cond_size, var_size, hidden, activation)
    
    def f(self, X: torch.Tensor, C: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Implementation of forward pass.

        Parameters:
        -----------
        X: torch.Tensor of shape [batch_size, var_size]
            Input sample to transform.
        C: torch.Tensor of shape [batch_size, cond_size] or None
            Condition values.

        Return:
        -------
        new_X: torch.Tensor of shape [batch_size, var_size]
            Transformed X.
        log_det: torch.Tensor of shape [batch_size]
            Logarithm of the Jacobian determinant.
        """
        if C is not None:
            XC = torch.cat((X * self.mask[None, :], C), dim=1)
        else:
            XC = X * self.mask[None, :]
        
        T = self.nn_t(XC)
        S = self.nn_s(XC)
        
        X_new = (X * torch.exp(S) + T) * (1 - self.mask[None, :]) + X * self.mask[None, :]
        log_det = (S * (1 - self.mask[None, :])).sum(dim=-1)
        return X_new, log_det
    
    def g(self, X: torch.Tensor, C: torch.Tensor | None = None) -> torch.Tensor:
        """
        Implementation of backward (inverse) pass.

        Parameters:
        -----------
        X: torch.Tensor of shape [batch_size, var_size]
            Input sample to transform.
        C: torch.Tensor of shape [batch_size, cond_size] or None
            Condition values.

        Return:
        -------
        new_X: torch.Tensor of shape [batch_size, var_size]
            Transformed X.
        """
        if C is not None:
            XC = torch.cat((X * self.mask[None, :], C), dim=1)
        else:
            XC = X * self.mask[None, :]
        
        T = self.nn_t(XC)
        S = self.nn_s(XC)
        
        X_new = ((X - T) * torch.exp(-S)) * (1 - self.mask[None, :]) + X * self.mask[None, :]
        return X_new


class RealNVP(GenModel):
    """
    RealNVP normalizing flow model.

    Parameters:
    -----------
    n_layers: int
        Number of RealNVP layers.
    hidden: tuple of ints
        Number of neurons in hidden layers. Example: (10, 20, 15).
    activation: string
        Activation function of the hidden neurons. Possible values: 'tanh', 'relu'.
    batch_size: int
        Batch size.
    n_epochs: int
        Number of epoches for fitting the model.
    lr: float
        Learning rate.
    weight_decay: float
        L2 regularization coefficient.
    verbose: int
        Controls the verbosity: the higher, the more messages.
        - >0: a progress bar for epochs is displayed;
        - >1: loss function value for each epoch is also displayed;
        - >2: loss function value for each batch is also displayed.
    """
    
    def __init__(self,
                 n_layers: int = 8,
                 hidden: tuple[int, ...] = (10,),
                 activation: str = 'tanh',
                 batch_size: int = 32,
                 n_epochs: int = 10,
                 lr: float = 0.0001,
                 weight_decay: float = 0,
                 device: torch.device = DEVICE,
                 verbose: int = 0) -> None:
        super(RealNVP, self).__init__()

        self.n_layers = n_layers
        self.hidden = hidden
        self.activation = activation
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.verbose = verbose
        self.loss_history: list[torch.Tensor] = []
        
        self.prior: DistType | None = None
        self.nf: NormalizingFlow | None = None
        self.opt: torch.optim.Optimizer | None = None
    
    def _model_init(self, X: np.ndarray, C: np.ndarray | None) -> None:
        
        var_size = X.shape[1]
        if C is not None:
            cond_size = C.shape[1]
        else:
            cond_size = 0
        
        # init prior
        self.prior = self.prior or torch.distributions.MultivariateNormal(
            torch.zeros(var_size, device=self.device),
            torch.eye(var_size, device=self.device)
        )
        # init NF model and optimizer
        if self.nf is None:
            layers = []
            for i in range(self.n_layers):
                alayer = RealNVPLayer(var_size=var_size,
                                      cond_size=cond_size,
                                      mask=((torch.arange(var_size) + i) % 2),
                                      hidden=self.hidden,
                                      activation=self.activation,
                                      device=self.device)
                layers.append(alayer)
            
            self.nf = NormalizingFlow(layers=layers, prior=self.prior, device=self.device)
            self.opt = torch.optim.Adam(self.nf.parameters(),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)
    
    def fit(self, X: np.ndarray, C: np.ndarray | None = None):
        """
        Fit the model.

        Parameters:
        -----------
        X: numpy.ndarray of shape [batch_size, var_size]
            Input sample to transform.
        C: numpy.ndarray of shape [batch_size, cond_size] or None
            Condition values.
        """
        
        # model init
        self._model_init(X, C)
        assert self.nf is not None
        assert self.opt is not None
        assert self.prior is not None
        
        # numpy to tensor, tensor to dataset
        X_tens = torch.tensor(X, dtype=torch.float32, device=self.device)
        if C is not None:
            C_tens = torch.tensor(C, dtype=torch.float32, device=self.device)
            dataset = TensorDataset(X_tens, C_tens)
        else:
            dataset = TensorDataset(X_tens)

        loss: torch.Tensor = torch.zeros(1)
        _range = range(self.n_epochs) if self.verbose < 1 else tqdm(range(self.n_epochs), unit='epoch')

        for epoch in _range:
            for i, batch in enumerate(DataLoader(dataset, batch_size=self.batch_size, shuffle=True)):
                
                X_batch = batch[0].to(DEVICE)
                if C is not None:
                    C_batch = batch[1].to(DEVICE)
                else:
                    C_batch = None
                
                # calculate loss
                loss = -self.nf.log_prob(X_batch, C_batch)
                
                # optimization step
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                
                # calculate and store loss
                self.loss_history.append(loss.detach().cpu())
                
                if self.verbose >= 2:
                    display_delta = max(1, (X.shape[0] // self.batch_size) // self.verbose)
                    if i % display_delta == 0:
                        _range.set_description(f"loss: {loss:.4f}")
            
            if self.verbose == 1:
                _range.set_description(f"loss: {loss:.4f}")
    
    def sample(self, C: np.ndarray | int = 100) -> np.ndarray:
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
        assert self.nf is not None
        if isinstance(C, int):
            X = self.nf.sample(C)
        else:
            C_tens = torch.tensor(C, dtype=torch.float32, device=DEVICE)
            X = self.nf.sample(C_tens)

        return X.cpu().detach().numpy()
