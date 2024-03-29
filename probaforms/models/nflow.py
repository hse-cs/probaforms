import typing as tp

import torch
import torch.nn as nn


DEVICE = torch.device('cpu')

ModuleType = tp.TypeVar('ModuleType', bound=nn.Module)


class InvertibleLayer(nn.Module):
    """
    Invertible function interface for normalizing flow models.

    Parameters:
    -----------
    var_size: int
        Input vector size.
    """
    
    def __init__(self, var_size: int):
        super(InvertibleLayer, self).__init__()
        self.var_size = var_size
    
    def f(self, X: torch.Tensor, C: torch.Tensor | None = None):
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
        pass
    
    def g(self, X: torch.Tensor, C: torch.Tensor | None = None):
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
        pass


class NormalizingFlow(nn.Module):
    """
    Normalizing Flow model interface.

    Parameters:
    -----------
    layers: list
        List of InvertibleLayers.
    prior: torch.distributions object
        Prior distribution of a latent variable.
    """
    
    def __init__(self,
                 layers: list[ModuleType],
                 prior: torch.distributions.distribution.Distribution,
                 device: torch.device) -> None:
        super(NormalizingFlow, self).__init__()
        
        self.layers = nn.ModuleList(layers)
        self.prior = prior
        self.device = device
        self.layers.to(self.device)
    
    def log_prob(self, X: torch.Tensor, C: torch.Tensor | None = None) -> torch.Tensor:
        """
        Calculates the loss function.

        Parameters:
        -----------
        X: torch.Tensor of shape [batch_size, var_size]
            Input sample to transform.
        C: torch.Tensor of shape [batch_size, cond_size] or None
            Condition values.

        Return:
        -------
        log_likelihood: torch.Tensor
            Calculated log likelihood.
        """
        
        log_likelihood = None
        X_tens = X.to(torch.float).to(self.device)
        C_tens = C if C is None else C.to(torch.float).to(self.device)
        
        for layer in self.layers:
            X_tens, change = layer.f(X_tens, C_tens)
            if log_likelihood is not None:
                log_likelihood = log_likelihood + change
            else:
                log_likelihood = change
        log_likelihood = log_likelihood + self.prior.log_prob(X_tens)
        
        return log_likelihood.mean()
    
    def sample(self, C: torch.Tensor | int) -> torch.Tensor:
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
        
        if isinstance(C, int):
            n, C_cond = C, None
        else:
            n, C_cond = len(C), C
        size = torch.Size((n,))

        X = self.prior.sample(size).to(self.device)

        for layer in self.layers[::-1]:
            X = layer.g(X, C_cond)

        return X
