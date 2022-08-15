import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import os


if DEVICE:=os.environ.get('device'):
    DEVICE = torch.device(DEVICE)
else:
    DEVICE = torch.device('cpu')


class InvertibleLayer(nn.Module):
    '''
    Invertible function interface for normalizing flow models.

    Parameters:
    -----------
    var_size: int
        Input vector size.
    '''
    def __init__(self, var_size):
        super(InvertibleLayer, self).__init__()

        self.var_size = var_size


    def f(self, X, C):
        '''
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
        '''
        pass


    def g(self, X, C):
        '''
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
        '''
        pass



class NormalizingFlow(nn.Module):
    '''
    Normalizing Flow model interface.

    Parameters:
    -----------
    layers: list
        List of InvertibleLayers.
    prior: torch.distributions object
        Prior distribution of a latent variable.
    '''

    def __init__(self, layers, prior):
        super(NormalizingFlow, self).__init__()

        self.layers = nn.ModuleList(layers)
        self.prior = prior


    def log_prob(self, X, C):
        '''
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
        '''

        log_likelihood = None

        for layer in self.layers:
            X, change = layer.f(X, C)
            if log_likelihood is not None:
                log_likelihood = log_likelihood + change
            else:
                log_likelihood = change
        log_likelihood = log_likelihood + self.prior.log_prob(X)

        return log_likelihood.mean()


    def sample(self, C):
        '''
        Sample new objects based on the give conditions.

        Parameters:
        -----------
        C: torch.Tensor of shape [batch_size, cond_size] or Int
            Condition values or number of samples to generate.

        Return:
        -------
        X: torch.Tensor of shape [batch_size, var_size]
            Generated sample.
        '''

        if type(C) == type(1):
            n = C
            C = None
        else:
            n = len(C)

        X = self.prior.sample((n,))
        for layer in self.layers[::-1]:
            X = layer.g(X, C)

        return X
