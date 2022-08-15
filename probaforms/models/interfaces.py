import torch
import torch.nn as nn
import os


class GenModel(nn.Module):
    '''
    Conditional Generative Model interface.
    '''

    def __init__(self):
        super(GenModel, self).__init__()


    def fit(self, X, C):
        '''
        Fit method.

        Parameters:
        -----------
        X: torch.Tensor of shape [batch_size, var_size]
            Input sample to transform.
        C: torch.Tensor of shape [batch_size, cond_size] or None
            Condition values.
        '''
        pass


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
        pass
