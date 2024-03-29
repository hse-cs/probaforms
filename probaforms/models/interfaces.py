import torch.nn as nn


class GenModel(nn.Module):
    """
    Conditional Generative Model interface.
    """
    
    def __init__(self):
        super(GenModel, self).__init__()
    
    def fit(self, X, C):
        """
        Fit method.

        Parameters:
        -----------
        X: numpy.ndarray of shape [batch_size, var_size]
            Input sample of real data.
        C: numpy.ndarray of shape [batch_size, cond_size] or None
            Condition values.
        """
        pass
    
    def sample(self, C):
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
        pass
