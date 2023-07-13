import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import mafflow as fnn
from interfaces import GenModel


if DEVICE:=os.environ.get('device'):
    DEVICE = torch.device(DEVICE)
else:
    DEVICE = torch.device('cpu')


class MAF(GenModel):
    '''
    Conditional MAF normalizing flow model.
    Parameters:
    -----------
    n_layers: int
        Number of CMAF layers.
    hidden: int
        Number of neurons in hidden layers.
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
    cond: bool
        True for CMAF, False for MAF.
    num_cond_inputs: None | int
        Dimension of conditional input.
    '''

    def __init__(self, n_layers=5, hidden=100, activation='tanh',
                       batch_size=32, n_epochs=10, lr=0.001, weight_decay = 1e-6,
                       num_cond_inputs = None, cond = False):
        super(MAF, self).__init__()

        self.n_layers = n_layers
        self.num_cond_inputs = num_cond_inputs
        self.hidden = hidden
        self.activation = activation
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.cond = cond
        self.n_epochs = n_epochs


    def _model_init(self, X):
        # construct MAF from MADEs
        modules = []
        for _ in range(self.n_layers):
                size = X.shape[1]
                modules += [
                    fnn.MADE(size, self.hidden, self.num_cond_inputs, act=self.activation),
                    fnn.BatchNormFlow(size),
                    fnn.Reverse(size)
                ]

        self.model = fnn.FlowSequential(*modules)

        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)

        self.model.to(DEVICE)


    def fit(self, X, C=None):
        '''
        Fit the model.
        Parameters:
        -----------
        X: numpy.ndarray of shape [batch_size, var_size]
            Input sample to transform.
        C: numpy.ndarray of shape [batch_size, cond_size] or None
            Condition values.
        '''

        # Changing dimensions if it's equal to one by adding noise
        if X.shape[1] == 1:
            X = np.hstack([X, np.random.normal(0, 1, X.shape)])
            self.noise = True
        else:
            self.noise = False

        # Model init
        self._model_init(X)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        global_step = 0

        # Preparing data
        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        if C is not None:
            C = torch.tensor(C, dtype=torch.float32, device=DEVICE)
            dataset = TensorDataset(X, C)
        else:
            dataset = TensorDataset(X)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training
        for epoch in tqdm(range(self.n_epochs), desc= "Training epoch"):
            self.model.train()
            train_loss = 0

            for batch_idx, data in enumerate(loader):
                if isinstance(data, list):
                    if len(data) > 1:
                        cond_data = data[1].float()
                        cond_data = cond_data.to(DEVICE)
                    else:
                        cond_data = None

                    data = data[0]
                
                data = data.to(DEVICE)
                optimizer.zero_grad()
                loss = -self.model.log_probs(data, cond_data).mean()
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                global_step += 1
                
            for module in self.model.modules():
                if isinstance(module, fnn.BatchNormFlow):
                    module.momentum = 0

            if self.cond:
                with torch.no_grad():
                    self.model(loader.dataset.tensors[0].to(data.device),
                               loader.dataset.tensors[1].to(data.device).float())
            else:
                with torch.no_grad():
                    self.model(loader.dataset.tensors[0].to(data.device))


            for module in self.model.modules():
                if isinstance(module, fnn.BatchNormFlow):
                    module.momentum = 1

    def sample(self, C=None, num_samples=100):
        '''
        Sample new objects based on the give conditions for CMAF.
        Sample several number of samples for MAF.
        Parameters:
        -----------
        C: numpy.ndarray of shape [batch_size, cond_size] | Int | None
            Condition values or number of samples to generate.
        num_samples: int
            Number of samples for MAF

        Return:
        -------
        X: numpy.ndarray of shape [batch_size, var_size]
            Generated sample.
        '''
        if type(C) != type(1):
            C = torch.tensor(C, dtype=torch.float32, device=DEVICE)
            num_samples = C.shape[0]
        else:
            C=None
        X = self.model.sample(num_samples=num_samples, cond_inputs=C).cpu().detach().numpy()
        return X[:, 0] if self.noise else X
