import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torch.utils.data import TensorDataset, DataLoader

from probaforms.models.interfaces import GenModel


# Conditional Normal Model

if DEVICE:=os.environ.get('device'):
    DEVICE = torch.device(DEVICE)
else:
    DEVICE = torch.device('cpu')

class Net(nn.Module):
    '''
    Neural network for the Conditional Normal Model.
    
    This network predicts the mean and covariance of a normal distribution conditioned on input data.
    Allows for both independent and full covariance structures.
    '''
    def __init__(self, var_size, cond_size, hidden=(10,), activation='tanh', independent_covariance=False):
        super(Net, self).__init__()
        
        self.independent_covariance = independent_covariance
        
        layers = []
        for i in range(len(hidden)):
            # add layer
            if i == 0:
                layers.append(nn.Linear(cond_size, hidden[i]))
            else:
                layers.append(nn.Linear(hidden[i - 1], hidden[i]))
                
            # add activation
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
            
        
        self.model = nn.Sequential(*layers)
        
        self.mu = nn.Linear(hidden[-1], var_size)
        self.log_sigma = nn.Linear(hidden[-1], var_size)
        self.out = nn.Linear(var_size, var_size)
            
    def forward(self, X, C):
        '''
        Forward pass through the network.
        
        Parameters:
        -----------
        X: torch.Tensor
            Input data samples.
        C: torch.Tensor
            Conditioning variables.
        
        Returns:
        --------
        x_tilde_: torch.Tensor
            Transformed data based on the predicted mean and covariance.
        inv: torch.Tensor or None
            Inverse transformation of the input data (only for full covariance case).
        mu: torch.Tensor
            Predicted mean of the distribution.
        sigma: torch.Tensor
            Predicted standard deviation (for independent covariance) or covariance matrix.
        '''
        C = self.model(C)
        mu = self.mu(C)
        sigma = torch.exp(self.log_sigma(C))
        
        eps = torch.randn(mu.shape).to(C.device)
        x_tilde_ = mu + eps * sigma
        
        if not self.independent_covariance:
            x_tilde_ = self.out(x_tilde_)
        
        inv = None
        if X is not None:
            inv = (X - self.out.bias) @ self.out.weight.T.inverse()

        return x_tilde_, inv, mu, sigma


class ConditionalNormal(GenModel):
    '''
    Conditional Normal model.

    This model estimates the parameters of a normal distribution (mean and covariance)
    conditional on some input data. It allows for the option of using an independent
    covariance matrix.

    Parameters:
    -----------
    use_independent_covariance: bool, optional (default=False)
        If True, the model assumes the covariance matrix is diagonal (independent variables).
        If False, the full covariance matrix is used.
    hidden: tuple of ints, optional (default=(10,))
        Number of neurons in hidden layers. Example: (10, 20, 15).
    activation: string, optional (default='tanh')
        Activation function of the hidden neurons. Possible values: 'tanh', 'relu'.
    batch_size: int, optional (default=32)
        Batch size.
    n_epochs: int, optional (default=10)
        Number of epochs for fitting the model.
    lr: float, optional (default=0.0001)
        Learning rate.
    weight_decay: float, optional (default=0)
        L2 regularization coefficient.
    verbose: int, optional (default=0)
        Controls the verbosity: the higher, the more messages.
        - >0: a progress bar for epochs is displayed;
        - >1: loss function value for each epoch is also displayed;
        - >2: loss function value for each batch is also displayed.
    '''
    def __init__(self, use_independent_covariance=False, hidden=(10,), activation='tanh', batch_size=32, n_epochs=10, 
                lr=0.0001, weight_decay=0, verbose=0):
        super(ConditionalNormal, self).__init__()
        
        self.independent_covariance = use_independent_covariance
        self.hidden = hidden
        self.activation = activation
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose
        
        self.opt = None
        
    def custom_loss(self, mu, sigma, x):
        loss = (x - mu)**2 / (2 * sigma**2) + torch.log(sigma)
        loss = torch.mean(loss)
        return loss
    
    def compute_loss(self, x_batch, cond_batch):
        _, inv, mu, sigma = self.model(x_batch, cond_batch)
        if self.independent_covariance:
            loss = self.custom_loss(mu=mu, sigma=sigma, x=x_batch)
        else:
            loss = self.custom_loss(mu=mu, sigma=sigma, x=inv)
        return loss
        
    def _model_init(self, X, C):
        
        var_size = X.shape[1]
        cond_size = C.shape[1]
        
        self.model = Net(var_size=var_size, cond_size=cond_size,
                        hidden=self.hidden, activation=self.activation, independent_covariance=self.independent_covariance)
        
        self.opt = torch.optim.Adam(list(self.model.parameters()),
                                    lr=self.lr, weight_decay=self.weight_decay)
 
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
        if C is None:
            C = torch.zeros(X.shape[0], 1)
        
        # model init
        self._model_init(X, C)
            
        # numpy to tensor, tensor to dataset
        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        C = torch.tensor(C, dtype=torch.float32, device=DEVICE)
        dataset = TensorDataset(X, C)

            
        self.model.train(True)
        self.loss_history = []
        
        _range = range(self.n_epochs) if self.verbose<1 else tqdm(range(self.n_epochs), unit='epoch')
        for epoch in _range:
            for i, batch in enumerate(DataLoader(dataset, batch_size=self.batch_size, shuffle=True)):

                X_batch = batch[0].to(DEVICE)
                C_batch = batch[1].to(DEVICE)
                

                # caiculate loss
                loss = self.compute_loss(X_batch, C_batch)

                # optimization step
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # caiculate and store loss
                self.loss_history.append(loss.detach().cpu())
                
                if self.verbose >= 2:
                    display_delta = max(1, (X.shape[0] // self.batch_size) // self.verbose)
                    if i % display_delta == 0:
                        _range.set_description(f"loss: {loss:.4f}")
            
            if self.verbose == 1:
                _range.set_description(f"loss: {loss:.4f}")    
    
    def sample(self, C=100):
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
        if type(C) != type(1):
            C = torch.tensor(C, dtype=torch.float, device=DEVICE)  
        else:
            C = torch.zeros(C, 1) 
            
        with torch.no_grad():
            x_tilde_, _, _, _ = self.model(None, C)
        return x_tilde_.cpu().detach().numpy()
    
