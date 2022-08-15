import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import os

from probaforms.models.interfaces import GenModel
from probaforms.models.nflow import InvertibleLayer, NormalizingFlow


if DEVICE:=os.environ.get('device'):
    DEVICE = torch.device(DEVICE)
else:
    DEVICE = torch.device('cpu')



def gen_network(n_inputs, n_outputs, hidden=(10,), activation='tanh'):

    model = nn.Sequential()
    for i in range(len(hidden)):

        # add layer
        if i == 0:
            alayer = nn.Linear(n_inputs, hidden[i])
        else:
            alayer = nn.Linear(hidden[i-1], hidden[i])
        model.append(alayer)

        # add activation
        if activation == 'tanh':
            act = nn.Tanh()
        elif activation == 'relu':
            act = nn.ReLU()
        else:
            act = nn.ReLU()
        model.append(act)

    # output layer
    model.append(nn.Linear(hidden[-1], n_outputs))

    return model



class RealNVPLayer(InvertibleLayer):
    '''
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
    '''

    def __init__(self, var_size, cond_size, mask, hidden=(10,), activation='tanh'):
        super(RealNVPLayer, self).__init__(var_size=var_size)

        self.mask = mask.to(DEVICE)
        self.nn_t = gen_network(var_size + cond_size, var_size, hidden, activation)
        self.nn_s = gen_network(var_size + cond_size, var_size, hidden, activation)


    def f(self, X, C=None):
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
        if C is not None:
            XC = torch.cat((X * self.mask[None, :], C), dim=1)
        else:
            XC = X * self.mask[None, :]

        T = self.nn_t(XC)
        S = self.nn_s(XC)

        X_new = (X * torch.exp(S) + T) * (1 - self.mask[None, :]) + X * self.mask[None, :]
        log_det = (S * (1 - self.mask[None, :])).sum(dim=-1)
        return X_new, log_det


    def g(self, X, C=None):
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
        if C is not None:
            XC = torch.cat((X * self.mask[None, :], C), dim=1)
        else:
            XC = X * self.mask[None, :]

        T = self.nn_t(XC)
        S = self.nn_s(XC)

        X_new = ((X - T) * torch.exp(-S)) * (1 - self.mask[None, :]) + X * self.mask[None, :]
        return X_new



class RealNVP(GenModel):
    '''
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
    '''

    def __init__(self, n_layers=8, hidden=(10,), activation='tanh',
                       batch_size=32, n_epochs=10, lr=0.0001, weight_decay=0):
        super(RealNVP, self).__init__()

        self.n_layers = n_layers
        self.hidden = hidden
        self.activation = activation
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay

        self.prior = None
        self.nf = None
        self.opt = None

        self.loss_history = []


    def _model_init(self, X, C):

        var_size = X.shape[1]
        if C is not None:
            cond_size = C.shape[1]
        else:
            cond_size = 0

        # init prior
        if self.prior is None:
            self.prior = torch.distributions.MultivariateNormal(torch.zeros(var_size, device=DEVICE),
                                                                torch.eye(var_size, device=DEVICE))
        # init NF model and optimizer
        if self.nf is None:

            layers = []
            for i in range(self.n_layers):
                alayer = RealNVPLayer(var_size=var_size,
                                      cond_size=cond_size,
                                      mask=((torch.arange(var_size) + i) % 2),
                                      hidden=self.hidden,
                                      activation=self.activation)
                layers.append(alayer)

            self.nf = NormalizingFlow(layers=layers, prior=self.prior).to(DEVICE)
            self.opt = torch.optim.Adam(self.nf.parameters(),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)


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

        # model init
        self._model_init(X, C)

        # numpy to tensor, tensor to dataset
        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        if C is not None:
            C = torch.tensor(C, dtype=torch.float32, device=DEVICE)
            dataset = TensorDataset(X, C)
        else:
            dataset = TensorDataset(X)

        criterion = nn.MSELoss()

        for epoch in range(self.n_epochs):
            for batch in DataLoader(dataset, batch_size=self.batch_size, shuffle=True):

                X_batch = batch[0].to(DEVICE)
                if C is not None:
                    C_batch = batch[1].to(DEVICE)
                else:
                    C_batch = None

                # caiculate loss
                loss = -self.nf.log_prob(X_batch, C_batch)

                # optimization step
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # caiculate and store loss
                self.loss_history.append(loss.detach().cpu())


    def sample(self, C=100):
        '''
        Sample new objects based on the give conditions.

        Parameters:
        -----------
        C: numpy.ndarray of shape [batch_size, cond_size] or Int
            Condition values or number of samples to generate.

        Return:
        -------
        X: numpy.ndarray of shape [batch_size, var_size]
            Generated sample.
        '''
        if type(C) != type(1):
            C = torch.tensor(C, dtype=torch.float32, device=DEVICE)
        X = self.nf.sample(C).cpu().detach().numpy()
        return X
