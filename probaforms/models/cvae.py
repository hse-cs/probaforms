import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from .interfaces import GenModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device('cpu')
DEVICE


class Encoder(nn.Module):
    def __init__(self, n_inputs, lat_size, hidden=(10,), activation='tanh'):
        super(Encoder, self).__init__()

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

        self.mu = nn.Linear(hidden[-1], lat_size)
        self.log_sigma = nn.Linear(hidden[-1], lat_size)


    def forward(self, X, C=None):
        '''
        Implementation of encoding.

        Parameters:
        -----------
        X: torch.Tensor of shape [batch_size, var_size]
            Input sample to transform.
        C: torch.Tensor of shape [batch_size, cond_size] or None
            Condition values.

        Return:
        -------
        mu: torch.Tensor of shape [batch_size, lat_size]
            Transformed X.
        log_sigma: torch.Tensor of shape [batch_size, lat_size]
            Transformed X.
        '''
        if C is None:
            Z = X
        else:
            Z = torch.cat((X, C), dim=1)
        Z = self.model(Z)
        mu = self.mu(Z)
        log_sigma = self.log_sigma(Z)
        return mu, log_sigma



class Decoder(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden=(10,), activation='tanh'):
        super(Decoder, self).__init__()

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


    def forward(self, X, C=None):
        '''
        Implementation of decoding.

        Parameters:
        -----------
        X: torch.Tensor of shape [batch_size, lat_size]
            Input sample to transform.
        C: torch.Tensor of shape [batch_size, cond_size] or None
            Condition values.

        Return:
        -------
        X_rec: torch.Tensor of shape [lat_size, n_outputs]
            Transformed X.
        '''
        if C is None:
            Z = X
        else:
            Z = torch.cat((X, C), dim=1)
        X_rec = self.model(Z)
        return X_rec


class CVAE(GenModel):
    '''
    Conditional VAE model.

    Parameters:
    -----------
    latent_dim: int
        Size of latent space of the VAE.
    hidden: tuple of ints
        Number of neurons in hidden layers of the decoder and encoder. Example: (10, 20, 15).
    activation: string
        Activation function of the hidden neurons of the decoder and encoder. Possible values: 'tanh', 'relu'.
    batch_size: int
        Batch size.
    n_epochs: int
        Number of epoches for fitting the model.
    lr: float
        Learning rate.
    weight_decay: float
        L2 regularization coefficient.
    KL_weight: float
        Weight of variational part of the loss function.
    '''

    def __init__(self, latent_dim=2, hidden=(10,), activation='tanh', batch_size=32, n_epochs=10, lr=0.0001,
                 weight_decay=0, KL_weight=0.001):
        super(CVAE, self).__init__()

        self.lat_size = latent_dim
        self.hidden = hidden
        self.activation = activation
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.latent_dim = latent_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.KL_weight = KL_weight

        self.criterion = nn.MSELoss()
        self.opt = None


    def _model_init(self, X, C=None):

        if C is None:
            c_len = 0
        else:
            c_len = C.shape[1]

        self.encoder = Encoder(n_inputs=X.shape[1]+c_len,
                               lat_size=self.lat_size,
                               hidden=self.hidden,
                               activation=self.activation)
        self.decoder = Decoder(n_inputs=self.lat_size+c_len,
                               n_outputs=X.shape[1],
                               hidden=self.hidden,
                               activation=self.activation)

        self.opt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                    lr=self.lr, weight_decay=self.weight_decay)

        self.encoder.to(DEVICE)
        self.decoder.to(DEVICE)

    def sample_z(self, mu, log_sigma):
        eps = torch.randn(mu.shape).to(DEVICE)
        return mu + torch.exp(log_sigma / 2) * eps

    def custom_loss(self, x, rec_x, mu, log_sigma):
        KL = torch.mean(-0.5 * torch.sum(1 + log_sigma - mu ** 2 - log_sigma.exp(), dim=1), dim=0)
        recon_loss = self.criterion(x, rec_x)
        return KL * self.KL_weight + recon_loss

    def compute_loss(self, x_batch, cond_batch):

        mu, log_sigma = self.encoder(x_batch, cond_batch)
        z_batch = self.sample_z(mu, log_sigma)
        x_batch_rec = self.decoder(z_batch, cond_batch)

        loss = self.custom_loss(x_batch, x_batch_rec, mu, log_sigma)

        return loss

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

        # numpy to tensor
        X_real = torch.tensor(X, dtype=torch.float, device=DEVICE)
        if C is None:
            dataset_real = TensorDataset(X_real)
        else:
            C_cond = torch.tensor(C, dtype=torch.float, device=DEVICE)
            dataset_real = TensorDataset(X_real, C_cond)

        # Turn on training
        self.encoder.train(True)
        self.decoder.train(True)

        self.loss_history = []

        # Fit CVAE
        for epoch in range(self.n_epochs):
            for i, abatch in enumerate(DataLoader(dataset_real, batch_size=self.batch_size, shuffle=True)):
                # caiculate loss
                if C is None:
                    loss = self.compute_loss(abatch[0], None)
                else:
                    loss = self.compute_loss(abatch[0], abatch[1])

                # optimization step
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            # caiculate and store loss after an epoch
            if C is None:
                loss_epoch = self.compute_loss(X_real, None)
            else:
                loss_epoch = self.compute_loss(X_real, C_cond)
            self.loss_history.append(loss_epoch.detach().cpu())

        # Turn off training
        self.encoder.train(False)
        self.decoder.train(False)

        return self

    def sample(self, C=10):
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
            Z = torch.normal(0, 1, (len(C), self.lat_size))
            C = torch.tensor(C, dtype=torch.float, device=DEVICE)
            X = self.decoder(Z, C).cpu().detach().numpy()
        else:
            Z = torch.normal(0, 1, (C, self.lat_size))
            X = self.decoder(Z, None).cpu().detach().numpy()
        return X