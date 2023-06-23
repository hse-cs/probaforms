import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from .interfaces import GenModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):

    def __init__(self, n_inputs, n_outputs, hidden=(10,), activation='tanh'):
        super(Generator, self).__init__()

        self.model = nn.Sequential()
        for i in range(len(hidden)):
            # add layer
            if i == 0:
                alayer = nn.Linear(n_inputs, hidden[i])
            else:
                alayer = nn.Linear(hidden[i - 1], hidden[i])
            self.model.append(alayer)
            # self.model.append(nn.BatchNorm1d(hidden[i]))
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
        '''
        if C is None:
            Z = X
        else:
            Z = torch.cat((X, C), dim=1)
        X_gen = self.model(Z)
        return X_gen


class Discriminator(nn.Module):

    def __init__(self, n_inputs, hidden=(10,), activation='tanh'):
        super(Discriminator, self).__init__()

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

    def forward(self, X, C=None):
        '''
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
        '''
        if C is None:
            Z = X
        else:
            Z = torch.cat((X, C), dim=1)
        pred = self.model(Z)
        return pred


class ConditionalWGAN(GenModel):
    '''
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
        '''

    def __init__(self, latent_dim=1,
                 generator_hidden=(100, 100), discriminator_hidden=(100, 100),
                 generator_activation='relu', discriminator_activation='relu',
                 batch_size=32, n_epochs=1000, lr=0.00005, weight_decay=0, n_critic=5):
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

        self.generator = None
        self.discriminator = None

        self.opt_gen = None
        self.opt_disc = None


    def _model_init(self, X, C=None):

        if C is None:
            c_len = 0
        else:
            c_len = C.shape[1]

        self.generator = Generator(n_inputs=self.latent_dim+c_len,
                                   n_outputs=X.shape[1],
                                   hidden=self.generator_hidden,
                                   activation=self.generator_activation)
        self.discriminator = Discriminator(n_inputs=X.shape[1]+c_len,
                                           hidden=self.discriminator_hidden,
                                           activation=self.discriminator_activation)

        self.opt_gen = torch.optim.RMSprop(self.generator.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.opt_disc = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.generator.to(DEVICE)
        self.discriminator.to(DEVICE)


    def fit(self, X, C=None):
        '''
        Fit the model.

        Parameters:
        -----------
        X: numpy.ndarray of shape [batch_size, var_size]
            Input sample of real data.
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
        self.generator.train(True)
        self.discriminator.train(True)

        self.disc_loss_history = []
        self.gen_loss_history = []

        iter_i = 0
        # Fit GAN
        for epoch in range(self.n_epochs):
            for i, abatch in enumerate(DataLoader(dataset_real, batch_size=self.batch_size, shuffle=True)):

                # generate a batch of fake observations
                z_noise = torch.normal(0, 1, (len(abatch[0]), self.latent_dim))
                if C is None:
                    fake_batch = self.generator(z_noise, None)
                else:
                    fake_batch = self.generator(z_noise, abatch[1])

                if iter_i % self.n_critic != 0:
                    ### Discriminator
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
                else:
                    ### Generator
                    if C is None:
                        loss_gen = -torch.mean(self.discriminator(fake_batch, None))
                    else:
                        loss_gen = -torch.mean(self.discriminator(fake_batch, abatch[1]))
                    # optimization step
                    self.opt_gen.zero_grad()
                    loss_gen.backward()
                    self.opt_gen.step()

                iter_i += 1

            # calculate and store loss after an epoch
            Z_noise = torch.normal(0, 1, (len(X_real), self.latent_dim))
            if C is None:
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
            Z = torch.normal(0, 1, (len(C), self.latent_dim))
            C = torch.tensor(C, dtype=torch.float, device=DEVICE)
            X = self.generator(Z, C).cpu().detach().numpy()
        else:
            Z = torch.normal(0, 1, (C, self.latent_dim))
            X = self.generator(Z, None).cpu().detach().numpy()
        return X
