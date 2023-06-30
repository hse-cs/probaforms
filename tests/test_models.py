import numpy as np
import pytest
import torch
import itertools

from probaforms.models.interfaces import GenModel
from probaforms.models import ResidualFlow


def subclasses(cls):
    return set(cls.__subclasses__()).union(s for c in cls.__subclasses__() for s in subclasses(c))


def gen_data(bias=0, N=500):
    X = np.linspace(bias, bias + 5, N).reshape(-1, 1)
    mu = np.exp(-X + bias)
    eps = np.random.normal(0, 1, X.shape)
    sigma = 0.05 * (X - bias + 0.5)
    X = torch.from_numpy(X).to(torch.float32)
    y = torch.from_numpy(mu + eps * sigma).to(torch.float32)
    return X, y


@pytest.mark.parametrize("model", subclasses(GenModel))
def test_with_conditions(model):
    n = 100
    X = np.random.normal(size=(n, 5))
    C = np.random.normal(size=(n, 3))
    gen = model()
    gen.fit(X, C)
    X_gen = gen.sample(C)
    assert X_gen.shape == X.shape


@pytest.mark.parametrize("model", subclasses(GenModel))
def test_without_conditions(model):
    n = 100
    X = np.random.normal(size=(n, 5))
    gen = model()
    gen.fit(X, C=None)
    X_gen = gen.sample(C=n)
    assert X_gen.shape == X.shape


logdets = ['exact', 'fixed', 'unbias']
is_conds = [True, False]
@pytest.mark.parametrize("logdet,is_cond", list(itertools.product(logdets, is_conds)))
def test_resflow(logdet, is_cond):
    n = 100
    X = torch.from_numpy(np.random.normal(size=(n, 5))).to(torch.float32)
    y = torch.from_numpy(np.random.normal(size=(n, 3))).to(torch.float32)
    len_y = y.shape[1]; len_X = X.shape[1]

    flow_args_dict = {
        'var_dim': len_y if is_cond else len_y + len_X,
        'cond_dim': len_X if is_cond else None,
        'hid_dim': 16,
        'n_block_layers': 3,
        'n_layers': 3,
        'spnorm_coeff': 0.95,
        'n_backward_iters': 100,
        'logdet': logdet,
    }

    wrapper = ResidualFlow(**flow_args_dict, n_epochs=50, batch_size=100)

    if is_cond:
        _ = wrapper.fit(y, X)
        _ = wrapper.sample(X).cpu()
    else:
        data = torch.cat([X, y], dim=1)
        _ = wrapper.fit(data)
        _ = wrapper.sample(500, batch_size=100).cpu()
