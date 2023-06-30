import numpy as np
import torch
import pytest

from probaforms.models.interfaces import GenModel

from probaforms.models import PlainBackboneResidual
from probaforms.models import DDPM

def subclasses(cls):
    return set(cls.__subclasses__()).union(s for c in cls.__subclasses__() for s in subclasses(c))


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


@pytest.mark.parametrize("is_cond", [(True), (False)])
def test_ddpm(is_cond):
    n = 100
    X = torch.from_numpy(np.random.normal(size=(n, 5))).to(torch.float32)
    y = torch.from_numpy(np.random.normal(size=(n, 2))).to(torch.float32)
    len_y = y.shape[1]; len_X = X.shape[1]

    backbone_args_dict = {
        'var_dim': len_y if is_cond else len_y + len_X,
        'cond_input_dim': len_X if is_cond else None,
        'hid_dim': 128,
        'num_blocks': 6,
        'n_steps': 100,
        'steps_dim': 128,
        'steps_depth': 5,
        'steps_out_dim': 8,
        'use_cond_emb': is_cond,
        'cond_hid_dim': None,
        'cond_output_dim': 4 * len_X if is_cond else None,
    }

    backbone = PlainBackboneResidual(**backbone_args_dict)

    betas = (1e-3, 1e-2)
    beta_grid = 'linear'
    sigma_method = 'beta_wave'
    n_epochs = 100

    wrapper = DDPM(backbone, betas, beta_grid, sigma_method, n_epochs=n_epochs, batch_size=100)

    if is_cond:
        _ = wrapper.fit(y, X)
        _ = wrapper.sample(X).cpu()
    else:
        data = torch.cat([X, y], dim=1)
        _ = wrapper.fit(data)
        _ = wrapper.sample(500, batch_size=100).cpu()
