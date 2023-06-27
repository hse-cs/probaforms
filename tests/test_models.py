import numpy as np
import pytest
import torch

from probaforms.models.interfaces import GenModel

from probaforms.models import ResidualUnconditional
from probaforms.models import ResidualConditional
from probaforms.models import ResidualFlowModel


def subclasses(cls):
    return set(cls.__subclasses__()).union(s for c in cls.__subclasses__() for s in subclasses(c))

print(subclasses(ResidualUnconditional))

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


@pytest.mark.parametrize("wrapper,model", [(ResidualUnconditional, ResidualFlowModel)])
def test_resflow_uncond(wrapper, model):
    flow_args_dict = {
        'var_dim': 2,
        'cond_dim': None,
        'hid_dim': 16,
        'n_block_layers': 3,
        'n_layers': 3,
        'spnorm_coeff': 0.95,
        'n_backward_iters': 100,
    }

    flow = model(**flow_args_dict)

    start_lr = 1e-1
    final_lr = 1e-2
    n_epochs = 100
    sched_lambda = (final_lr / start_lr) ** (1 / n_epochs)

    optim = torch.optim.Adam(flow.parameters(), lr=start_lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ExponentialLR(optim, sched_lambda)
    wrapper = wrapper(flow, optim, n_epochs=n_epochs, batch_size=100, scheduler=sched)
    X, y = gen_data(0, 500)
    data = torch.cat([X, y], dim=1)
    _ = wrapper.fit(data)
    _ = wrapper.sample(500, batched=100).cpu()


@pytest.mark.parametrize("wrapper,model", [(ResidualConditional, ResidualFlowModel)])
def test_resflow_cond(wrapper, model):
    flow_args_dict = {
        'var_dim': 1,
        'cond_dim': 1,
        'hid_dim': 16,
        'n_block_layers': 3,
        'n_layers': 3,
        'spnorm_coeff': 0.95,
        'n_backward_iters': 100,
    }

    flow = model(**flow_args_dict)

    start_lr = 1e-1
    final_lr = 1e-2
    n_epochs = 100
    sched_lambda = (final_lr / start_lr) ** (1 / n_epochs)

    optim = torch.optim.Adam(flow.parameters(), lr=start_lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ExponentialLR(optim, sched_lambda)
    wrapper = wrapper(flow, optim, n_epochs=n_epochs, batch_size=100, scheduler=sched)
    X, y = gen_data(0, 500)
    _ = wrapper.fit(y, X)
    _ = wrapper.sample(X, batched=None).cpu()
