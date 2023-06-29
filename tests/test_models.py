import numpy as np
import pytest
import torch
import itertools

from probaforms.models.interfaces import GenModel

from probaforms.models import ResidualFlowModel
from probaforms.models import ResidualUnconditional
from probaforms.models import ResidualConditional

from probaforms.models import PlainBackboneResidual
from probaforms.models import DDPMUnconditional
from probaforms.models import DDPMConditional
from probaforms.models import DiffusionMLP


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


logdets = ['exact', 'fixed', 'unbias']
res_wrappers = [ResidualUnconditional, ResidualConditional]
@pytest.mark.parametrize("wrapper_model,logdet", list(itertools.product(res_wrappers, logdets)))
def test_resflow(wrapper_model, logdet):
    is_cond = (wrapper_model == ResidualConditional)

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

    flow = ResidualFlowModel(**flow_args_dict)

    start_lr = 1e-1
    final_lr = 1e-2
    n_epochs = 100
    sched_lambda = (final_lr / start_lr) ** (1 / n_epochs)

    optim = torch.optim.Adam(flow.parameters(), lr=start_lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ExponentialLR(optim, sched_lambda)
    wrapper = wrapper_model(flow, optim, n_epochs=n_epochs, batch_size=100, scheduler=sched)

    if is_cond:
        _ = wrapper.fit(y, X)
        _ = wrapper.sample(X, batched=None).cpu()
    else:
        data = torch.cat([X, y], dim=1)
        _ = wrapper.fit(data)
        _ = wrapper.sample(500, batched=100).cpu()


@pytest.mark.parametrize("wrapper_model", [(DDPMUnconditional), (DDPMConditional)])
def test_ddpm(wrapper_model):
    is_cond = (wrapper_model == DDPMConditional)

    n = 100
    X = torch.from_numpy(np.random.normal(size=(n, 5))).to(torch.float32)
    y = torch.from_numpy(np.random.normal(size=(n, 2))).to(torch.float32)
    len_y = y.shape[1]; len_X = X.shape[1]

    backbone_args_dict = {
        'var_dim': len_y if is_cond else len_y + len_X,
        'cond_dim': 4 * len_X if is_cond else None,
        'hid_dim': 128,
        'num_blocks': 6,
        'n_steps': 100,
        'steps_dim': 128,
        'steps_depth': 5,
        'steps_out_dim': 8,
    }

    backbone = PlainBackboneResidual(**backbone_args_dict)

    beta_grid = 'linear'
    betas = (1e-3, 1e-2)
    sigma_method = 'beta_wave'

    dmlp = DiffusionMLP(backbone, betas, beta_grid, sigma_method)

    start_lr = 1e-1
    final_lr = 1e-2
    n_epochs = 100
    sched_lambda = (final_lr / start_lr) ** (1 / n_epochs)

    optim = torch.optim.Adam(dmlp.parameters(), lr=start_lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ExponentialLR(optim, sched_lambda)
    wrapper = wrapper_model(dmlp, optim, n_epochs=n_epochs, batch_size=100, scheduler=sched)

    if is_cond:
        X, y = gen_data(0, 500)
        _ = wrapper.fit(y, X)
        _ = wrapper.sample(X, batched=100).cpu()
    else:
        data = torch.cat([X, y], dim=1)
        _ = wrapper.fit(data)
        _ = wrapper.sample(500, batched=100).cpu()
