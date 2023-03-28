import numpy as np
import pytest
from inspect import getmembers, isfunction
from probaforms import metrics

funcs = [f[1] for f in getmembers(metrics, isfunction)]


@pytest.mark.parametrize("metric", funcs)
def test_interface(metric):
    N = 100
    X_real = np.random.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], N)
    X_fake = np.random.multivariate_normal([0, 0.], [[1, 0.7], [0.7, 1]], N)
    mu, sigma = metric(X_real, X_fake)
