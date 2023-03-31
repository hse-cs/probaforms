import numpy as np
import pytest
from inspect import getmembers, isfunction
from probaforms import metrics

funcs = [f[1] for f in getmembers(metrics, isfunction)]


@pytest.mark.parametrize("metric", funcs)
def test_same_in_size_2d(metric):
    N = 100
    X_real = np.random.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], N)
    X_fake = np.random.multivariate_normal([0, 0.], [[1, 0.7], [0.7, 1]], N)
    mu, sigma = metric(X_real, X_fake)

@pytest.mark.parametrize("metric", funcs)
def test_same_in_size_1d(metric):
    N = 100
    X_real = np.random.normal(0, 1, N).reshape(-1, 1)
    X_fake = np.random.normal(1, 1, N).reshape(-1, 1)
    mu, sigma = metric(X_real, X_fake)

@pytest.mark.parametrize("metric", funcs)
def test_different_in_size_2d(metric):
    X_real = np.random.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], 100)
    X_fake = np.random.multivariate_normal([0, 0.], [[1, 0.7], [0.7, 1]], 153)
    mu, sigma = metric(X_real, X_fake)

    X_real = np.random.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], 342)
    X_fake = np.random.multivariate_normal([0, 0.], [[1, 0.7], [0.7, 1]], 100)
    mu, sigma = metric(X_real, X_fake)
