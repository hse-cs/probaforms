import numpy as np
from .ks1d import _bootstrap_metric
from sklearn.neighbors import KernelDensity


def compute_probs1d(data, bins=10):
    h, e = np.histogram(data, bins)
    p = h/h.sum()
    return p, e


def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def get_probs1d(data1, data2, bins=10):

    data12 = np.concatenate((data1, data2), axis=0)
    _, e = compute_probs1d(data12, bins)

    p, _ = compute_probs1d(data1, e)
    q, _ = compute_probs1d(data2, e)

    return p, q


def _kl1d(data1, data2, bins=10):
    p, q = get_probs1d(data1, data2, bins)
    eps = 10**-5 / bins
    return kl_divergence(p+eps, q+eps)


def _js1d(data1, data2, bins=10):
    p, q = get_probs1d(data1, data2, bins)
    eps = 10**-5 / bins
    return js_divergence(p+eps, q+eps)


def compute_probs1d_kde(data, grid):
    kd = KernelDensity(bandwidth='silverman')
    kd.fit(data.reshape(-1, 1))
    p = np.exp(kd.score_samples(grid))
    p /= p.sum()
    return p


def get_probs1d_kde(data1, data2, bins=101):

    data12 = np.concatenate((data1, data2), axis=0)
    grid = np.linspace(data12.min(), data12.max(), bins).reshape(-1, 1)

    p = compute_probs1d_kde(data1, grid)
    q = compute_probs1d_kde(data2, grid)

    return p, q


def _kl1d_kde(data1, data2, bins=101):
    p, q = get_probs1d_kde(data1, data2, bins)
    eps = 10**-5 / bins
    return kl_divergence(p+eps, q+eps)


def _js1d_kde(data1, data2, bins=101):
    p, q = get_probs1d_kde(data1, data2, bins)
    eps = 10**-5 / bins
    return js_divergence(p+eps, q+eps)


def kullback_leibler_1d(X_real, X_fake, n_iters=100, bins=10):
    '''
    [Not recommended to use]
    Calculates the Kullback-Leibler divergence for real and fake samples.
    The function calculates metric values for each input feature,
    and then averaged them.

    Parameters:
    -----------
    X_real: numpy.ndarray of shape [n_samples, n_features]
        Real sample.
    X_fake: numpy.ndarray of shape [n_samples, n_features]
        Generated sample.
    n_iters: int
        The number of bootstrap iterations. Default = 100.
    bins: int
        The number of bins used to compute probability densities. Default = 10.

    Return:
    -------
    distance: float
        The estimated metric value.
    Std: fload
        The standard deviation of the value.
    '''

    return _bootstrap_metric(_kl1d, X_real, X_fake, n_iters, bins)


def jensen_shannon_1d(X_real, X_fake, n_iters=100, bins=10):
    '''
    [Not recommended to use]
    Calculates the Jensen-Shannon divergence for real and fake samples.
    The function calculates metric values for each input feature,
    and then averaged them.

    Parameters:
    -----------
    X_real: numpy.ndarray of shape [n_samples, n_features]
        Real sample.
    X_fake: numpy.ndarray of shape [n_samples, n_features]
        Generated sample.
    n_iters: int
        The number of bootstrap iterations. Default = 100.
    bins: int
        The number of bins used to compute probability densities. Default = 10.

    Return:
    -------
    distance: float
        The estimated metric value.
    Std: fload
        The standard deviation of the value.
    '''

    return _bootstrap_metric(_js1d, X_real, X_fake, n_iters, bins)



def kullback_leibler_1d_kde(X_real, X_fake, n_iters=100, bins=101):
    '''
    Calculates the Kullback-Leibler divergence for real and fake samples.
    The function calculates metric values for each input feature using 1D KDE,
    and then averaged them.

    Parameters:
    -----------
    X_real: numpy.ndarray of shape [n_samples, n_features]
        Real sample.
    X_fake: numpy.ndarray of shape [n_samples, n_features]
        Generated sample.
    n_iters: int
        The number of bootstrap iterations. Default = 100.
    bins: int
        The number of bins used to compute probability densities. Default = 10.

    Return:
    -------
    distance: float
        The estimated metric value.
    Std: fload
        The standard deviation of the value.
    '''

    return _bootstrap_metric(_kl1d_kde, X_real, X_fake, n_iters, bins)


def jensen_shannon_1d_kde(X_real, X_fake, n_iters=100, bins=101):
    '''
    Calculates the Jensen-Shannon divergence for real and fake samples.
    The function calculates metric values for each input feature using 1D KDE,
    and then averaged them.

    Parameters:
    -----------
    X_real: numpy.ndarray of shape [n_samples, n_features]
        Real sample.
    X_fake: numpy.ndarray of shape [n_samples, n_features]
        Generated sample.
    n_iters: int
        The number of bootstrap iterations. Default = 100.
    bins: int
        The number of bins used to compute probability densities. Default = 10.

    Return:
    -------
    distance: float
        The estimated metric value.
    Std: fload
        The standard deviation of the value.
    '''

    return _bootstrap_metric(_js1d_kde, X_real, X_fake, n_iters, bins)
