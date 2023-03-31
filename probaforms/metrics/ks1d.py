import numpy as np
from sklearn.utils import resample
from scipy.stats import ks_2samp


def _ks1d(data1, data2):
    ks, _ = ks_2samp(data1, data2)
    return ks


def _bootstrap_metric(metric_func, X_real, X_fake, n_iters=100, *args):
    '''
    Calculates a bottstraped metric for real and fake samples.

    Parameters:
    -----------
    metric_func: funciton
        The metric function, that takes two 1D arrays as inputs.
    X_real: numpy.ndarray of shape [n_samples, n_features]
        Real sample.
    X_fake: numpy.ndarray of shape [n_samples, n_features]
        Generated sample.
    n_iters: int
        The number of bootstrap iterations. Default = 100.

    Return:
    -------
    distance: float
        The estimated metric value.
    Std: fload
        The standard deviation of the value.
    '''

    scores = []

    for i in range(n_iters):

        X_real_boot = resample(X_real)
        X_fake_boot = resample(X_fake)

        score_boot = 0
        n_dim = X_real.shape[1]
        for d in range(n_dim):
            mval = metric_func(X_real_boot[:, d], X_fake_boot[:, d], *args)
            score_boot += mval / n_dim

        scores.append(score_boot)

    scores = np.array(scores)
    return scores.mean(axis=0), scores.std(axis=0)


def kolmogorov_smirnov_1d(X_real, X_fake, n_iters=100):
    '''
    Calculates the Kolmogorov Smirnov statistics for real and fake samples.
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

    Return:
    -------
    distance: float
        The estimated KS statistics.
    Std: fload
        The standard deviation of the distance.
    '''

    return _bootstrap_metric(_ks1d, X_real, X_fake, n_iters)
