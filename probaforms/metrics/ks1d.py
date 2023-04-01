import numpy as np
from sklearn.utils import resample
from scipy.stats import ks_2samp, cramervonmises_2samp, anderson_ksamp
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning) # for _anderson1d



def _ks1d(data1, data2):
    ks, _ = ks_2samp(data1, data2)
    return ks

def _cvm1d(x, y):
    res = cramervonmises_2samp(x, y)
    return res.statistic

def _roc1d(x, y):
    labels = np.array([0]*len(x) + [1]*len(y))
    score = np.concatenate((x, y), axis=0)
    auc = roc_auc_score(labels, score)
    auc = np.abs(auc - 0.5) + 0.5
    return auc

def _anderson1d(x, y):
    res = anderson_ksamp([x, y])
    return res.statistic


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
    Calculates the Kolmogorov Smirnov statistic for real and fake samples.
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
        The estimated KS statistic.
    Std: fload
        The standard deviation of the distance.
    '''

    return _bootstrap_metric(_ks1d, X_real, X_fake, n_iters)


def cramer_von_mises_1d(X_real, X_fake, n_iters=100):
    '''
    Calculates the Cramer-von Mises statistics for real and fake samples.
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
        The estimated CvM statistic.
    Std: fload
        The standard deviation of the distance.
    '''

    return _bootstrap_metric(_cvm1d, X_real, X_fake, n_iters)


def roc_auc_score_1d(X_real, X_fake, n_iters=100):
    '''
    Calculates Area Under the Receiver Operating Characteristic Curve (ROC AUC) for real and fake samples.
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
        The estimated statistic.
    Std: fload
        The standard deviation of the distance.
    '''

    return _bootstrap_metric(_roc1d, X_real, X_fake, n_iters)


def anderson_darling_1d(X_real, X_fake, n_iters=100):
    '''
    Calculates the Anderson-Darling statistic for real and fake samples.
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
        The estimated statistic.
    Std: fload
        The standard deviation of the distance.
    '''

    return _bootstrap_metric(_anderson1d, X_real, X_fake, n_iters)
