import numpy as np
from sklearn.utils import resample
from scipy.stats import ks_2samp


def kolmogorov_smirnov_1d(X_real, X_fake, n_iters=100):
    '''
    Calculates the Kolmogorov Smirnov statistics for real and fake samples.

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

    scores = []

    inds = np.arange(len(X_real))

    for i in range(n_iters):
        inds_boot = resample(inds)

        X_real_boot = X_real[inds_boot]
        X_fake_boot = X_fake[inds_boot]

        score_boot = 0
        n_dim = X_real.shape[1]
        for d in range(n_dim):
            ks, _ = ks_2samp(X_real[:, d], X_fake[:, d])
            score_boot += ks / n_dim

        scores.append(score_boot)

    scores = np.array(scores)
    return scores.mean(axis=0), scores.std(axis=0)
