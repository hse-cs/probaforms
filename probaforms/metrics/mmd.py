import numpy as np
from sklearn import metrics
import joblib
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler


def mmd_calc(X, Y):

    agg_matrix = np.concatenate((X, Y), axis=0)
    distances = metrics.pairwise_distances(agg_matrix)
    median_distance = np.median(distances)
    gamma = 1.0 / (2 * median_distance**2)

    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)

    mmd = XX.mean() + YY.mean() - 2 * XY.mean()

    return mmd


def maximum_mean_discrepancy(X, Y, n_iters=100, standardize=False):
    '''
    [Computationally expensive. Recommended for samples size < 5000]
    Calculates the Maximum Mean Discrepancy between real and fake samples.

    Parameters:
    -----------
    X: numpy.ndarray of shape [n_samples, n_features]
        Real sample.
    Y: numpy.ndarray of shape [n_samples, n_features]
        Generated sample.
    n_iters: int
        The number of bootstrap iterations. Default = 100.
    standardize: boolean
        If True, the StandardScaler is fitted on the real and
        applied to the real and fake samples. Default = False.

    Return:
    -------
    distance: float
        The estimated Frechet Distance.
    Std: fload
        The standard deviation of the distance.
    '''

    if standardize:
        scaler =  StandardScaler()
        X = scaler.fit_transform(X)
        Y = scaler.transform(Y)

    mmds = []

    for i in range(n_iters):

        X_boot = resample(X)
        Y_boot = resample(Y)

        mmd_boot = mmd_calc(X_boot, Y_boot)
        mmds.append(mmd_boot)

    return np.mean(mmds), np.std(mmds)
