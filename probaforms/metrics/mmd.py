import numpy as np
from sklearn import metrics
import joblib
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

def maximum_mean_discrepancy(X, Y, n_iters=100, standardize=True):
    '''
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

        agg_matrix = np.concatenate((X_boot, Y_boot), axis=0)
        distances = metrics.pairwise_distances(agg_matrix)
        median_distance = np.median(distances)
        gamma = 1.0 / (2 * median_distance**2)

        XX = metrics.pairwise.rbf_kernel(X_boot, X_boot, gamma)
        YY = metrics.pairwise.rbf_kernel(Y_boot, Y_boot, gamma)
        XY = metrics.pairwise.rbf_kernel(X_boot, Y_boot, gamma)

        mmd_boot = XX.mean() + YY.mean() - 2 * XY.mean()
        mmds.append(mmd_boot)

    return np.mean(mmds), np.std(mmds)
