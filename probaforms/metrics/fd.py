import numpy as np
from scipy.linalg import sqrtm
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


def frechet_distance(X_real, X_fake, n_iters=100, standardize=False):
    '''
    Calculates the Frechet Distance between real and fake samples.

    Parameters:
    -----------
    X_real: numpy.ndarray of shape [n_samples, n_features]
        Real sample.
    X_fake: numpy.ndarray of shape [n_samples, n_features]
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

    frd = []

    if standardize:
        X_scaler = StandardScaler()
        X_real = X_scaler.fit_transform(X_real)
        X_fake = X_scaler.transform(X_fake)

    for i in range(n_iters):

        X_real_boot = resample(X_real)
        X_fake_boot = resample(X_fake)

        X_real_mean, X_real_cov = X_real_boot.mean(axis=0), np.cov(X_real_boot, rowvar=False)
        X_fake_mean, X_fake_cov = X_fake_boot.mean(axis=0), np.cov(X_fake_boot, rowvar=False)

        if X_real_boot.shape[1] == 1:
            X_real_cov = np.array([[X_real_cov]])
        if X_fake_boot.shape[1] == 1:
            X_fake_cov = np.array([[X_fake_cov]])

        diff = np.sum((X_real_mean - X_fake_mean)**2.0)
        covmean, _ = sqrtm(X_real_cov.dot(X_fake_cov), disp=False)

        if np.iscomplexobj(covmean): covmean = covmean.real
        tr_covmean = np.trace(covmean)

        ifrd = diff + np.trace(X_real_cov) + np.trace(X_fake_cov) - 2 * tr_covmean
        frd.append(ifrd)

    frd = np.array(frd)
    return frd.mean(axis=0), frd.std(axis=0)
