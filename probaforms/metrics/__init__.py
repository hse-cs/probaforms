from .fd import frechet_distance
from .mmd import maximum_mean_discrepancy
from .ks1d import kolmogorov_smirnov_1d, cramer_von_mises_1d
from .ks1d import roc_auc_score_1d, anderson_darling_1d
from .div1d import kullback_leibler_1d, kullback_leibler_1d_kde
from .div1d import jensen_shannon_1d, jensen_shannon_1d_kde


__all__ = [
    'frechet_distance',
    'maximum_mean_discrepancy',
    'kolmogorov_smirnov_1d',
    'kullback_leibler_1d',
    'jensen_shannon_1d',
    'kullback_leibler_1d_kde',
    'jensen_shannon_1d_kde',
    'cramer_von_mises_1d',
    'roc_auc_score_1d',
    'anderson_darling_1d'
]
