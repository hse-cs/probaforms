import torch
from torch import randn as rand_normal
import numpy as np


# ==================== logdet estimators for residual flow ========================

def logdet_Jg_exact(g, x):
    '''
    Exact logdet determinant computation (naive forehead approach)

    :param g: outputs g(x)
    :param x: inputs to g function (optimized network)
    :return: log(I + Jg(x)), where Jg(x) is the Jacobian defined as dg(x) / dx
    '''

    var_dim = g.shape[1]

    Jg = [
        torch.autograd.grad(g[:, i].sum(), x, create_graph=True, retain_graph=True)[0]
        for i in range(x.size(1))
    ]

    Jg = torch.stack(Jg, dim=1)
    ident = torch.eye(x.size(1)).type_as(x).to(x.device)
    return torch.logdet(ident + Jg)


def logdet_Jg_cutoff(g, x, n_samples=1, n_power_series=8):
    '''
    Biased logdet estimator with FIXED (!) number of trace's series terms, see paper, eq. (7)
    Skilling-Hutchinson trace estimator is used to estimate the trace of Jacobian matrices


    Unfortunately, this estimator requires each term to be stored in memory because ∂/∂θ needs to be
    applied to each term. The total memory cost is then O(n · m) where n is the number of computed
    terms and m is the number of residual blocks in the entire network. This is extremely memory-hungry
    during training, and a large random sample of n can occasionally result in running out of memory

    :param g: outputs g(x)
    :param x: inputs to g function (optimized network)
    :param n_samples: number of v samples
    :param n_power_series: fixed number of computed terms, param n in paper
    :return: log determinant approximation using FIXED (!) length cutoff for infinite series
            which can be used with residual block f(x) = x + g(x)
    '''

    var_dim = g.shape[1]

    # sample v ~ N(0, 1)
    v = rand_normal([g.size(0), n_samples, g.size(1)])
    v = v.type_as(x).to(x.device)

    # v^T Jg -- vector-Jacobian product
    w_t_J_fn = lambda w, x=x, g=g: torch.autograd.grad(
        g, x, grad_outputs=w, retain_graph=True, create_graph=True)[0]

    sum_diag = 0.0
    w = v.clone()
    for k in range(1, n_power_series + 1):
        w = [w_t_J_fn(w[:, i, :]) for i in range(n_samples)]
        w = torch.stack(w, dim=1)

        # v^T Jg^k v term
        inner = torch.einsum('bnd,bnd->bn', w, v)
        sum_diag += (-1) ** (k + 1) * (inner / k)

    # mathematical expectation
    return sum_diag.sum(dim=1) / n_samples


def logdet_Jg_unbias(g, x, n_samples=1, p=0.5, n_exact=1, is_training=True):
    '''
    Unbiased logdet estimator with UNFIXED (!) number of trace's series terms, see paper, eq. (6), also see eq. (8)
    Number of terms is sampled by geometric distribution
    Skilling-Hutchinson trace estimator is used to estimate the trace of Jacobian matrices

    As the power series in (8) does not need to be differentiated through, using this reduces the memory
    requirement by a factor of n. This is especially useful when using the unbiased estimator as the
    memory will be constant regardless of the number of terms we draw from p(N)

    :param g: outputs g(x)
    :param x: inputs to g function (optimized network)
    :param n_samples: number of v samples
    :param p: geometric distribution parameter
    :param n_exact: number of terms to be exactly computed
    :param is_training: True if training phase else False
    :return: log determinant approximation using unbiased series length sampling (UNFIXED LEN)
            which can be used with residual block f(x) = x + g(x)
    '''

    '''
    In conditional case inputs x = [y, cond] of shape (var_dim + cond_dim)
    Outputs g(x) shape is always (var_dim)
    '''

    var_dim = g.shape[1]

    def geom_cdf(k):
        # P[N >= k] = 1 - f_geom(k), Geom(p) probability
        return (1.0 - p) ** max(0, k - n_exact)

    res = 0.0
    for j in range(n_samples):
        n_power_series = n_exact + np.random.geometric(p)
        v = torch.randn_like(g)  # N(0, 1) by paper
        w = v

        sum_vj = 0.0
        for k in range(1, n_power_series + 1):
            # v^T Jg -- vector-Jacobian product
            w = torch.autograd.grad(g, x, w, create_graph=is_training, retain_graph=True)[0]
            w = w[:, :var_dim].reshape(w.shape[0], -1)  # x = [y, cond], derivatives only w.r.t. y
            P_N_ge_k = geom_cdf(k - 1)  # P[N >= k]
            tr = torch.sum(w * v, dim=1)  # v^T Jg v
            sum_vj = sum_vj + (-1) ** (k + 1) * (tr / (k * P_N_ge_k))
        res += sum_vj
    return res / n_samples


def logdet_Jg_neumann(g, x, n_samples=1, p=0.5, n_exact=1):
    '''
    Unbiased Neumann logdet estimator see paper with russian roulette applied, see paper, eq. (8) and app. C
    Provides Neumann gradient series with russian roulette and trace estimator applied to obtain the theorem (8)

    :param g: outputs g(x)
    :param x: inputs to g function (optimized network)
    :param n_samples: number of v samples
    :param p: geometric distribution parameter
    :param n_exact: number of terms to be exactly computed
    :return: log determinant approximation using unbiased series length sampling
     ---
    NOTE: this method using neumann series does not return exact "log_df_dz"
    but the one that can be only used in gradient wrt parameters
    see: https://github.com/rtqichen/residual-flows/blob/f9dd4cd0592d1aa897f418e25cae169e77e4d692/lib/layers/iresblock.py#L249
    and: https://github.com/tatsy/normalizing-flows-pytorch/blob/f5238fa8ce62a130679a1cf4474e195926b4842f/flows/iresblock.py#L84
    '''

    '''
    In conditional case inputs x = [y, cond] of shape (var_dim + cond_dim)
    Outputs g(x) shape is always (var_dim)
    '''

    var_dim = g.shape[1]

    def geom_cdf(k):
        # P[N >= k] = 1 - f_geom(k), Geom(p) probability
        return (1.0 - p) ** max(0, k - n_exact)

    res = 0.0
    for j in range(n_samples):
        n_power_series = n_exact + np.random.geometric(p)

        v = torch.randn_like(g)
        w = v

        sum_vj = v
        with torch.no_grad():
            # v^T Jg sum
            for k in range(1, n_power_series + 1):
                # v^T Jg -- vector-Jacobian product
                w = torch.autograd.grad(g, x, w, retain_graph=True)[0]
                w = w[:, :var_dim].view(w.shape[0], -1)  # x = [y, cond], derivatives only w.r.t. y
                P_N_ge_k = geom_cdf(k - 1)  # P[N >= k]
                sum_vj = sum_vj + ((-1) ** k / P_N_ge_k) * w

        # Jg v
        sum_vj = torch.autograd.grad(g, x, sum_vj, create_graph=True)[0]
        sum_vj = sum_vj[:, :var_dim].view(sum_vj.shape[0], -1)  # аналогично
        res += torch.sum(sum_vj * v, dim=1)
    return res / n_samples


class MemorySavedLogDetEstimator(torch.autograd.Function):
    """
    Memory saving logdet estimator, see paper, 3.2 and app. C
    Provides custom memory-saving backprop
    """

    @staticmethod
    def forward(ctx, logdet_fn, x, net_g_fn, training, *g_params):
        '''
        Args:
            ctx: context object (see https://pytorch.org/docs/stable/autograd.html#function)
            logdet_fn: logdet estimator function for loss calculation
            x: inputs to g(x)
            net_g_fn: optimized function (network)
            training: True if training phase, else False
            *g_params: parameters of g

        Returns:
            g(x): outputs g for inputs x
            logdet: estimated logdet
        '''

        ctx.training = training
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            g = net_g_fn(x)
            ctx.x = x  # shape (var_dim + cond_dim) if cond else (var_dim)
            ctx.g = g  # shape (var_dim) in any case

            # Backward-in-forward: early computation of gradient
            # Pass params x and theta, return grads w.r.t. x and theta
            # https://pytorch.org/docs/stable/generated/torch.autograd.grad.html
            theta = list(g_params)
            if ctx.training:
                # logdet for neumann series
                logdetJg = logdet_Jg_neumann(g, x).sum()
                dlogdetJg_dx, *dlogdetJg_dtheta = torch.autograd.grad(logdetJg, [x] + theta,
                                                                      retain_graph=True,
                                                                      allow_unused=True)
                ctx.save_for_backward(dlogdetJg_dx, *theta, *dlogdetJg_dtheta)

            # logdet for loss calculation
            logdet = logdet_fn(g, x)
        return safe_detach(g), safe_detach(logdet)

    @staticmethod
    def backward(ctx, dL_dg, dL_dlogdetJg):
        """
        NOTE: Be careful that chain rule for partial differentiation is as follows
        df(y, z)    df   dy     df   dz
        -------- =  -- * --  +  -- * --
        dx          dy   dx     dz   dx
        """

        training = ctx.training
        if not training:
            raise ValueError('Provide training=True if using backward.')

        # chain rule for partial differentiation (1st term)
        with torch.enable_grad():
            g, x = ctx.g, ctx.x
            dlogdetJg_dx, *saved_tensors = ctx.saved_tensors
            n_params = len(saved_tensors) // 2
            theta = saved_tensors[:n_params]
            dlogdetJg_dtheta = saved_tensors[n_params:] # 2nd multiplier of (9)

            dL_dx_1st, *dL_dtheta_1st = torch.autograd.grad(g, [x] + theta,
                                                            grad_outputs=dL_dg,
                                                            allow_unused=True)

        # chain rule for partial differentiation (2nd term)
        # NOTE: dL_dlogdetJg consists of same values for all dimensions (see forward).
        dL_dlogdetJg_scalar = dL_dlogdetJg[0].detach()  # 1st multiplier of (9)
        with torch.no_grad():
            dL_dx_2nd = dlogdetJg_dx * dL_dlogdetJg_scalar  # see paper eq. (9)
            dL_dtheta_2nd = tuple(
                [g * dL_dlogdetJg_scalar if g is not None else None for g in dlogdetJg_dtheta])

        with torch.no_grad():
            dL_dx = dL_dx_1st + dL_dx_2nd
            dL_dtheta = tuple([
                g1 + g2 if g2 is not None else g1 for g1, g2 in zip(dL_dtheta_1st, dL_dtheta_2nd)
            ])

        return (None, dL_dx, None, None) + dL_dtheta


def memory_saved_logdet_wrapper(logdet_fn, x, net_g_fn, training):
    # x = [y] or [y, cond]
    g_params = list(net_g_fn.parameters())
    return MemorySavedLogDetEstimator.apply(logdet_fn, x, net_g_fn, training, *g_params)


def safe_detach(x):
    """
    detach operation which keeps reguires_grad
    """
    return x.detach().requires_grad_(x.requires_grad)
