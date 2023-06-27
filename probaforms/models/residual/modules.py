import torch
from torch import nn
import numpy as np

from .gradients import memory_saved_logdet_wrapper, safe_detach
from .gradients import logdet_Jg_exact, logdet_Jg_cutoff, logdet_Jg_unbias


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Modified spectral normalization [Miyato et al. 2018] for invertible residual networks
    Most of this implementation is borrowed from the following link:
    https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    See paper, app. D, eq. (16), and slide 25 of:
    https://github.com/HSE-LAMBDA/DeepGenerativeModels/blob/spring-2021/lectures/9-NF2.pdf
    """

    def __init__(self, module, coeff=0.97, eps=1.0e-5, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.coeff = coeff
        self.eps = eps
        self.name = name
        self.power_iterations = power_iterations

        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        w = getattr(self.module, self.name + '_bar')

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        scale = self.coeff / (sigma + self.eps)

        delattr(self.module, self.name)
        if scale < 1.0:
            setattr(self.module, self.name, w * scale.expand_as(w))
        else:
            setattr(self.module, self.name, w)

    def _made_params(self):
        try:
            _ = getattr(self.module, self.name + '_u')
            _ = getattr(self.module, self.name + '_v')
            _ = getattr(self.module, self.name + '_bar')
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = w.data.new(height).normal_(0, 1)
        v = w.data.new(width).normal_(0, 1)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        self.module.register_buffer(self.name + '_u', u)
        self.module.register_buffer(self.name + '_v', v)
        self.module.register_parameter(self.name + '_bar', w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class ActNorm(nn.Module):
    def __init__(self, var_dim, eps=1e-5):
        super(ActNorm, self).__init__()
        self.var_dim = var_dim
        self.eps = eps

        self.register_parameter('log_scale', nn.Parameter(torch.zeros(self.var_dim)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(self.var_dim)))
        self.initialized = False

    def forward(self, z, log_df_dz):
        if not self.initialized:
            z_reshape = z.view(z.size(0), self.var_dim, -1)
            log_std = torch.log(torch.std(z_reshape, dim=[0, 2]) + self.eps)
            mean = torch.mean(z_reshape, dim=[0, 2])
            self.log_scale.data.copy_(log_std.view(self.var_dim))
            self.bias.data.copy_(mean.view(self.var_dim))
            self.initialized = True

        z = (z - self.bias) / torch.exp(self.log_scale)

        num_pixels = np.prod(z.size()) // (z.size(0) * z.size(1))
        log_df_dz -= torch.sum(self.log_scale) * num_pixels
        return z, log_df_dz

    def backward(self, y, log_df_dz):
        y = y * torch.exp(self.log_scale) + self.bias
        num_pixels = np.prod(y.size()) // (y.size(0) * y.size(1))
        log_df_dz += torch.sum(self.log_scale) * num_pixels
        return y, log_df_dz


class LipSwish(nn.Module):
    def __init__(self):
        super(LipSwish, self).__init__()
        beta = nn.Parameter(torch.ones([1], dtype=torch.float32))
        self.register_parameter('beta', beta)

    def forward(self, x, cond=None):
        return x * torch.sigmoid(self.beta * x) / 1.1


class InvertibleResBlockBase(nn.Module):
    """
    invertible residual block
    """
    def __init__(self, coeff=0.97, ftol=1e-4, logdet_estimator='unbias', n_backward_iters=100):
        super(InvertibleResBlockBase, self).__init__()

        self.coeff = coeff
        self.ftol = ftol
        self.estimator = logdet_estimator
        self.proc_g_fn = memory_saved_logdet_wrapper
        self.logdet_fn = self._get_logdet_estimator()
        self.n_iters = n_backward_iters

        self.g_fn = ...
        self.var_dim = ...

    def _get_logdet_estimator(self):
        if self.training:
            # force use unbiased log-det estimator
            logdet_fn = lambda g, z: logdet_Jg_unbias(g, z, 1, is_training=self.training)
        else:
            if self.estimator == 'exact':
                logdet_fn = logdet_Jg_exact
            elif self.estimator == 'fixed':
                logdet_fn = lambda g, z: logdet_Jg_cutoff(g, z, n_samples=5, n_power_series=10)
            elif self.estimator == 'unbias':
                logdet_fn = lambda g, z: logdet_Jg_unbias(
                    g, z, n_samples=5, n_exact=10, is_training=self.training)
            else:
                raise Exception('Unknown logdet estimator: %s' % self.estimator)

        return logdet_fn

    def forward(self, x, log_df_dz):
        # x = [y] or [y, cond]
        g, logdet = self.proc_g_fn(self.logdet_fn, x, self.g_fn, self.training)
        # residual z = F(x) = y + g(y, cond), g is a network
        z = x[:, :self.var_dim] + g
        log_df_dz += logdet
        return z, log_df_dz

    def backward(self, z, log_df_dz):
        x = safe_detach(z.clone())
        cond = x[:, self.var_dim:].clone()

        with torch.enable_grad():
            x.requires_grad_(True)
            cond.requires_grad_(True)
            for k in range(self.n_iters):
                x = safe_detach(x)
                prev_x_var = safe_detach(x[:, :self.var_dim])
                # fixed point iteration x = y - g(x)
                new_x_var = z[:, :self.var_dim] - self.g_fn(x)
                if torch.all(torch.abs(new_x_var - prev_x_var) < self.ftol):
                    break

                x = safe_detach(torch.cat([new_x_var, cond], dim=1).requires_grad_(True))

            del prev_x_var
            logdet = self.logdet_fn(self.g_fn(x), x)
        return new_x_var, log_df_dz - logdet

    # def backward(self, z, log_df_dz):
    #     new_x_var = z[:, :self.var_dim] - self.g_fn(z)
    #     return new_x_var, 0.0


class ResBackbone(nn.Module):
    def __init__(self, in_features,
                 out_features,
                 base_filters=32,
                 n_layers=2,
                 coeff=0.97):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        hidden_dims = [in_features] + [base_filters] * n_layers + [out_features]
        self.layers = nn.ModuleList()
        for i, (in_dims, out_dims) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            module = nn.Linear(in_dims, out_dims)
            self.layers.append(SpectralNorm(module, coeff=coeff))
            if i != len(hidden_dims) - 2:
                self.layers.append(LipSwish())

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x


class InvertibleResLinear(InvertibleResBlockBase):
    def __init__(self,
                 in_features,
                 out_features,
                 base_filters=32,
                 n_layers=2,
                 coeff=0.97,
                 ftol=1.0e-4,
                 logdet_estimator='unbias',
                 n_backward_iters=100):
        '''
        Pass concat [X, y] if conditioning, return only y
        See class BaseFlow in model.py
        '''
        super(InvertibleResLinear, self).__init__(coeff, ftol, logdet_estimator, n_backward_iters)
        self.g_fn = ResBackbone(in_features, out_features, base_filters, n_layers, coeff)
        self.var_dim = self.g_fn.out_features
