import numpy as np
import pyoti.sparse as oti


def adjust_sigma_n_bound(base_bounds, true_noise_std=None, cushion=.5):
    """
    Adjust the log10 bounds of sigma_n based on known noise in the data.
    """
    lower, upper = base_bounds
    if true_noise_std is not None:
        log_sigma = np.log10(true_noise_std)
        return (log_sigma - cushion, log_sigma + cushion)
    return base_bounds


class KernelFactory:
    def __init__(self, dim, normalize, differences_by_dim, true_noise_std=None):
        self.dim = dim
        self.normalize = normalize
        self.differences_by_dim = differences_by_dim
        self.true_noise_std = true_noise_std  # Optional true noise info
        self.bounds = []

    def get_bounds_from_data(self):
        self.bounds = []
        for diffs in self.differences_by_dim:
            min_val = float(diffs.real.min())
            max_val = float(diffs.real.max())
            self.bounds.append((min_val, max_val))

    def create_kernel(self, kernel_name, kernel_type):
        if not self.normalize:
            self.get_bounds_from_data()

        if kernel_type == "anisotropic":
            return self._create_anisotropic(kernel_name)
        elif kernel_type == "isotropic":
            return self._create_isotropic(kernel_name)
        else:
            raise ValueError("Invalid kernel_type")

    def _create_anisotropic(self, kernel):
        base_sigma_n_bound = (-16, -3)
        sigma_n_bound = adjust_sigma_n_bound(
            base_sigma_n_bound, self.true_noise_std)

        if kernel == "SE":
            self._add_bounds([(-1, 3), sigma_n_bound])
            return self.se_kernel_anisotropic
        elif kernel == "RQ":
            self._add_bounds([(0, 10), (-9, 6), sigma_n_bound])
            return self.rq_kernel_anisotropic
        elif kernel == "SineExp":
            self._add_bounds([(0.0, 1e2)] * self.dim +
                             [(-9, 5), sigma_n_bound])
            return self.sine_exp_kernel_anisotropic
        else:
            raise NotImplementedError("Anisotropic kernel not implemented")

    def _create_isotropic(self, kernel):
        base_sigma_n_bound = (-16, -3)
        sigma_n_bound = adjust_sigma_n_bound(
            base_sigma_n_bound, self.true_noise_std)

        if self.normalize:
            core_bounds = [(-3, 3)]
        else:
            self.get_bounds_from_data()
            core_bounds = [(
                float(min([d.min() for d in self.differences_by_dim.real])),
                float(max([d.max() for d in self.differences_by_dim.real]))
            )]

        if kernel == "SE":
            self.bounds = core_bounds + [(-1, 3), sigma_n_bound]
            return self.se_kernel_isotropic
        elif kernel == "RQ":
            self.bounds = core_bounds + [(0, 10), (-9, 6), sigma_n_bound]
            return self.rq_kernel_isotropic
        elif kernel == "SineExp":
            self.bounds = core_bounds + [(0.0, 1e2), (-9, 4), sigma_n_bound]
            return self.sine_exp_kernel_isotropic
        else:
            raise NotImplementedError("Isotropic kernel not implemented")

    def _add_bounds(self, extra_bounds):
        if self.normalize:
            self.bounds = [(-3, 3)] * self.dim + extra_bounds
        else:
            self.bounds += extra_bounds

    # KERNEL FUNCTIONS BELOW --------------------------

    def se_kernel_anisotropic(self, differences_by_dim, length_scales, index=-1):
        ell = 10 ** (length_scales[:-1])
        sigma_f = length_scales[-1]
        sqdist = sum((ell[i] * differences_by_dim[i])
                     ** 2 for i in range(self.dim))
        return (10 ** sigma_f) ** 2 * oti.exp(-0.5 * sqdist)

    def rq_kernel_anisotropic(self, differences_by_dim, length_scales, index=-1):
        ell = 10 ** (length_scales[:self.dim])
        alpha = np.exp(length_scales[self.dim])
        sigma_f = length_scales[-1]
        sqdist = 1 + sum((ell[i] * differences_by_dim[i])
                         ** 2 for i in range(self.dim))
        return (10 ** sigma_f) ** 2 * (1 + sqdist / (2 * alpha)) ** (-alpha)

    def sine_exp_kernel_anisotropic(self, differences_by_dim, length_scales, index=-1):
        ell = 10 ** (length_scales[:self.dim])
        p = length_scales[self.dim:-1]
        sigma_f = length_scales[-1]
        sqdist = 1 + sum((ell[i] * oti.sin((np.pi / p[i]) * differences_by_dim[i])) ** 2
                         for i in range(self.dim))
        return (10 ** sigma_f) ** 2 * oti.exp(-2 * sqdist)

    def se_kernel_isotropic(self, differences_by_dim, length_scales, index=-1):
        ell = 10 ** length_scales[0]
        sigma_f = length_scales[-1]
        sqdist = sum(
            (ell * differences_by_dim[i]) ** 2 for i in range(self.dim))
        return (10 ** sigma_f) ** 2 * oti.exp(-0.5 * sqdist)

    def rq_kernel_isotropic(self, differences_by_dim, length_scales, index=-1):
        ell = 10 ** length_scales[0]
        alpha = np.exp(length_scales[1])
        sigma_f = length_scales[-1]
        sqdist = 1 + \
            sum((ell * differences_by_dim[i]) ** 2 for i in range(self.dim))
        return (10 ** sigma_f) ** 2 * (1 + sqdist / (2 * alpha)) ** (-alpha)

    def sine_exp_kernel_isotropic(self, differences_by_dim, length_scales, index=-1):
        ell = 10 ** length_scales[0]
        p = length_scales[1]
        sigma_f = length_scales[-1]
        sqdist = sum((ell ** 2 * oti.sin(np.pi * differences_by_dim[i] / p)) ** 2
                     for i in range(self.dim))
        return (10 ** sigma_f) ** 2 * oti.exp(-2 * sqdist)
