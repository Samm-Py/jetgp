import numpy as np
import pyoti.sparse as oti
from math import factorial
import sympy as sp
# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------


def matern_kernel_builder(nu):

    # Define symbols
    r = sp.symbols('r')
    nu = sp.Rational(2*nu, 2)  # ν = 1/2

    # Define prefactor
    prefactor = (2**(1 - nu)) / sp.gamma(nu)

    # Argument of Bessel function
    z = sp.sqrt(2 * nu) * r

    # Bessel function K_nu(z)

    # Full expression
    k_r = prefactor * z**nu * sp.simplify(sp.besselk(nu, z))

    # Simplify
    k_r_simplified = sp.simplify(k_r)

    expr = k_r_simplified

    # Step 1: Custom dictionary for lambdify
    custom_dict = {"exp": oti.exp}

    # Step 2: Lambdify the expression
    matern_kernel_func = sp.lambdify(
        (r), expr, modules=[custom_dict, "numpy"])

    return matern_kernel_func


# -------------------------------------------------------------------
# Kernel Factory Class
# -------------------------------------------------------------------


class KernelFactory:
    """
    Factory for generating different kernel functions (SE, RQ, SineExp, Matérn)
    in isotropic and anisotropic forms.

    Attributes
    ----------
    dim : int
        Dimensionality of the input space.
    normalize : bool
        Whether to normalize inputs (scaling differences to [-3, 3]).
    differences_by_dim : list of arrays
        Pairwise differences between input points, by dimension.
    true_noise_std : float, optional
        Known noise standard deviation (for adjusting noise bounds).
    bounds : list of tuples
        Hyperparameter bounds (log10 space).
    nu : float
        Smoothness parameter for the Matérn kernel.
    n_order : int
        Order of derivatives for kernel smoothness.
    """

    def __init__(self, dim, normalize, differences_by_dim, n_order, true_noise_std=None):
        self.dim = dim
        self.normalize = normalize
        self.differences_by_dim = differences_by_dim
        self.true_noise_std = true_noise_std
        self.bounds = []
        self.nu = 2 * n_order + 0.5  # Ensures ν is a half-integer
        self.n_order = n_order

    def get_bounds_from_data(self):
        """
        Computes bounds for hyperparameters based on the observed data range.
        """
        self.bounds = []
        for diffs in self.differences_by_dim:
            min_val = float(diffs.real.min())
            max_val = float(diffs.real.max())
            self.bounds.append((min_val, max_val))

    def create_kernel(self, kernel_name, kernel_type):
        """
        Returns a kernel function based on specified name and type.

        Parameters
        ----------
        kernel_name : str
            Name of the kernel ('SE', 'RQ', 'SineExp', 'Matern').
        kernel_type : str
            Type of kernel ('anisotropic' or 'isotropic').

        Returns
        -------
        callable
            The selected kernel function.
        """
        if not self.normalize:
            self.get_bounds_from_data()

        if kernel_type == "anisotropic":
            return self._create_anisotropic(kernel_name)
        elif kernel_type == "isotropic":
            return self._create_isotropic(kernel_name)
        else:
            raise ValueError("Invalid kernel_type")

    def _create_anisotropic(self, kernel):
        """
        Sets bounds and returns the anisotropic kernel function.
        """
        sigma_n_bound = (-16, -3)

        if kernel == "SE":
            self._add_bounds([(-1, 3), sigma_n_bound])
            return self.se_kernel_anisotropic
        elif kernel == "RQ":
            self._add_bounds([(0, 10), (-9, 6), sigma_n_bound])
            return self.rq_kernel_anisotropic
        elif kernel == "SineExp":
            self._add_bounds([(0.0, 1e2)] * self.dim +
                             [(-9, 5), sigma_n_bound])
        elif kernel == "Matern":
            self._add_bounds([(-1, 3), sigma_n_bound])
            self.matern_kernel_prebuild = matern_kernel_builder(self.nu)
            return self.matern_kernel_anisotropic
        else:
            raise NotImplementedError("Anisotropic kernel not implemented")

    def _create_isotropic(self, kernel):
        """
        Sets bounds and returns the isotropic kernel function.
        """
        sigma_n_bound = (-16, -3)

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
        elif kernel == "Matern":

            self.bounds = core_bounds + [(-1, 3), sigma_n_bound]
            return self.matern_kernel_isotropic
        else:
            raise NotImplementedError("Isotropic kernel not implemented")

    def _add_bounds(self, extra_bounds):
        """
        Adds hyperparameter bounds, adjusting for normalization.
        """
        if self.normalize:
            self.bounds = [(-3, 3)] * self.dim + extra_bounds
        else:
            self.bounds += extra_bounds

    # -------------------------------------------------------------------
    # Anisotropic Kernel Implementations
    # -------------------------------------------------------------------

    def se_kernel_anisotropic(self, differences_by_dim, length_scales, index=-1):
        """
        Anisotropic Squared Exponential (SE) kernel.
        """
        ell = 10 ** (length_scales[:-1])
        sigma_f = length_scales[-1]
        sqdist = sum((ell[i] * differences_by_dim[i])
                     ** 2 for i in range(self.dim))
        return (10 ** sigma_f) ** 2 * oti.exp(-0.5 * sqdist)

    def rq_kernel_anisotropic(self, differences_by_dim, length_scales, index=-1):
        """
        Anisotropic Rational Quadratic (RQ) kernel.
        """
        ell = 10 ** (length_scales[:self.dim])
        alpha = np.exp(length_scales[self.dim])
        sigma_f = length_scales[-1]
        sqdist = 1 + sum((ell[i] * differences_by_dim[i])
                         ** 2 for i in range(self.dim))
        return (10 ** sigma_f) ** 2 * (1 + sqdist / (2 * alpha)) ** (-alpha)

    def sine_exp_kernel_anisotropic(self, differences_by_dim, length_scales, index=-1):
        """
        Anisotropic Sine-Exponential kernel.
        """
        ell = 10 ** (length_scales[:self.dim])
        p = length_scales[self.dim:-1]
        sigma_f = length_scales[-1]
        sqdist = 1 + sum((ell[i] * oti.sin((np.pi / p[i]) *
                         differences_by_dim[i])) ** 2 for i in range(self.dim))
        return (10 ** sigma_f) ** 2 * oti.exp(-2 * sqdist)

    def matern_kernel_anisotropic(self, differences_by_dim, length_scales, index=-1):
        """
        Anisotropic Matérn kernel (half-integer ν).
        """
        ell = 10 ** (length_scales[:-1])
        sigma_f = length_scales[-1]
        sqdist = oti.sqrt(
            sum((ell[i] * (differences_by_dim[i] + 1e-6)) ** 2 for i in range(self.dim)))

        return (10 ** sigma_f) ** 2 * self.matern_kernel_prebuild(sqdist)

    # -------------------------------------------------------------------
    # Isotropic Kernel Implementations
    # -------------------------------------------------------------------

    def se_kernel_isotropic(self, differences_by_dim, length_scales, index=-1):
        """
        Isotropic Squared Exponential (SE) kernel.
        """
        ell = 10 ** length_scales[0]
        sigma_f = length_scales[-1]
        sqdist = sum(
            (ell * differences_by_dim[i]) ** 2 for i in range(self.dim))
        return (10 ** sigma_f) ** 2 * oti.exp(-0.5 * sqdist)

    def rq_kernel_isotropic(self, differences_by_dim, length_scales, index=-1):
        """
        Isotropic Rational Quadratic (RQ) kernel.
        """
        ell = 10 ** length_scales[0]
        alpha = np.exp(length_scales[1])
        sigma_f = length_scales[-1]
        sqdist = 1 + \
            sum((ell * differences_by_dim[i]) ** 2 for i in range(self.dim))
        return (10 ** sigma_f) ** 2 * (1 + sqdist / (2 * alpha)) ** (-alpha)

    def sine_exp_kernel_isotropic(self, differences_by_dim, length_scales, index=-1):
        """
        Isotropic Sine-Exponential kernel.
        """
        ell = 10 ** length_scales[0]
        p = length_scales[1]
        sigma_f = length_scales[-1]
        sqdist = sum((ell ** 2 * oti.sin(np.pi *
                     differences_by_dim[i] / p)) ** 2 for i in range(self.dim))
        return (10 ** sigma_f) ** 2 * oti.exp(-2 * sqdist)

    def metern_kernel_isotropic(self, differences_by_dim, length_scales, index=-1):
        """
        Isotropic Matérn kernel (half-integer ν).
        """
        ell = 10 ** length_scales[0]
        sigma_f = length_scales[-1]
        sqdist = oti.sqrt(
            sum((ell * (differences_by_dim[i] + 1e-6)) ** 2 for i in range(self.dim)))
        return (10 ** sigma_f) ** 2 * self.matern_kernel_prebuild(sqdist)
