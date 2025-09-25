import numpy as np
import pyoti.sparse as oti
import utils
from line_profiler import profile
# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------


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

    def __init__(self, dim, normalize, differences_by_dim, n_order, true_noise_std=None, smoothness_parameter = None):
        self.dim = dim
        self.normalize = normalize
        self.differences_by_dim = differences_by_dim
        self.true_noise_std = true_noise_std
        self.bounds = []
        if smoothness_parameter is not None:
            self.alpha = smoothness_parameter
            self.nu = 2 * smoothness_parameter + 0.5
        self.n_order = n_order

    def get_bounds_from_data(self):
        """
        Computes bounds for hyperparameters based on the observed data range.
        """
        self.bounds = []
        for diffs in self.differences_by_dim:
            min_val = float(diffs.real.min())
            max_val = float(diffs.real.max())
            self.bounds.append((-4, 10**(max_val)))

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

        Returns
        -------
        callable
            The anisotropic kernel function.
        """
        sigma_n_bound = (-16, -3)

        if kernel == "SE":
            self._add_bounds([(-1, 5), sigma_n_bound])
            return self.se_kernel_anisotropic
        elif kernel == "RQ":
            self._add_bounds([(-1, 5), (-1, 5), sigma_n_bound])
            return self.rq_kernel_anisotropic
        elif kernel == "SineExp":
            self._add_bounds([(0.0, 5)] * self.dim +
                             [(-1, 5), sigma_n_bound])
            return self.sine_exp_kernel_anisotropic
        elif kernel == "Matern":
            self._add_bounds([(-1, 5), sigma_n_bound])
            self.matern_kernel_prebuild = utils.matern_kernel_builder(self.nu)
            return self.matern_kernel_anisotropic
        elif kernel == "SI":
            self._add_bounds([(-5, 5), sigma_n_bound])
            self.SI_kernel_prebuild = utils.generate_bernoulli_lambda(self.alpha)
            return self.SI_kernel_anisotropic
        else:
            raise NotImplementedError("Anisotropic kernel not implemented")

    def _create_isotropic(self, kernel):
        """
        Sets bounds and returns the isotropic kernel function.

        Returns
        -------
        callable
            The isotropic kernel function.
        """
        sigma_n_bound = (-16, -3)

        if self.normalize:
            core_bounds = [(-3, 3)]
        else:
            self.get_bounds_from_data()
            core_bounds = [(
                float(min([d.real.min() for d in self.differences_by_dim])),
                float(max([d.real.max() for d in self.differences_by_dim]))
            )]

        if kernel == "SE":
            self.bounds = core_bounds + [(-1, 5), sigma_n_bound]
            return self.se_kernel_isotropic
        elif kernel == "RQ":
            self.bounds = core_bounds + [(-1, 5), (-1, 5), sigma_n_bound]
            return self.rq_kernel_isotropic
        elif kernel == "SineExp":
            self.bounds = core_bounds + [(0.0, 3.0), (-1, 5), sigma_n_bound]
            return self.sine_exp_kernel_isotropic
        elif kernel == "Matern":
            self.bounds = core_bounds + [(-1, 5), sigma_n_bound]
            self.matern_kernel_prebuild = utils.matern_kernel_builder(self.nu)
            return self.matern_kernel_isotropic
        elif kernel == "SI":
            self.bounds = core_bounds + [(-5, 5), sigma_n_bound]
            self.SI_kernel_prebuild = utils.generate_bernoulli_lambda(self.alpha)
            return self.SI_kernel_prebuild 
        else:
            raise NotImplementedError("Isotropic kernel not implemented")

    def _add_bounds(self, extra_bounds):
        """
        Append additional hyperparameter bounds to the kernel's configuration.

        Parameters
        ----------
        extra_bounds : list of tuple
            Bounds to append, where each tuple is a (min, max) pair in log10 scale.
        """
        if self.normalize:
            self.bounds = [(-3, 3)] * self.dim + extra_bounds
        else:
            self.bounds += extra_bounds

    # # -------------------------------------------------------------------
    # # Anisotropic Kernel Implementations
    # # -------------------------------------------------------------------
    # @profile
    # def se_kernel_anisotropic(self, differences_by_dim, length_scales, index=-1):
    #     """
    #     Anisotropic Squared Exponential (SE) kernel.

    #     Parameters
    #     ----------
    #     differences_by_dim : list of ndarray
    #     length_scales : list
    #     index : int, optional

    #     Returns
    #     -------
    #     ndarray
    #     """
    #     # print(differences_by_dim[0].shape)
    #     ell = 10 ** (length_scales[:-1])
    #     sigma_f = length_scales[-1]
    #     # sum( 0.5 * 10 ** (len_scale[i]) * ( x[i] - x'[i] )**2 )
    #     # sqdist = sum(
    #     #     (
    #     #          (-0.5 * ell[i] * ell[i] )
    #     #         *
    #     #         ( differences_by_dim[i] * differences_by_dim[i] )
    #     #     ) for i in range(self.dim)
    #     # )
    #     tmp1 = oti.zeros(differences_by_dim[0].shape)
    #     tmp2 = oti.zeros(differences_by_dim[0].shape)
    #     sqdist = oti.zeros(differences_by_dim[0].shape)
    #     for i in range(self.dim):
    #         # subdivide by terms
    #         # tmp1 = differences_by_dim[i] * differences_by_dim[i]
    #         oti.mul(differences_by_dim[i] , differences_by_dim[i], out = tmp1)
    #         t1 =  ell[i] * ell[i] * (-0.5)
    #         # tmp2 = t1 * tmp1
    #         oti.mul(t1, tmp1, out = tmp2)
    #         # sqdist += tmp2
    #         oti.sum(sqdist, tmp2, out = sqdist)
    #     # end for
    #     oti.exp( sqdist,out=tmp1)
    #     oti.mul( ((10 ** sigma_f) ** 2), tmp1, out=tmp2)
    #     # return ( (10 ** sigma_f) ** 2 ) * oti.exp(sqdist)
    #     return tmp2

    # -------------------------------------------------------------------
    # Anisotropic Kernel Implementations (MAC Mod 2)
    # -------------------------------------------------------------------
    def se_kernel_anisotropic(self, differences_by_dim, length_scales, index=-1):
        """
        Anisotropic Squared Exponential (SE) kernel.

        Parameters
        ----------
        differences_by_dim : list of ndarray
        length_scales : list
        index : int, optional

        Returns
        -------
        ndarray
        """
        # print(differences_by_dim[0].shape)
        ell = 10 ** (length_scales[:-1])
        sigma_f = length_scales[-1]
        # sum( 0.5 * 10 ** (len_scale[i]) * ( x[i] - x'[i] )**2 )
        # sqdist = sum(
        #     (
        #          (-0.5 * ell[i] * ell[i] )
        #         *
        #         ( differences_by_dim[i] * differences_by_dim[i] )
        #     ) for i in range(self.dim)
        # )
        tmp1 = oti.zeros(differences_by_dim[0].shape)
        tmp2 = oti.zeros(differences_by_dim[0].shape)
        sqdist = oti.zeros(differences_by_dim[0].shape)
        for i in range(self.dim):
            # subdivide by terms
            # tmp1 = differences_by_dim[i] * differences_by_dim[i]
            oti.mul(ell[i], differences_by_dim[i], out=tmp1)
            oti.mul(tmp1, tmp1, out=tmp2)
            # oti.pow(tmp1 , 2, out = tmp2)
            # t1 =  ell[i] * ell[i] #* (-0.5)
            # tmp2 = t1 * tmp1
            # oti.mul(t1, tmp1, out = tmp2)
            # sqdist += tmp2
            oti.sum(sqdist, tmp2, out=sqdist)
        # end for
        oti.exp((-0.5)*sqdist, out=tmp1)
        oti.mul(((10 ** sigma_f) ** 2), tmp1, out=tmp2)
        # return ( (10 ** sigma_f) ** 2 ) * oti.exp(sqdist)
        return tmp2

    # # -------------------------------------------------------------------
    # # Anisotropic Kernel Implementations (Sam's original)
    # # -------------------------------------------------------------------
    # @profile
    # def se_kernel_anisotropic(self, differences_by_dim, length_scales, index=-1):
    #     """
    #     Anisotropic Squared Exponential (SE) kernel.

    #     Parameters
    #     ----------
    #     differences_by_dim : list of ndarray
    #     length_scales : list
    #     index : int, optional

    #     Returns
    #     -------
    #     ndarray
    #     """
    #     ell = 10 ** (length_scales[:-1])
    #     sigma_f = length_scales[-1]
    #     # sqdist = sum((ell[i] * ell[i] * (differences_by_dim[i]*differences_by_dim[i] ))
    #     #              for i in range(self.dim))
    #     sqdist = sum( ( (ell[i]) * (differences_by_dim[i]))**2
    #                  for i in range(self.dim))
    #     return (10 ** sigma_f) ** 2 * oti.exp(-0.5 * sqdist)

    # @profile
    # def rq_kernel_anisotropic(self, differences_by_dim, length_scales, index=-1):
    #     """
    #     Anisotropic Rational Quadratic (RQ) kernel.
    #     """
    #     ell = 10 ** (length_scales[:self.dim])
    #     alpha = np.exp(length_scales[self.dim])
    #     sigma_f = length_scales[-1]
    #     sqdist = 1 + sum((ell[i] * differences_by_dim[i])
    #                      ** 2 for i in range(self.dim))
    #     return (10 ** sigma_f) ** 2 * (1 + sqdist / (2 * alpha)) ** (-alpha)

    def rq_kernel_anisotropic(self, differences_by_dim, length_scales, index=-1):
        """
        Anisotropic Rational Quadratic (RQ) kernel using in-place operations.

        Parameters
        ----------
        differences_by_dim : list of ndarray
        length_scales : list
        index : int, optional

        Returns
        -------
        ndarray
        """
        # --- Hyperparameter Setup ---
        ell = 10 ** (length_scales[:self.dim])
        alpha = 10 ** (length_scales[self.dim])
        sigma_f = length_scales[-1]

        # --- Pre-allocate Temporary Arrays ---
        shape = differences_by_dim[0].shape
        tmp1 = oti.zeros(shape)
        tmp2 = oti.zeros(shape)
        sqdist = oti.zeros(shape)

        # --- Calculate Squared Distance Term ---
        # sqdist = sum((ell[i] * differences_by_dim[i])**2 for i in range(self.dim))
        for i in range(self.dim):
            oti.mul(ell[i], differences_by_dim[i], out=tmp1)
            oti.mul(tmp1, tmp1, out=tmp2)
            oti.sum(sqdist, tmp2, out=sqdist)

        # --- Calculate Final Kernel Value ---
        # Formula: (10**sigma_f)**2 * (1 + sqdist / (2 * alpha))**(-alpha)

        # tmp1 = sqdist / (2 * alpha)
        oti.mul(sqdist, 1.0 / (2 * alpha), out=tmp1)

        # tmp2 = 1 + tmp1
        oti.sum(1.0, tmp1, out=tmp2)

        # tmp1 = tmp2**(-alpha)
        oti.pow(tmp2, -alpha, out=tmp1)

        # tmp2 = (10**sigma_f)**2 * tmp1
        signal_variance = (10**sigma_f)**2
        oti.mul(signal_variance, tmp1, out=tmp2)

        return tmp2

    # def sine_exp_kernel_anisotropic(self, differences_by_dim, length_scales, index=-1):
    #     """
    #     Anisotropic Sine-Exponential kernel.
    #     """
    #     ell = 10 ** (length_scales[:self.dim])
    #     p = length_scales[self.dim:-1]
    #     sigma_f = length_scales[-1]
    #     sqdist = 1 + sum((ell[i] * oti.sin((np.pi / p[i]) *
    #                      differences_by_dim[i])) ** 2 for i in range(self.dim))
    #     return (10 ** sigma_f) ** 2 * oti.exp(-2 * sqdist)

    # @profile
    def sine_exp_kernel_anisotropic(self, differences_by_dim, length_scales, index=-1):
        """
        Anisotropic Sine-Exponential kernel using in-place operations.

        This is also known as the Exp-Sine-Squared or Periodic kernel.

        Parameters
        ----------
        differences_by_dim : list of ndarray
        length_scales : list
        index : int, optional

        Returns
        -------
        ndarray
        """
        # --- Hyperparameter Setup ---
        # Note: For this kernel, length_scales typically includes length-scale (ell),
        # periodicity (p), and signal variance (sigma_f). We assume one p per dimension.
        ell = 10 ** (length_scales[:self.dim])
        # Periodicity parameter for each dimension
        p = 10 ** (length_scales[self.dim: 2 * self.dim])
        sigma_f = length_scales[-1]

        # --- Pre-allocate Temporary Arrays ---
        shape = differences_by_dim[0].shape
        tmp1 = oti.zeros(shape)
        tmp2 = oti.zeros(shape)
        sqdist = oti.zeros(shape)

        # --- Calculate the argument inside the exponent ---
        # sqdist = sum( ( sin(π * |x-x'| / p) / ell )^2 )
        # The user's original formula was slightly different, this is a common form.
        # We will implement: sum((ell[i] * sin((π / p[i]) * diff)) ** 2)
        for i in range(self.dim):
            # tmp1 = (np.pi / p[i]) * differences_by_dim[i]
            oti.mul(np.pi / p[i], differences_by_dim[i], out=tmp1)

            # tmp2 = sin(tmp1)
            oti.sin(tmp1, out=tmp2)

            # tmp1 = ell[i] * tmp2
            oti.mul(ell[i], tmp2, out=tmp1)

            # tmp2 = tmp1**2
            oti.mul(tmp1, tmp1, out=tmp2)

            # sqdist += tmp2
            oti.sum(sqdist, tmp2, out=sqdist)

        # --- Calculate Final Kernel Value ---
        # Formula: (10**sigma_f)**2 * exp(-2 * sqdist)

        # tmp1 = -2 * sqdist
        oti.mul(-2.0, sqdist, out=tmp1)

        # tmp2 = exp(tmp1)
        oti.exp(tmp1, out=tmp2)

        # tmp1 = (10**sigma_f)**2 * tmp2
        signal_variance = (10**sigma_f)**2
        oti.mul(signal_variance, tmp2, out=tmp1)

        return tmp1

    def matern_kernel_anisotropic(self, differences_by_dim, length_scales, index=-1):
        """
        Anisotropic Matérn kernel (half-integer ν).
        """
        ell = 10 ** (length_scales[:-1])
        sigma_f = length_scales[-1]
        sqdist = oti.sqrt(
            sum((ell[i] * (differences_by_dim[i] + 1e-6)) ** 2 for i in range(self.dim)))
        return (10 ** sigma_f) ** 2 * self.matern_kernel_prebuild(sqdist)
    def SI_kernel_anisotropic(self, differences_by_dim, length_scales, index=-1):
        """
        Anisotropic Matérn kernel (half-integer ν).
        """
        ell = 10 ** (length_scales[:-1])
        sigma_f = length_scales[-1]
        val = 1
        for i in range(self.dim):
            val = val * (1 + ell[i] * self.SI_kernel_prebuild(differences_by_dim[i]))
        return (10 ** sigma_f) ** 2 * val

    # -------------------------------------------------------------------
    # Isotropic Kernel Implementations
    # -------------------------------------------------------------------

    def se_kernel_isotropic(self, differences_by_dim, length_scales, index=-1):
        """
        Isotropic Squared Exponential (SE) kernel.
        """
        # print(differences_by_dim[0].shape)
        ell = 10 ** length_scales[0]
        sigma_f = length_scales[-1]
        # sum( 0.5 * 10 ** (len_scale[i]) * ( x[i] - x'[i] )**2 )
        # sqdist = sum(
        #     (
        #          (-0.5 * ell[i] * ell[i] )
        #         *
        #         ( differences_by_dim[i] * differences_by_dim[i] )
        #     ) for i in range(self.dim)
        # )
        tmp1 = oti.zeros(differences_by_dim[0].shape)
        tmp2 = oti.zeros(differences_by_dim[0].shape)
        sqdist = oti.zeros(differences_by_dim[0].shape)
        for i in range(self.dim):
            # subdivide by terms
            # tmp1 = differences_by_dim[i] * differences_by_dim[i]
            oti.mul(ell, differences_by_dim[i], out=tmp1)
            oti.mul(tmp1, tmp1, out=tmp2)
            # oti.pow(tmp1 , 2, out = tmp2)
            # t1 =  ell[i] * ell[i] #* (-0.5)
            # tmp2 = t1 * tmp1
            # oti.mul(t1, tmp1, out = tmp2)
            # sqdist += tmp2
            oti.sum(sqdist, tmp2, out=sqdist)
        # end for
        oti.exp((-0.5)*sqdist, out=tmp1)
        oti.mul(((10 ** sigma_f) ** 2), tmp1, out=tmp2)
        # return ( (10 ** sigma_f) ** 2 ) * oti.exp(sqdist)
        return tmp2

    def rq_kernel_isotropic(self, differences_by_dim, length_scales, index=-1):
        """
        Isotropic Rational Quadratic (RQ) kernel.
        """
        # --- Hyperparameter Setup ---
        ell = 10 ** length_scales[0]
        alpha = np.exp(length_scales[1])
        sigma_f = length_scales[-1]

        # --- Pre-allocate Temporary Arrays ---
        shape = differences_by_dim[0].shape
        tmp1 = oti.zeros(shape)
        tmp2 = oti.zeros(shape)
        sqdist = oti.zeros(shape)

        # --- Calculate Squared Distance Term ---
        # sqdist = sum((ell[i] * differences_by_dim[i])**2 for i in range(self.dim))
        for i in range(self.dim):
            oti.mul(ell, differences_by_dim[i], out=tmp1)
            oti.mul(tmp1, tmp1, out=tmp2)
            oti.sum(sqdist, tmp2, out=sqdist)

        # --- Calculate Final Kernel Value ---
        # Formula: (10**sigma_f)**2 * (1 + sqdist / (2 * alpha))**(-alpha)

        # tmp1 = sqdist / (2 * alpha)
        oti.mul(sqdist, 1.0 / (2 * alpha), out=tmp1)

        # tmp2 = 1 + tmp1
        oti.sum(1.0, tmp1, out=tmp2)

        # tmp1 = tmp2**(-alpha)
        oti.pow(tmp2, -alpha, out=tmp1)

        # tmp2 = (10**sigma_f)**2 * tmp1
        signal_variance = (10**sigma_f)**2
        oti.mul(signal_variance, tmp1, out=tmp2)

        return tmp2

    def sine_exp_kernel_isotropic(self, differences_by_dim, length_scales, index=-1):
        """
        Isotropic Sine-Exponential kernel.
        """
        ell = 10 ** length_scales[0]
        p = 10**(length_scales[1])
        sigma_f = length_scales[-1]

        # --- Pre-allocate Temporary Arrays ---
        shape = differences_by_dim[0].shape
        tmp1 = oti.zeros(shape)
        tmp2 = oti.zeros(shape)
        sqdist = oti.zeros(shape)

        # --- Calculate the argument inside the exponent ---
        # sqdist = sum( ( sin(π * |x-x'| / p) / ell )^2 )
        # The user's original formula was slightly different, this is a common form.
        # We will implement: sum((ell[i] * sin((π / p[i]) * diff)) ** 2)
        for i in range(self.dim):
            # tmp1 = (np.pi / p[i]) * differences_by_dim[i]
            oti.mul(np.pi / p, differences_by_dim[i], out=tmp1)

            # tmp2 = sin(tmp1)
            oti.sin(tmp1, out=tmp2)

            # tmp1 = ell[i] * tmp2
            oti.mul(ell, tmp2, out=tmp1)

            # tmp2 = tmp1**2
            oti.mul(tmp1, tmp1, out=tmp2)

            # sqdist += tmp2
            oti.sum(sqdist, tmp2, out=sqdist)

        # --- Calculate Final Kernel Value ---
        # Formula: (10**sigma_f)**2 * exp(-2 * sqdist)

        # tmp1 = -2 * sqdist
        oti.mul(-2.0, sqdist, out=tmp1)

        # tmp2 = exp(tmp1)
        oti.exp(tmp1, out=tmp2)

        # tmp1 = (10**sigma_f)**2 * tmp2
        signal_variance = (10**sigma_f)**2
        oti.mul(signal_variance, tmp2, out=tmp1)

        return tmp1

    def matern_kernel_isotropic(self, differences_by_dim, length_scales, index=-1):
        """
        Isotropic Matérn kernel (half-integer ν).
        """
        ell = 10 ** length_scales[0]
        sigma_f = length_scales[-1]
        sqdist = oti.sqrt(
            sum((ell * (differences_by_dim[i] + 1e-6)) ** 2 for i in range(self.dim)))
        return (10 ** sigma_f) ** 2 * self.matern_kernel_prebuild(sqdist)
