import numpy as np
import pyoti.sparse as oti
import jetgp.utils
from line_profiler import profile


class KernelFactory:
    """
    Factory for generating different kernel functions (SE, RQ, SineExp, Matérn)
    in isotropic and anisotropic forms with caching for improved performance.

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

    def __init__(self, dim, normalize, differences_by_dim, n_order, 
                 true_noise_std=None, smoothness_parameter=None):
        self.dim = dim
        self.normalize = normalize
        self.differences_by_dim = differences_by_dim
        self.true_noise_std = true_noise_std
        self.bounds = []
        if smoothness_parameter is not None:
            self.alpha = smoothness_parameter
            self.nu = smoothness_parameter + 0.5
        self.n_order = n_order
        
        # Initialize caching infrastructure
        self._init_caches()

    # -------------------------------------------------------------------
    # Caching Infrastructure
    # -------------------------------------------------------------------

    def _init_caches(self):
        """Initialize all cache variables."""
        # Temporary array cache
        self._cached_shape = None
        self._tmp1 = None
        self._tmp2 = None
        self._sqdist = None
        
        # Hyperparameter cache
        self._cached_length_scales = None
        self._cached_ell = None
        self._cached_sigma_f_sq = None
        self._cached_alpha = None
        self._cached_p = None
        self._cached_pi_over_p = None

    def clear_caches(self):
        """Clear all caches. Call when training data changes."""
        self._init_caches()

    def _ensure_temp_arrays(self, shape):
        """
        Ensure temporary arrays exist with correct shape.
        
        Parameters
        ----------
        shape : tuple
            Required shape for temporary arrays.
            
        Returns
        -------
        tuple
            (tmp1, tmp2, sqdist) temporary arrays.
        """
        if self._cached_shape != shape:
            self._cached_shape = shape
            self._tmp1 = oti.zeros(shape)
            self._tmp2 = oti.zeros(shape)
            self._sqdist = oti.zeros(shape)
        return self._tmp1, self._tmp2, self._sqdist

    def _reset_sqdist(self):
        """Reset sqdist accumulator to zero."""
        if self._sqdist is None:
            return
        # Use the most efficient method available in oti
        if hasattr(self._sqdist, 'fill'):
            self._sqdist.fill(0)
        elif hasattr(oti, 'set_zero'):
            oti.set_zero(self._sqdist)
        else:
            # Fallback: multiply by zero
            oti.mul(0.0, self._sqdist, out=self._sqdist)

    # -------------------------------------------------------------------
    # Hyperparameter Caching Methods
    # -------------------------------------------------------------------

    def _cache_se_params_aniso(self, length_scales):
        """Cache anisotropic SE kernel hyperparameters."""
        ls_tuple = tuple(float(x) for x in length_scales)
        if self._cached_length_scales != ('se_aniso', ls_tuple):
            self._cached_length_scales = ('se_aniso', ls_tuple)
            self._cached_ell = 10 ** np.array(length_scales[:-1])
            self._cached_sigma_f_sq = (10 ** length_scales[-1]) ** 2
        return self._cached_ell, self._cached_sigma_f_sq

    def _cache_se_params_iso(self, length_scales):
        """Cache isotropic SE kernel hyperparameters."""
        ls_tuple = tuple(float(x) for x in length_scales)
        if self._cached_length_scales != ('se_iso', ls_tuple):
            self._cached_length_scales = ('se_iso', ls_tuple)
            self._cached_ell = 10 ** length_scales[0]
            self._cached_sigma_f_sq = (10 ** length_scales[-1]) ** 2
        return self._cached_ell, self._cached_sigma_f_sq

    def _cache_rq_params_aniso(self, length_scales):
        """Cache anisotropic RQ kernel hyperparameters."""
        ls_tuple = tuple(float(x) for x in length_scales)
        if self._cached_length_scales != ('rq_aniso', ls_tuple):
            self._cached_length_scales = ('rq_aniso', ls_tuple)
            self._cached_ell = 10 ** np.array(length_scales[:self.dim])
            self._cached_alpha = 10 ** length_scales[self.dim]
            self._cached_sigma_f_sq = (10 ** length_scales[-1]) ** 2
        return self._cached_ell, self._cached_alpha, self._cached_sigma_f_sq

    def _cache_rq_params_iso(self, length_scales):
        """Cache isotropic RQ kernel hyperparameters."""
        ls_tuple = tuple(float(x) for x in length_scales)
        if self._cached_length_scales != ('rq_iso', ls_tuple):
            self._cached_length_scales = ('rq_iso', ls_tuple)
            self._cached_ell = 10 ** length_scales[0]
            self._cached_alpha = np.exp(length_scales[1])
            self._cached_sigma_f_sq = (10 ** length_scales[-1]) ** 2
        return self._cached_ell, self._cached_alpha, self._cached_sigma_f_sq

    def _cache_sine_exp_params_aniso(self, length_scales):
        """Cache anisotropic Sine-Exponential kernel hyperparameters."""
        ls_tuple = tuple(float(x) for x in length_scales)
        if self._cached_length_scales != ('sine_exp_aniso', ls_tuple):
            self._cached_length_scales = ('sine_exp_aniso', ls_tuple)
            self._cached_ell = 10 ** np.array(length_scales[:self.dim])
            self._cached_p = 10 ** np.array(length_scales[self.dim:2*self.dim])
            self._cached_pi_over_p = np.pi / self._cached_p
            self._cached_sigma_f_sq = (10 ** length_scales[-1]) ** 2
        return self._cached_ell, self._cached_pi_over_p, self._cached_sigma_f_sq

    def _cache_sine_exp_params_iso(self, length_scales):
        """Cache isotropic Sine-Exponential kernel hyperparameters."""
        ls_tuple = tuple(float(x) for x in length_scales)
        if self._cached_length_scales != ('sine_exp_iso', ls_tuple):
            self._cached_length_scales = ('sine_exp_iso', ls_tuple)
            self._cached_ell = 10 ** length_scales[0]
            self._cached_p = 10 ** length_scales[1]
            self._cached_pi_over_p = np.pi / self._cached_p
            self._cached_sigma_f_sq = (10 ** length_scales[-1]) ** 2
        return self._cached_ell, self._cached_pi_over_p, self._cached_sigma_f_sq

    def _cache_matern_params_aniso(self, length_scales):
        """Cache anisotropic Matern kernel hyperparameters."""
        ls_tuple = tuple(float(x) for x in length_scales)
        if self._cached_length_scales != ('matern_aniso', ls_tuple):
            self._cached_length_scales = ('matern_aniso', ls_tuple)
            self._cached_ell = 10 ** np.array(length_scales[:-1])
            self._cached_sigma_f_sq = (10 ** length_scales[-1]) ** 2
        return self._cached_ell, self._cached_sigma_f_sq

    def _cache_matern_params_iso(self, length_scales):
        """Cache isotropic Matern kernel hyperparameters."""
        ls_tuple = tuple(float(x) for x in length_scales)
        if self._cached_length_scales != ('matern_iso', ls_tuple):
            self._cached_length_scales = ('matern_iso', ls_tuple)
            self._cached_ell = 10 ** length_scales[0]
            self._cached_sigma_f_sq = (10 ** length_scales[-1]) ** 2
        return self._cached_ell, self._cached_sigma_f_sq

    def _cache_si_params_aniso(self, length_scales):
        """Cache anisotropic SI kernel hyperparameters."""
        ls_tuple = tuple(float(x) for x in length_scales)
        if self._cached_length_scales != ('si_aniso', ls_tuple):
            self._cached_length_scales = ('si_aniso', ls_tuple)
            self._cached_ell = 10 ** np.array(length_scales[:-1])
            self._cached_sigma_f_sq = (10 ** length_scales[-1]) ** 2
        return self._cached_ell, self._cached_sigma_f_sq

    def _cache_si_params_iso(self, length_scales):
        """Cache isotropic SI kernel hyperparameters."""
        ls_tuple = tuple(float(x) for x in length_scales)
        if self._cached_length_scales != ('si_iso', ls_tuple):
            self._cached_length_scales = ('si_iso', ls_tuple)
            self._cached_ell = 10 ** length_scales[0]
            self._cached_sigma_f_sq = (10 ** length_scales[-1]) ** 2
        return self._cached_ell, self._cached_sigma_f_sq

    # -------------------------------------------------------------------
    # Bounds and Factory Methods
    # -------------------------------------------------------------------

    def get_bounds_from_data(self):
        """
        Computes bounds for hyperparameters based on the observed data range.
        """
        self.bounds = []
        for diffs in self.differences_by_dim:
            min_val = float(diffs.real.min())
            max_val = float(diffs.real.max())
            self.bounds.append((-3, np.log(max_val)))

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
        # Clear caches when creating a new kernel
        self.clear_caches()
        
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
            self._add_bounds([(0.0, 5)] * self.dim + [(-1, 5), sigma_n_bound])
            return self.sine_exp_kernel_anisotropic
        elif kernel == "Matern":
            self._add_bounds([(-1, 5), sigma_n_bound])
            self.matern_kernel_prebuild = jetgp.utils.matern_kernel_builder(self.nu)
            return self.matern_kernel_anisotropic
        elif kernel == "SI":
            self._add_bounds([(-5, 5), sigma_n_bound])
            self.SI_kernel_prebuild = jetgp.utils.generate_bernoulli_lambda(self.alpha)
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
            self.matern_kernel_prebuild = jetgp.utils.matern_kernel_builder(self.nu)
            return self.matern_kernel_isotropic
        elif kernel == "SI":
            self.bounds = core_bounds + [(-5, 5), sigma_n_bound]
            self.SI_kernel_prebuild = jetgp.utils.generate_bernoulli_lambda(self.alpha)
            return self.SI_kernel_isotropic
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

    # -------------------------------------------------------------------
    # Anisotropic Kernel Implementations with Caching
    # -------------------------------------------------------------------

    @profile
    def se_kernel_anisotropic(self, differences_by_dim, length_scales):
        """
        Anisotropic Squared Exponential (SE) kernel with caching.

        Parameters
        ----------
        differences_by_dim : list of ndarray
            Pairwise differences by dimension.
        length_scales : list
            Hyperparameters: [ell_1, ..., ell_dim, sigma_f]

        Returns
        -------
        ndarray
            Kernel matrix values.
        """
        ell, sigma_f_sq = self._cache_se_params_aniso(length_scales)
        tmp1, tmp2, sqdist = self._ensure_temp_arrays(differences_by_dim[0].shape)
        self._reset_sqdist()

        for i in range(self.dim):
            oti.mul(ell[i], differences_by_dim[i], out=tmp1)
            oti.mul(tmp1, tmp1, out=tmp2)
            oti.sum(sqdist, tmp2, out=sqdist)

        oti.exp((-0.5) * sqdist, out=tmp1)
        oti.mul(sigma_f_sq, tmp1, out=tmp2)
        return tmp2

    def rq_kernel_anisotropic(self, differences_by_dim, length_scales):
        """
        Anisotropic Rational Quadratic (RQ) kernel with caching.

        Parameters
        ----------
        differences_by_dim : list of ndarray
            Pairwise differences by dimension.
        length_scales : list
            Hyperparameters: [ell_1, ..., ell_dim, alpha, sigma_f]

        Returns
        -------
        ndarray
            Kernel matrix values.
        """
        ell, alpha, sigma_f_sq = self._cache_rq_params_aniso(length_scales)
        tmp1, tmp2, sqdist = self._ensure_temp_arrays(differences_by_dim[0].shape)
        self._reset_sqdist()

        for i in range(self.dim):
            oti.mul(ell[i], differences_by_dim[i], out=tmp1)
            oti.mul(tmp1, tmp1, out=tmp2)
            oti.sum(sqdist, tmp2, out=sqdist)

        # (1 + sqdist / (2 * alpha))^(-alpha)
        inv_2alpha = 1.0 / (2 * alpha)
        oti.mul(sqdist, inv_2alpha, out=tmp1)
        oti.sum(1.0, tmp1, out=tmp2)
        oti.pow(tmp2, -alpha, out=tmp1)
        oti.mul(sigma_f_sq, tmp1, out=tmp2)
        return tmp2

    def sine_exp_kernel_anisotropic(self, differences_by_dim, length_scales):
        """
        Anisotropic Sine-Exponential (Periodic) kernel with caching.

        Parameters
        ----------
        differences_by_dim : list of ndarray
            Pairwise differences by dimension.
        length_scales : list
            Hyperparameters: [ell_1, ..., ell_dim, p_1, ..., p_dim, sigma_f]

        Returns
        -------
        ndarray
            Kernel matrix values.
        """
        ell, pi_over_p, sigma_f_sq = self._cache_sine_exp_params_aniso(length_scales)
        tmp1, tmp2, sqdist = self._ensure_temp_arrays(differences_by_dim[0].shape)
        self._reset_sqdist()

        for i in range(self.dim):
            oti.mul(pi_over_p[i], differences_by_dim[i], out=tmp1)
            oti.sin(tmp1, out=tmp2)
            oti.mul(ell[i], tmp2, out=tmp1)
            oti.mul(tmp1, tmp1, out=tmp2)
            oti.sum(sqdist, tmp2, out=sqdist)

        oti.mul(-2.0, sqdist, out=tmp1)
        oti.exp(tmp1, out=tmp2)
        oti.mul(sigma_f_sq, tmp2, out=tmp1)
        return tmp1

    def matern_kernel_anisotropic(self, differences_by_dim, length_scales):
        """
        Anisotropic Matérn kernel (half-integer ν) with caching.

        Parameters
        ----------
        differences_by_dim : list of ndarray
            Pairwise differences by dimension.
        length_scales : list
            Hyperparameters: [ell_1, ..., ell_dim, sigma_f]

        Returns
        -------
        ndarray
            Kernel matrix values.
        """
        ell, sigma_f_sq = self._cache_matern_params_aniso(length_scales)
        
        # Compute scaled distance
        sqdist = oti.sqrt(
            sum((ell[i] * (differences_by_dim[i] + 1e-16)) ** 2 
                for i in range(self.dim))
        )
        return sigma_f_sq * self.matern_kernel_prebuild(sqdist)

    def SI_kernel_anisotropic(self, differences_by_dim, length_scales):
        """
        Anisotropic SI kernel with caching.

        Parameters
        ----------
        differences_by_dim : list of ndarray
            Pairwise differences by dimension.
        length_scales : list
            Hyperparameters: [ell_1, ..., ell_dim, sigma_f]

        Returns
        -------
        ndarray
            Kernel matrix values.
        """
        ell, sigma_f_sq = self._cache_si_params_aniso(length_scales)
        
        val = 1
        for i in range(self.dim):
            val = val * (1 + ell[i] * self.SI_kernel_prebuild(differences_by_dim[i]))
        return sigma_f_sq * val

    # -------------------------------------------------------------------
    # Isotropic Kernel Implementations with Caching
    # -------------------------------------------------------------------

    def se_kernel_isotropic(self, differences_by_dim, length_scales):
        """
        Isotropic Squared Exponential (SE) kernel with caching.

        Parameters
        ----------
        differences_by_dim : list of ndarray
            Pairwise differences by dimension.
        length_scales : list
            Hyperparameters: [ell, sigma_f]

        Returns
        -------
        ndarray
            Kernel matrix values.
        """
        ell, sigma_f_sq = self._cache_se_params_iso(length_scales)
        tmp1, tmp2, sqdist = self._ensure_temp_arrays(differences_by_dim[0].shape)
        self._reset_sqdist()

        for i in range(self.dim):
            oti.mul(ell, differences_by_dim[i], out=tmp1)
            oti.mul(tmp1, tmp1, out=tmp2)
            oti.sum(sqdist, tmp2, out=sqdist)

        oti.exp((-0.5) * sqdist, out=tmp1)
        oti.mul(sigma_f_sq, tmp1, out=tmp2)
        return tmp2

    def rq_kernel_isotropic(self, differences_by_dim, length_scales):
        """
        Isotropic Rational Quadratic (RQ) kernel with caching.

        Parameters
        ----------
        differences_by_dim : list of ndarray
            Pairwise differences by dimension.
        length_scales : list
            Hyperparameters: [ell, alpha, sigma_f]

        Returns
        -------
        ndarray
            Kernel matrix values.
        """
        ell, alpha, sigma_f_sq = self._cache_rq_params_iso(length_scales)
        tmp1, tmp2, sqdist = self._ensure_temp_arrays(differences_by_dim[0].shape)
        self._reset_sqdist()

        for i in range(self.dim):
            oti.mul(ell, differences_by_dim[i], out=tmp1)
            oti.mul(tmp1, tmp1, out=tmp2)
            oti.sum(sqdist, tmp2, out=sqdist)

        # (1 + sqdist / (2 * alpha))^(-alpha)
        inv_2alpha = 1.0 / (2 * alpha)
        oti.mul(sqdist, inv_2alpha, out=tmp1)
        oti.sum(1.0, tmp1, out=tmp2)
        oti.pow(tmp2, -alpha, out=tmp1)
        oti.mul(sigma_f_sq, tmp1, out=tmp2)
        return tmp2

    def sine_exp_kernel_isotropic(self, differences_by_dim, length_scales):
        """
        Isotropic Sine-Exponential (Periodic) kernel with caching.

        Parameters
        ----------
        differences_by_dim : list of ndarray
            Pairwise differences by dimension.
        length_scales : list
            Hyperparameters: [ell, p, sigma_f]

        Returns
        -------
        ndarray
            Kernel matrix values.
        """
        ell, pi_over_p, sigma_f_sq = self._cache_sine_exp_params_iso(length_scales)
        tmp1, tmp2, sqdist = self._ensure_temp_arrays(differences_by_dim[0].shape)
        self._reset_sqdist()

        for i in range(self.dim):
            oti.mul(pi_over_p, differences_by_dim[i], out=tmp1)
            oti.sin(tmp1, out=tmp2)
            oti.mul(ell, tmp2, out=tmp1)
            oti.mul(tmp1, tmp1, out=tmp2)
            oti.sum(sqdist, tmp2, out=sqdist)

        oti.mul(-2.0, sqdist, out=tmp1)
        oti.exp(tmp1, out=tmp2)
        oti.mul(sigma_f_sq, tmp2, out=tmp1)
        return tmp1

    def matern_kernel_isotropic(self, differences_by_dim, length_scales):
        """
        Isotropic Matérn kernel (half-integer ν) with caching.

        Parameters
        ----------
        differences_by_dim : list of ndarray
            Pairwise differences by dimension.
        length_scales : list
            Hyperparameters: [ell, sigma_f]

        Returns
        -------
        ndarray
            Kernel matrix values.
        """
        ell, sigma_f_sq = self._cache_matern_params_iso(length_scales)
        
        # Compute scaled distance
        sqdist = oti.sqrt(
            sum((ell * (differences_by_dim[i] + 1e-16)) ** 2 
                for i in range(self.dim))
        )
        return sigma_f_sq * self.matern_kernel_prebuild(sqdist)

    def SI_kernel_isotropic(self, differences_by_dim, length_scales):
        """
        Isotropic SI kernel with caching.

        Parameters
        ----------
        differences_by_dim : list of ndarray
            Pairwise differences by dimension.
        length_scales : list
            Hyperparameters: [ell, sigma_f]

        Returns
        -------
        ndarray
            Kernel matrix values.
        """
        ell, sigma_f_sq = self._cache_si_params_iso(length_scales)
        
        val = 1
        for i in range(self.dim):
            val = val * (1 + ell * self.SI_kernel_prebuild(differences_by_dim[i]))
        return sigma_f_sq * val