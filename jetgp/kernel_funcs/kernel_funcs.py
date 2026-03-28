import numpy as np
import jetgp.utils
from line_profiler import profile
import importlib
import subprocess
import sys
import os
import warnings


def get_oti_module(n_bases, n_order, auto_compile=True, otilib_path=None, use_sparse=False):
    """
    Dynamically import the correct PyOTI static library.
    If the module doesn't exist and auto_compile=True, attempts to compile it.
    Falls back to pyoti.sparse if compilation fails or is disabled.

    Parameters
    ----------
    n_bases : int
        Number of bases (dimension of the input space).
    n_order : int
        Derivative order for the GP. The OTI order will be 2*n_order.
    auto_compile : bool, optional (default=False)
        If True, attempt to compile missing modules automatically.
        Requires jetgp.cmod_writer and jetgp.build_static to be available.
    otilib_path : str, optional
        Path to otilib-master directory. If None, attempts auto-detection.

    -------
    module
        The appropriate pyoti.static.onummXnY module, or pyoti.sparse as fallback.
    """
    if n_order == 0:
        module_name = "pyoti.real"
        return importlib.import_module(module_name)

    oti_order = 2 * n_order
    module_name = f"pyoti.static.onumm{n_bases}n{oti_order}"
    if use_sparse:
        return importlib.import_module("pyoti.sparse")
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        if not auto_compile:
            warnings.warn(
                f"PyOTI static module '{module_name}' not found. "
                f"Falling back to pyoti.sparse which is significantly slower.\n"
                f"For better performance, compile the static module manually:\n"
                f"  1. cd /path/to/otilib-master/build\n"
                f"  2. cmake ..\n"
                f"  3. make m{n_bases}n{oti_order} -j8\n"
                f"  4. python build_static.py m{n_bases}n{oti_order}",
                UserWarning
            )
            return importlib.import_module("pyoti.sparse")

        # Check if auto-compile tools are available
        try:
            from jetgp.cmod_writer import writer
            from jetgp.build_static import build_module
        except ImportError:
            warnings.warn(
                f"PyOTI static module '{module_name}' not found and auto-compile "
                f"tools (jetgp.cmod_writer, jetgp.build_static) are not available.\n"
                f"Falling back to pyoti.sparse which is significantly slower.\n"
                f"For better performance, compile the static module manually:\n"
                f"  1. cd /path/to/otilib-master/build\n"
                f"  2. cmake ..\n"
                f"  3. make m{n_bases}n{oti_order} -j8\n"
                f"  4. python build_static.py m{n_bases}n{oti_order}",
                UserWarning
            )
            return importlib.import_module("pyoti.sparse")

        print(f"Module '{module_name}' not found. Attempting to compile...")

        try:
            _compile_oti_module(n_bases, oti_order, otilib_path)

            # Clear import caches and retry
            importlib.invalidate_caches()

            return importlib.import_module(module_name)
        except Exception as e:
            warnings.warn(
                f"Failed to compile PyOTI static module '{module_name}': {e}\n"
                f"Falling back to pyoti.sparse which is significantly slower.\n"
                f"For better performance, compile the static module manually:\n"
                f"  1. cd /path/to/otilib-master/build\n"
                f"  2. cmake ..\n"
                f"  3. make m{n_bases}n{oti_order} -j8\n"
                f"  4. python build_static.py m{n_bases}n{oti_order}",
                UserWarning
            )
            return importlib.import_module("pyoti.sparse")


def _get_otilib_path(otilib_path=None):
    """
    Auto-detect otilib path from the installed pyoti package location.

    Parameters
    ----------
    otilib_path : str, optional
        Override path to otilib-master directory.

    Returns
    -------
    otilib_path : str
        Path to otilib-master directory.
    """
    # Use explicit argument if provided
    if otilib_path is not None:
        if not os.path.isdir(otilib_path):
            raise RuntimeError(f"otilib path does not exist: {otilib_path}")
        return otilib_path

    # Check environment variable
    otilib_path = os.environ.get('OTILIB_PATH')
    if otilib_path is not None:
        if not os.path.isdir(otilib_path):
            raise RuntimeError(f"OTILIB_PATH does not exist: {otilib_path}")
        return otilib_path

    # Auto-detect from installed pyoti
    try:
        import pyoti

        # Get the pyoti package location
        if hasattr(pyoti, '__path__'):
            pyoti_install_path = pyoti.__path__[0]
        elif hasattr(pyoti, '__file__'):
            pyoti_install_path = os.path.dirname(pyoti.__file__)
        else:
            raise AttributeError("Cannot determine pyoti installation path")

        # Navigate up from pyoti to find otilib root
        # Typical structure: otilib-master/src/python/pyoti/pyoti/__init__.py
        current = pyoti_install_path
        for _ in range(6):  # Navigate up to 6 levels
            parent = os.path.dirname(current)

            # Check if this looks like otilib root
            # Must have BOTH CMakeLists.txt AND src/ directory (not just build dir)
            potential_cmake = os.path.join(parent, 'CMakeLists.txt')
            potential_src = os.path.join(parent, 'src')
            potential_include = os.path.join(parent, 'include')

            if (os.path.isfile(potential_cmake) and
                os.path.isdir(potential_src) and
                    os.path.isdir(potential_include)):
                otilib_path = parent
                break

            current = parent

    except ImportError:
        pass

    # Final validation
    if otilib_path is None:
        raise RuntimeError(
            "Could not auto-detect otilib path. Please either:\n"
            "  1. Set the OTILIB_PATH environment variable\n"
            "  2. Pass otilib_path explicitly to get_oti_module()\n"
            "  3. Ensure pyoti is installed from the otilib source tree"
        )

    if not os.path.isdir(otilib_path):
        raise RuntimeError(f"otilib path does not exist: {otilib_path}")

    return otilib_path


def _compile_oti_module(n_bases, oti_order, otilib_path=None):
    """
    Compile a PyOTI static module.

    Parameters
    ----------
    n_bases : int
        Number of bases.
    oti_order : int
        OTI order (already multiplied by 2).
    otilib_path : str, optional
        Path to otilib-master directory.
    """
    # Auto-detect path
    otilib_path = _get_otilib_path(otilib_path)

    build_dir = os.path.join(otilib_path, 'build')
    module_target = f"m{n_bases}n{oti_order}"

    print(f"Compiling OTI module: {module_target}")
    print(f"  otilib_path: {otilib_path}")
    print(f"  build_dir: {build_dir}")

    # Step 1: Generate C code using cmod_writer (from jetgp)
    print(f"Step 1/4: Generating C code for m={n_bases}, n={oti_order}...")
    _run_cmod_writer(n_bases, oti_order, otilib_path)

    # Step 2: Run cmake (if needed)
    print("Step 2/4: Running cmake...")
    _run_cmake(build_dir)

    # Step 3: Run make
    print(f"Step 3/4: Compiling {module_target}...")
    _run_make(build_dir, module_target)

    # Step 4: Build and install Python module
    print(f"Step 4/4: Building Python module...")
    _run_build_static(build_dir, module_target, otilib_path)

    print(f"Successfully compiled {module_target}")


def _run_cmod_writer(n_bases, oti_order, otilib_path):
    """Generate C code using cmod_writer from jetgp."""
    from jetgp.cmod_writer import writer

    w = writer(nbases=n_bases, order=oti_order)
    w.write_files(base_dir=otilib_path)


def _run_cmake(build_dir):
    """Run cmake in the build directory."""
    os.makedirs(build_dir, exist_ok=True)

    result = subprocess.run(
        ['cmake', '..'],
        cwd=build_dir,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"cmake failed with return code {result.returncode}.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


def _run_make(build_dir, module_target, n_jobs=8):
    """Run make for the specific module target."""
    result = subprocess.run(
        ['make', module_target, f'-j{n_jobs}'],
        cwd=build_dir,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"make failed with return code {result.returncode}.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


def _run_build_static(build_dir, module_target, otilib_path):
    """Build and install the Python module using jetgp's build_static."""
    from jetgp.build_static import build_module

    build_module(module_target, otilib_path=otilib_path, build_dir=build_dir)


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
                 true_noise_std=None, smoothness_parameter=None, oti_module=None):
        self.dim = dim
        self.normalize = normalize
        self.differences_by_dim = differences_by_dim
        self.true_noise_std = true_noise_std
        self.bounds = []
        self.oti = oti_module
        if smoothness_parameter is not None:
            self.alpha = smoothness_parameter
            self.nu = smoothness_parameter + 0.5
        else:
            self.alpha = 1
            self.nu = 1.5
        self.n_order = n_order
        # Dynamic OTI import

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
            self._tmp1 = self.oti.zeros(shape)
            self._tmp2 = self.oti.zeros(shape)
            self._sqdist = self.oti.zeros(shape)
        return self._tmp1, self._tmp2, self._sqdist

    def _reset_sqdist(self):
        """Reset sqdist accumulator to zero."""
        if self._sqdist is None:
            return
        # Use the most efficient method available in oti
        if hasattr(self._sqdist, 'fill'):
            self._sqdist.fill(0)
        elif hasattr(self.oti, 'set_zero'):
            self.oti.set_zero(self._sqdist)
        else:
            # Fallback: multiply by zero
            self.oti.mul(0.0, self._sqdist, out=self._sqdist)

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
            self.matern_kernel_prebuild = jetgp.utils.matern_kernel_builder(
                self.nu, oti_module=self.oti)
            return self.matern_kernel_anisotropic
        elif kernel == "SI":
            self._add_bounds([(-5, 5), sigma_n_bound])
            self.SI_kernel_prebuild = jetgp.utils.generate_bernoulli_lambda(
                self.alpha)
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
            self.matern_kernel_prebuild = jetgp.utils.matern_kernel_builder(
                self.nu, oti_module=self.oti)
            return self.matern_kernel_isotropic
        elif kernel == "SI":
            self.bounds = core_bounds + [(-5, 5), sigma_n_bound]
            self.SI_kernel_prebuild = jetgp.utils.generate_bernoulli_lambda(
                self.alpha)
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
        tmp1, tmp2, sqdist = self._ensure_temp_arrays(
            differences_by_dim[0].shape)
        self._reset_sqdist()

        for i in range(self.dim):
            self.oti.mul(ell[i], differences_by_dim[i], out=tmp1)
            self.oti.mul(tmp1, tmp1, out=tmp2)
            self.oti.sum(sqdist, tmp2, out=sqdist)

        self.oti.exp((-0.5) * sqdist, out=tmp1)
        self.oti.mul(sigma_f_sq, tmp1, out=tmp2)
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
        tmp1, tmp2, sqdist = self._ensure_temp_arrays(
            differences_by_dim[0].shape)
        self._reset_sqdist()

        for i in range(self.dim):
            self.oti.mul(ell[i], differences_by_dim[i], out=tmp1)
            self.oti.mul(tmp1, tmp1, out=tmp2)
            self.oti.sum(sqdist, tmp2, out=sqdist)

        # (1 + sqdist / (2 * alpha))^(-alpha)
        inv_2alpha = 1.0 / (2 * alpha)
        self.oti.mul(sqdist, inv_2alpha, out=tmp1)
        self.oti.sum(1.0, tmp1, out=tmp2)
        self.oti.pow(tmp2, -alpha, out=tmp1)
        self.oti.mul(sigma_f_sq, tmp1, out=tmp2)
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
        ell, pi_over_p, sigma_f_sq = self._cache_sine_exp_params_aniso(
            length_scales)
        tmp1, tmp2, sqdist = self._ensure_temp_arrays(
            differences_by_dim[0].shape)
        self._reset_sqdist()

        for i in range(self.dim):
            self.oti.mul(pi_over_p[i], differences_by_dim[i], out=tmp1)
            self.oti.sin(tmp1, out=tmp2)
            self.oti.mul(ell[i], tmp2, out=tmp1)
            self.oti.mul(tmp1, tmp1, out=tmp2)
            self.oti.sum(sqdist, tmp2, out=sqdist)

        self.oti.mul(-2.0, sqdist, out=tmp1)
        self.oti.exp(tmp1, out=tmp2)
        self.oti.mul(sigma_f_sq, tmp2, out=tmp1)
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

        # Compute scaled distance — regularise r directly (not each diff)
        # so that r.e([d]) = 0 at training-point diagonals, preserving
        # correct OTI derivative structure for the covariance blocks.
        _eps = 1e-10
        sqdist = self.oti.sqrt(
            sum((ell[i] * differences_by_dim[i]) ** 2
                for i in range(self.dim)) + _eps ** 2
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
            val = val * \
                (1 + ell[i] * self.SI_kernel_prebuild(differences_by_dim[i]))
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
        tmp1, tmp2, sqdist = self._ensure_temp_arrays(
            differences_by_dim[0].shape)
        self._reset_sqdist()

        for i in range(self.dim):
            self.oti.mul(ell, differences_by_dim[i], out=tmp1)
            self.oti.mul(tmp1, tmp1, out=tmp2)
            self.oti.sum(sqdist, tmp2, out=sqdist)

        self.oti.exp((-0.5) * sqdist, out=tmp1)
        self.oti.mul(sigma_f_sq, tmp1, out=tmp2)
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
        tmp1, tmp2, sqdist = self._ensure_temp_arrays(
            differences_by_dim[0].shape)
        self._reset_sqdist()

        for i in range(self.dim):
            self.oti.mul(ell, differences_by_dim[i], out=tmp1)
            self.oti.mul(tmp1, tmp1, out=tmp2)
            self.oti.sum(sqdist, tmp2, out=sqdist)

        # (1 + sqdist / (2 * alpha))^(-alpha)
        inv_2alpha = 1.0 / (2 * alpha)
        self.oti.mul(sqdist, inv_2alpha, out=tmp1)
        self.oti.sum(1.0, tmp1, out=tmp2)
        self.oti.pow(tmp2, -alpha, out=tmp1)
        self.oti.mul(sigma_f_sq, tmp1, out=tmp2)
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
        ell, pi_over_p, sigma_f_sq = self._cache_sine_exp_params_iso(
            length_scales)
        tmp1, tmp2, sqdist = self._ensure_temp_arrays(
            differences_by_dim[0].shape)
        self._reset_sqdist()

        for i in range(self.dim):
            self.oti.mul(pi_over_p, differences_by_dim[i], out=tmp1)
            self.oti.sin(tmp1, out=tmp2)
            self.oti.mul(ell, tmp2, out=tmp1)
            self.oti.mul(tmp1, tmp1, out=tmp2)
            self.oti.sum(sqdist, tmp2, out=sqdist)

        self.oti.mul(-2.0, sqdist, out=tmp1)
        self.oti.exp(tmp1, out=tmp2)
        self.oti.mul(sigma_f_sq, tmp2, out=tmp1)
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

        # Compute scaled distance — regularise r directly (not each diff)
        _eps = 1e-10
        sqdist = self.oti.sqrt(
            sum((ell * differences_by_dim[i]) ** 2
                for i in range(self.dim)) + _eps ** 2
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
            val = val * \
                (1 + ell * self.SI_kernel_prebuild(differences_by_dim[i]))
        return sigma_f_sq * val
