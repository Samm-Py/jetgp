Hyperparameter Optimization
============================

Overview
--------
JetGP uses **maximum likelihood estimation (MLE)** to optimize the hyperparameters of Gaussian Process models.
The optimization is performed after model initialization and before making predictions.
All GP modules share a common optimization interface through the ``optimize_hyperparameters()`` method.

JetGP supports multiple optimization algorithms, ranging from gradient-based methods to 
population-based metaheuristics, allowing users to choose the most appropriate approach for their problem.

------------------------------------------------------------

Basic Usage
-----------

After initializing a GP model (DEGP, DDEGP, GDDEGP, or WDEGP), 
hyperparameters are optimized by calling the ``optimize_hyperparameters()`` method:

**Example:**

::

    # Initialize the model
    gp = degp(
        X_train, y_train, n_order, n_bases=1, der_indices=der_indices,
        normalize=True, kernel="SE", kernel_type="anisotropic"
    )
    
    # Optimize hyperparameters
    params = gp.optimize_hyperparameters(
        optimizer='cobyla', 
        n_restart_optimizer=10
    )

The method returns the optimized hyperparameters, which are automatically stored within the model
and used for subsequent predictions.

For model initialization details, see :doc:`initialization`.

------------------------------------------------------------

What is Optimized
-----------------

The hyperparameters optimized depend on the kernel and kernel type selected during model initialization.

**Common hyperparameters:**

- **Signal variance** (:math:`\sigma_f^2`): Controls the overall magnitude of function variation.
  Larger values indicate greater function amplitude.

- **Length scale(s)** (:math:`\ell` or :math:`\ell_j`): Controls how quickly correlation decays with distance.
  Smaller length scales allow the model to capture more rapid variation.
  
  - *Isotropic kernels*: Single :math:`\ell` shared across all input dimensions
  - *Anisotropic kernels*: Separate :math:`\ell_j` for each input dimension, allowing different 
    correlation structures along different axes

- **Noise variance** (:math:`\sigma_n^2`): Models observation noise or measurement uncertainty.
  Set to a small value (or fixed) for noise-free simulations.

**Kernel-specific hyperparameters:**

- **Rational Quadratic (RQ)**: Shape parameter :math:`\alpha` controlling the mixture of length scales
- **Matern**: Smoothness parameter :math:`\nu` (typically fixed via ``smoothness_parameter``)
- **SineExp**: Period parameters :math:`p_j` for periodic structure

**Number of hyperparameters:**

.. list-table:: Hyperparameter count by kernel configuration
   :header-rows: 1
   :widths: 30 35 35

   * - Kernel
     - Isotropic
     - Anisotropic
   * - SE (RBF)
     - 3 (:math:`\sigma_f^2, \ell, \sigma_n^2`)
     - :math:`2 + d` (:math:`\sigma_f^2, \ell_1, \ldots, \ell_d, \sigma_n^2`)
   * - RQ
     - 4 (:math:`\sigma_f^2, \ell, \alpha, \sigma_n^2`)
     - :math:`3 + d`
   * - Matern
     - 3 (ν typically fixed)
     - :math:`2 + d`
   * - SineExp
     - :math:`3 + d` (includes periods)
     - :math:`2 + 2d`

where :math:`d` is the input dimension (``n_bases``).

.. note::
   For **WDEGP** models, hyperparameters are **shared across all submodels**.
   The optimization maximizes the combined log marginal likelihood, ensuring
   consistent covariance structure across the weighted ensemble.

------------------------------------------------------------

Common Arguments
----------------

**optimizer**  
Specifies which optimization algorithm to use for hyperparameter tuning.  
Available options:

- ``"pso"`` – Particle Swarm Optimization
- ``"jade"`` – JADE (Adaptive Differential Evolution)
- ``"lbfgs"`` – Limited-memory BFGS (gradient-based)
- ``"powell"`` – Powell's derivative-free method
- ``"cobyla"`` – Constrained Optimization BY Linear Approximation

**n_restart_optimizer**  
Number of random restarts for the optimization procedure.  
Multiple restarts help avoid local optima by initializing the search from different starting points.

- **For gradient-based and derivative-free methods** (``lbfgs``, ``powell``, ``cobyla``): Each restart 
  begins from a randomly sampled initial hyperparameter configuration (unless ``x0`` is provided for 
  the first restart). The best result across all restarts is returned.
- **For population-based methods** (``pso``, ``jade``): This parameter is **ignored**. These methods 
  perform a single optimization run using their population-based search strategy, which inherently 
  explores multiple regions of the search space.

**Default:** ``10``

**x0** (optional)  
Initial guess for hyperparameters, used only for restart-based methods (``lbfgs``, ``powell``, ``cobyla``).  
If provided, the first optimization run starts from this point, while subsequent restarts use random initialization.  
Population-based methods (``pso``, ``jade``) use ``initial_positions`` instead.

**debug** 
If ``True``, prints intermediate optimization results during execution.  
Useful for monitoring convergence and diagnosing optimization issues.

**Default:** ``False``

------------------------------------------------------------

Available Optimizers
--------------------

**1. pso (Particle Swarm Optimization)**

A population-based metaheuristic that simulates social behavior of particles searching for optima.
Each particle represents a candidate solution that moves through the search space, influenced by 
its own best position and the global best position found by the swarm.

Unlike restart-based methods, PSO performs a **single optimization run** without restarts, 
as the population naturally explores multiple regions simultaneously.

**Recommended for:**  
- High-dimensional hyperparameter spaces
- Non-smooth likelihood surfaces
- Initial exploration when gradient information is unreliable
- Multi-modal optimization landscapes

**Additional keyword arguments:**

- ``pop_size`` – Number of particles in the swarm (default: ``20``)  
  Larger populations improve exploration but increase computational cost
  
- ``n_generations`` – Maximum number of generations/iterations (default: ``50``)  
  More generations allow better convergence to global optimum
  
- ``local_opt_every`` – Frequency of local gradient-based refinement (default: ``15``)  
  Every N generations, applies a gradient-based optimization (L-BFGS-B) to the current global best position.
  This **hybrid approach** combines global exploration from PSO with local exploitation from gradients,
  significantly improving convergence speed and final solution quality. Set to ``None`` or a large number 
  to disable local optimization.
  
- ``initial_positions`` – Optional array of initial particle positions (default: ``None``)  
  Shape: ``(pop_size, n_dims)``. If provided, initializes swarm at specified locations instead of random sampling
  
- ``omega`` – Inertia weight controlling particle momentum (default: ``0.5``)  
  Higher values (0.7-0.9) encourage exploration; lower values (0.4-0.6) encourage exploitation
  
- ``phip`` – Cognitive acceleration coefficient (default: ``0.5``)  
  Controls attraction to particle's personal best position (typical range: 0.5-2.5)
  
- ``phig`` – Social acceleration coefficient (default: ``0.5``)  
  Controls attraction to global best position (typical range: 0.5-2.5)
  
- ``seed`` – Random seed for reproducibility (default: ``42``)

**Note:** The ``n_restart_optimizer`` parameter is ignored for PSO.

**Example:**

::

    params = gp.optimize_hyperparameters(
        optimizer='pso',
        pop_size=30,
        n_generations=100,
        local_opt_every=20,  # Apply gradient refinement every 20 generations
        omega=0.7,
        phip=1.5,
        phig=1.5,
        seed=123,
        debug=True
    )

----

**2. jade (Adaptive Differential Evolution)**

An evolutionary algorithm based on the JADE (Adaptive Differential Evolution with Optional External Archive) 
variant. It adaptively adjusts mutation and crossover parameters during optimization and maintains 
an archive of recently explored solutions to improve diversity.

Like PSO, JADE performs a **single optimization run** without restarts, as the population-based 
search inherently explores multiple regions of the hyperparameter space.

**Recommended for:**  
- Rugged likelihood surfaces with many local optima
- Problems where gradient-based methods fail to converge
- Robust global search requirements
- Complex, non-linear optimization landscapes

**Additional keyword arguments:**

- ``pop_size`` – Number of individuals in the population (default: ``20``)  
  Larger populations improve exploration but increase computational cost
  
- ``n_generations`` – Maximum number of generations (default: ``50``)  
  More generations allow better convergence to global optimum
  
- ``p`` – Proportion of best individuals for mutation (default: ``0.1``)  
  Controls the greediness of the mutation strategy (typical range: 0.05-0.2). 
  Lower values make the search more exploitative; higher values increase exploration.
  
- ``c`` – Learning rate for parameter adaptation (default: ``0.1``)  
  Controls how quickly F and CR parameters adapt (typical range: 0.05-0.2).
  Higher values lead to faster adaptation but may be less stable.
  
- ``local_opt_every`` – Frequency of local gradient-based refinement (default: ``15``)  
  Every N generations, applies a gradient-based optimization (L-BFGS-B) to the current best individual.
  This **hybrid strategy** combines JADE's robust global search with gradient-based local refinement,
  improving both convergence speed and solution quality. Set to ``None`` or a large number to disable.
  
- ``initial_positions`` – Optional array of initial population positions (default: ``None``)  
  Shape: ``(pop_size, n_dims)``. If provided, initializes population at specified locations
  
- ``seed`` – Random seed for reproducibility (default: ``42``)

**Note:** The ``n_restart_optimizer`` parameter is ignored for JADE.

**Example:**

::

    params = gp.optimize_hyperparameters(
        optimizer='jade',
        pop_size=40,
        n_generations=150,
        p=0.15,
        c=0.1,
        local_opt_every=25,  # Apply gradient refinement every 25 generations
        seed=456,
        debug=True
    )

----

**3. lbfgs (Limited-memory BFGS)**

A gradient-based quasi-Newton method that approximates the inverse Hessian matrix using 
limited memory. Extremely efficient for smooth objective functions with reliable gradients.

This optimizer uses a **multi-restart strategy** to explore different regions of the hyperparameter space
and avoid local optima.

**Recommended for:**  
- Low to moderate dimensional problems (typically < 20 hyperparameters)
- Smooth likelihood surfaces
- Fast convergence when good initial guesses are available
- Production use when gradients are reliable

**Additional keyword arguments:**

- ``x0`` – Initial guess for first restart (default: ``None``)  
  If not provided, uses random initialization within bounds
  
- ``n_restart_optimizer`` – Number of random restarts (default: ``10``)  
  More restarts improve robustness against local optima
  
- ``maxiter`` – Maximum number of iterations per restart (default: ``100``)
  
- ``ftol`` – Function tolerance for convergence (default: ``1e-8``)  
  Optimization stops when ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``
  
- ``gtol`` – Gradient tolerance for convergence (default: ``1e-8``)  
  Optimization stops when ``max{|proj g_i|} <= gtol``

- ``disp`` – Display convergence messages (default: ``False``)
  
- ``debug`` – Print intermediate results (default: ``False``)

**Example:**

::

    params = gp.optimize_hyperparameters(
        optimizer='lbfgs',
        n_restart_optimizer=15,
        maxiter=200,
        ftol=1e-9,
        gtol=1e-9,
        debug=True
    )

----

**4. powell (Powell's Method)**

A derivative-free optimization algorithm that performs sequential line searches along 
conjugate directions. Does not require gradient information but can still achieve 
fast convergence on smooth problems.

This optimizer uses a **multi-restart strategy** to explore different initial conditions
and improve robustness.

**Recommended for:**  
- Problems where gradient computation is expensive or unavailable
- Moderate-dimensional spaces (typically < 15 hyperparameters)
- Situations where analytical derivatives are unreliable
- Smooth but noisy objective functions

**Additional keyword arguments:**

- ``x0`` – Initial guess for first restart (default: ``None``)
  
- ``n_restart_optimizer`` – Number of random restarts (default: ``10``)  
  More restarts improve robustness against local optima
  
- ``maxiter`` – Maximum number of iterations per restart (default: SciPy default)
  
- ``xtol`` – Relative tolerance for parameter convergence (default: SciPy default)  
  Absolute error in solution acceptable for convergence
  
- ``ftol`` – Relative tolerance for function convergence (default: SciPy default)  
  Relative error in objective function acceptable for convergence
  
- ``disp`` – Display convergence messages (default: ``False``)
  
- ``debug`` – Print intermediate results (default: ``False``)

**Example:**

::

    params = gp.optimize_hyperparameters(
        optimizer='powell',
        n_restart_optimizer=20,
        maxiter=500,
        xtol=1e-6,
        ftol=1e-6,
        debug=True
    )

----

**5. cobyla (Constrained Optimization BY Linear Approximation)**

A derivative-free method designed for constrained optimization problems.
Constructs linear approximations of the objective function and handles bounds 
through inequality constraints.

This optimizer uses a **multi-restart strategy** to explore different starting points
and avoid local optima.

**Recommended for:**  
- Problems with bound constraints on hyperparameters
- When gradients are unavailable or unreliable
- Moderate-dimensional optimization (typically < 15 hyperparameters)
- Noisy or discontinuous objective functions

**Additional keyword arguments:**

- ``x0`` – Initial guess for first restart (default: ``None``)
  
- ``n_restart_optimizer`` – Number of random restarts (default: ``10``)  
  More restarts improve robustness against local optima
  
- ``maxiter`` – Maximum number of function evaluations per restart (default: SciPy default)
  
- ``rhobeg`` – Initial trust region radius (default: SciPy default, typically 1.0)  
  Controls the initial step size
  
- ``catol`` – Absolute tolerance for constraint violations (default: SciPy default)
  
- ``f_target`` – Target function value; stops if reached (default: SciPy default)
  
- ``disp`` – Display convergence messages (default: ``False``)
  
- ``debug`` – Print intermediate results (default: ``False``)

**Example:**

::

    params = gp.optimize_hyperparameters(
        optimizer='cobyla',
        n_restart_optimizer=20,
        maxiter=500,
        rhobeg=0.5,
        catol=1e-6,
        debug=True
    )

------------------------------------------------------------

Diagnosing Optimization Quality
-------------------------------

Successful hyperparameter optimization is critical for accurate GP predictions.
Here are signs to look for when evaluating optimization results.

**Signs of successful optimization:**

- Consistent log-likelihood values across multiple restarts (for restart-based methods)
- Log-likelihood stabilizes before reaching maximum iterations (for population-based methods)
- Predictions show reasonable uncertainty (not too wide or too narrow)
- Length scales are within a reasonable range relative to the data spread
- Hyperparameters do not hit bounds

**Signs of potential issues:**

- Very different log-likelihood values across restarts (indicates multiple local optima)
- Hyperparameters frequently hitting bounds (may need to adjust bound settings)
- Very large length scales (:math:`\ell \gg` data range): Model is underfitting, treating data as nearly constant
- Very small length scales (:math:`\ell \ll` typical point spacing): Model may be overfitting or capturing noise
- Very small noise variance with noisy data: Model may be interpolating noise
- Optimization fails to converge within iteration limits

**Troubleshooting strategies:**

1. **Multiple local optima:** Increase ``n_restart_optimizer`` for restart-based methods,
   or switch to population-based methods (``pso``, ``jade``)

2. **Slow convergence:** For population-based methods, increase ``n_generations`` or ``pop_size``.
   Enable hybrid refinement with ``local_opt_every=15-25``

3. **Hitting bounds:** Check if data is properly normalized (``normalize=True``).
   Consider if the kernel choice is appropriate for the problem

4. **Unreasonable length scales:** Verify data scaling and consider using ``normalize=True``.
   Check if the number of training points is sufficient

5. **Poor predictions despite convergence:** Try a different kernel or kernel type.
   Consider whether derivative information is helping or introducing numerical issues

------------------------------------------------------------

Notes
-----

- All optimizers maximize the **log marginal likelihood** (or equivalently minimize its negative).
- Hyperparameter bounds are typically set automatically based on the data scale and problem characteristics.
- The optimization is performed in the normalized space when ``normalize=True``, 
  with hyperparameters automatically transformed back to the original scale.
- **Gradient-based methods** (``lbfgs``): 
  - Use gradients for efficient convergence on smooth surfaces
  - Use multiple restarts controlled by ``n_restart_optimizer``
  - First restart uses ``x0`` if provided; subsequent restarts use random initialization
- **Derivative-free methods** (``powell``, ``cobyla``): 
  - Do not require gradient information
  - Use multiple restarts controlled by ``n_restart_optimizer``
  - First restart uses ``x0`` if provided; subsequent restarts use random initialization
- **Population-based methods** (``pso``, ``jade``): 
  - Perform a single run; ``n_restart_optimizer`` is ignored
  - Use ``initial_positions`` instead of ``x0`` to specify starting population
  - The ``local_opt_every`` parameter enables hybrid optimization by periodically applying gradient-based refinement

------------------------------------------------------------

Summary
-------

JetGP provides a flexible hyperparameter optimization framework supporting both 
gradient-based and gradient-free methods. The unified interface allows easy experimentation 
with different optimizers while maintaining consistent model behavior.

**Quick reference:**

- **Fastest, smooth problems:** ``lbfgs`` with 10-20 restarts
- **No gradients, moderate complexity:** ``powell`` or ``cobyla`` with 15-25 restarts
- **Multi-modal, global search:** ``pso`` or ``jade`` with hybrid refinement (``local_opt_every=15-25``)
- **Highly non-smooth problems:** ``pso`` or ``jade`` without gradient refinement (``local_opt_every=None``)
- **Default recommendation:** Start with ``lbfgs`` (n_restart_optimizer=10-20); fall back to ``pso`` with ``local_opt_every=15`` if needed

**Key distinctions:**

- Gradient-based methods (``lbfgs``) use gradients for efficient local convergence
- Derivative-free methods (``powell``, ``cobyla``) work without gradient information
- Both gradient-based and derivative-free methods use **multiple restarts** to explore different starting points
- Population-based methods (``pso``, ``jade``) use **single runs** with inherent parallel exploration
- The ``local_opt_every`` parameter in PSO and JADE enables **hybrid optimization**, 
  combining global exploration with gradient-based local refinement for improved performance

For most applications, starting with ``lbfgs`` and falling back to hybrid ``pso`` or ``jade`` 
for difficult cases provides a robust and efficient optimization strategy.

For making predictions with optimized parameters, see :doc:`predictions`.