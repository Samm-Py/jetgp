JetGP Modules Overview
======================

Overview
--------
JetGP provides a unified framework for constructing **derivative-enhanced Gaussian Process (GP)** models.  
Each module corresponds to a distinct GP formulation, differing in how function and derivative information are incorporated.  
The core model families include:

- **DEGP** – Derivative-Enhanced Gaussian Process  
- **DDEGP** – Directional Derivative-Enhanced Gaussian Process  
- **GDDEGP** – Generalized Directional Derivative-Enhanced Gaussian Process  
- **WDEGP** – Weighted Derivative-Enhanced Gaussian Process (supports DEGP, DDEGP, or GDDEGP submodels)

All these models share a common structure and initialization interface.  
This page provides an overview of their usage and explains the shared arguments that appear in all implementations.

For detailed initialization instructions, see :doc:`initialization`.

------------------------------------------------------------

Available Modules
-----------------

**1. full_degp**

- **File:** ``full_degp/degp.py``  
- **Class:** ``degp``  
- **Description:**  
  Implements a standard Derivative-Enhanced Gaussian Process (DEGP) where function values and their derivatives  
  (up to arbitrary order) are jointly modeled using **coordinate-aligned partial derivatives** 
  (e.g., :math:`\partial f/\partial x_1`, :math:`\partial f/\partial x_2`).

**2. full_ddegp**

- **File:** ``full_ddegp/ddegp.py``  
- **Class:** ``ddegp``  
- **Description:**  
  Constructs a **Directional Derivative-Enhanced Gaussian Process (DDEGP)** using directional derivative information  
  evaluated along **global directions**.  
  This means that all training points share the same derivative directions, allowing the model to capture variation  
  along specific, predefined axes of sensitivity.  
  Such an approach is well suited for problems where global sensitivity along known direction(s) is of interest.

  **Additional initialization parameter:**
  
  - ``rays`` – A 2D array of shape ``(d, n_directions)`` specifying the global directional vectors.  
    Each column is a direction vector shared by all training points.

**3. full_gddegp**

- **File:** ``full_gddegp/gddegp.py``  
- **Class:** ``gddegp``  
- **Description:**  
  Implements the **Generalized Directional Derivative-Enhanced Gaussian Process (GDDEGP)**, an extension of DDEGP  
  that allows **distinct directional derivative information at each training point**.  
  In contrast to DDEGP, the directional vectors need not be the same globally, enabling the model to represent  
  spatially varying directional sensitivities.  
  
  A key advantage of GDDEGP is that predictions can be made along **any direction**—not just those used  
  during training—as long as the derivative order was included in training.

  **Additional initialization parameter:**
  
  - ``rays_list`` – A list of 2D arrays, one per direction index.  
    For direction ``i``, ``rays_list[i]`` has shape ``(d, n_points_with_direction_i)``,  
    where column ``j`` corresponds to the point at ``derivative_locations[i][j]``.

  **Additional prediction parameter:**
  
  - ``rays_predict`` – Required when ``return_deriv=True``. A list of 2D arrays specifying  
    the directional vectors at each prediction point. These can be **any directions**, not  
    limited to those used in training.

**4. wdegp**

- **File:** ``wdegp/wdegp.py``  
- **Class:** ``wdegp``  
- **Description:**  
  Implements the **Weighted Derivative-Enhanced Gaussian Process (WDEGP)** framework, inspired by the  
  *Weighted Gradient-Enhanced Kriging (WGEK)* method introduced by Han *et al.* :cite:`HanWeightedGEK`  
  and extended in :cite:`improved_gek`.  

  In this formulation, the training data are divided into smaller subsets to build multiple **submodels**,  
  each trained on a reduced set of derivatives. Predictions from all submodels are then  
  **combined through weighted summation**. This approach preserves the accuracy advantages of 
  derivative-enhanced models while **reducing the computational cost** associated with large matrix inversions.

  WDEGP is a **unified framework** that supports three submodel types via the ``submodel_type`` parameter:
  
  - ``'degp'`` – Submodels use coordinate-aligned partial derivatives (default)
  - ``'ddegp'`` – Submodels use global directional derivatives (requires ``rays``)
  - ``'gddegp'`` – Submodels use point-wise directional derivatives (requires ``rays_list``)

  See :ref:`wdegp_specific` for detailed documentation of the WDEGP interface.

------------------------------------------------------------

Module Comparison
-----------------

.. list-table:: Comparison of JetGP modules
   :header-rows: 1
   :widths: 15 20 20 45

   * - Module
     - Derivative Type
     - Ray Specification
     - Use Case
   * - DEGP
     - Partial
     - N/A
     - Standard derivative data with coordinate-aligned derivatives
   * - DDEGP
     - Directional
     - Global ``rays``
     - Fixed directions across all training points
   * - GDDEGP
     - Directional
     - Point-wise ``rays_list``
     - Spatially-varying directions (e.g., gradient-aligned)
   * - WDEGP
     - Any
     - Depends on ``submodel_type``
     - Partitioned training data for computational efficiency

------------------------------------------------------------

.. _common_arguments:

Common Arguments
----------------

**n_bases**  
Specifies the **dimensionality of the input space** (i.e., the number of input variables).  
For a function :math:`f(x_1, x_2, \ldots, x_d)`, set ``n_bases = d``.

Example::

  # For a 2D function f(x, y)
  n_bases = 2
  
  # For a 1D function f(x)
  n_bases = 1

**n_order**  
Specifies the **maximum derivative order** included in the model.  
This determines the highest-order derivatives that can be used for training.

Example::

  # First-order derivatives only (∂f/∂x, ∂f/∂y)
  n_order = 1
  
  # Up to second-order derivatives (∂f/∂x, ∂²f/∂x², etc.)
  n_order = 2

.. note::
   The ``n_order`` parameter should be consistent with the derivatives specified in ``der_indices``.  
   When using the Matern kernel, ensure that ``smoothness_parameter >= 2 * n_order``  
   so the kernel is sufficiently smooth to support the highest-order derivatives.

**der_indices**  
Specifies which derivatives are included in the model.  
This argument is a **nested list**, where each sublist contains all derivative components of a particular order.  
Each derivative component is itself a list specifying the variable indices and derivative order.  

Examples:

**1D function**
::
  
  der_indices = [[[[1, 1]]]]             # first-order derivative only
  der_indices = [[[[1, 1]]], [[[1, 2]]]] # first- and second-order derivatives

**2D function** – all derivatives up to second order
::
  
  der_indices = [
      [ [[1, 1]], [[2, 1]] ],                      # first-order derivatives
      [ [[1, 2]], [[1, 1], [2, 1]], [[2, 2]] ]    # second-order derivatives
  ]

**2D function** – all derivatives up to third order
::
  
  der_indices = [
      [ [[1, 1]], [[2, 1]] ],                                        # first-order
      [ [[1, 2]], [[1, 1], [2, 1]], [[2, 2]] ],                     # second-order
      [ [[1, 3]], [[1, 2], [2, 1]], [[1, 1], [2, 2]], [[2, 3]] ]    # third-order
  ]

**derivative_locations**  
Specifies which training points have each derivative defined in ``der_indices``.  
This argument is a **list of lists**, where each sublist contains the training point indices for the corresponding derivative.  
The structure must match ``der_indices`` exactly—one entry per derivative.  
Indices can be **non-contiguous** (e.g., ``[0, 2, 5, 7]`` is valid).

Examples:

**1D function** – first-order derivative at points 2, 3, 4, 5
::

  der_indices = [[[[1, 1]]]]
  derivative_locations = [[2, 3, 4, 5]]  # 1 derivative → 1 entry

**1D function** – first and second-order derivatives at different locations
::

  der_indices = [[[[1, 1]]], [[[1, 2]]]]
  derivative_locations = [
      [0, 1, 2, 3, 4, 5],  # df/dx at all 6 points
      [2, 3, 4]            # d²f/dx² at middle 3 points only
  ]

**2D function** – all first-order derivatives at same locations
::

  der_indices = [[ [[1, 1]], [[2, 1]] ]]
  derivative_locations = [
      [0, 1, 2, 3, 4],  # ∂f/∂x₁ at these points
      [0, 1, 2, 3, 4]   # ∂f/∂x₂ at these points
  ]

**2D function** – first and second-order with different coverage
::

  der_indices = [
      [ [[1, 1]], [[2, 1]] ],                      # 2 first-order derivatives
      [ [[1, 2]], [[1, 1], [2, 1]], [[2, 2]] ]    # 3 second-order derivatives
  ]
  derivative_locations = [
      [0, 1, 2, 3, 4, 5, 6, 7, 8],  # ∂f/∂x₁ at all 9 points
      [0, 1, 2, 3, 4, 5, 6, 7, 8],  # ∂f/∂x₂ at all 9 points
      [4, 5, 7, 8],                  # ∂²f/∂x₁² at interior only
      [4, 5, 7, 8],                  # ∂²f/∂x₁∂x₂ at interior only
      [4, 5, 7, 8]                   # ∂²f/∂x₂² at interior only
  ]

**Non-contiguous indices** – derivatives at alternating points
::

  der_indices = [[[[1, 1]]], [[[1, 2]]]]
  derivative_locations = [
      [0, 2, 4, 6, 8],  # df/dx at even indices
      [1, 3, 5, 7, 9]   # d²f/dx² at odd indices
  ]

**y_train**  
Contains the training observations: function values and derivative values.  
This argument is a **list of arrays**, where:

- The **first entry** contains function values at all training points, shape ``(n_train, 1)``
- **Subsequent entries** contain derivative values, one array per derivative in ``der_indices``
- Each derivative array has shape ``(n_points_with_derivative, 1)``

The structure must match ``der_indices`` and ``derivative_locations`` exactly.

Examples:

**1D function** – function values + first derivative at subset of points
::

  # 6 training points, derivative at points [2, 3, 4, 5]
  der_indices = [[[[1, 1]]]]
  derivative_locations = [[2, 3, 4, 5]]
  
  y_train = [
      y_func.reshape(-1, 1),                           # shape (6, 1) - function at all points
      y_deriv[derivative_locations[0]].reshape(-1, 1)  # shape (4, 1) - df/dx at 4 points
  ]

**1D function** – first and second derivatives at different locations
::

  # 6 training points
  der_indices = [[[[1, 1]]], [[[1, 2]]]]
  derivative_locations = [
      [0, 1, 2, 3, 4, 5],  # df/dx at all 6 points
      [2, 3, 4]            # d²f/dx² at 3 points
  ]
  
  y_train = [
      y_func.reshape(-1, 1),                               # shape (6, 1)
      y_deriv_1[derivative_locations[0]].reshape(-1, 1),   # shape (6, 1)
      y_deriv_2[derivative_locations[1]].reshape(-1, 1)    # shape (3, 1)
  ]

**2D function** – first-order derivatives at all points
::

  # 9 training points on 3×3 grid
  der_indices = [[ [[1, 1]], [[2, 1]] ]]
  derivative_locations = [
      [0, 1, 2, 3, 4, 5, 6, 7, 8],  # ∂f/∂x₁
      [0, 1, 2, 3, 4, 5, 6, 7, 8]   # ∂f/∂x₂
  ]
  
  y_train = [
      y_func.reshape(-1, 1),      # shape (9, 1) - function values
      y_deriv_x1.reshape(-1, 1),  # shape (9, 1) - ∂f/∂x₁
      y_deriv_x2.reshape(-1, 1)   # shape (9, 1) - ∂f/∂x₂
  ]

**2D function** – mixed derivative coverage
::

  # 9 training points, second-order only at interior (points 4, 5, 7, 8)
  der_indices = [
      [ [[1, 1]], [[2, 1]] ],                      # first-order
      [ [[1, 2]], [[1, 1], [2, 1]], [[2, 2]] ]    # second-order
  ]
  derivative_locations = [
      [0, 1, 2, 3, 4, 5, 6, 7, 8],  # ∂f/∂x₁ at all
      [0, 1, 2, 3, 4, 5, 6, 7, 8],  # ∂f/∂x₂ at all
      [4, 5, 7, 8],                  # ∂²f/∂x₁²
      [4, 5, 7, 8],                  # ∂²f/∂x₁∂x₂
      [4, 5, 7, 8]                   # ∂²f/∂x₂²
  ]
  
  interior_pts = [4, 5, 7, 8]
  y_train = [
      y_func.reshape(-1, 1),                          # shape (9, 1)
      y_deriv_x1.reshape(-1, 1),                      # shape (9, 1)
      y_deriv_x2.reshape(-1, 1),                      # shape (9, 1)
      y_deriv_x1x1[interior_pts].reshape(-1, 1),      # shape (4, 1)
      y_deriv_x1x2[interior_pts].reshape(-1, 1),      # shape (4, 1)
      y_deriv_x2x2[interior_pts].reshape(-1, 1)       # shape (4, 1)
  ]


**normalize**  
Controls whether training inputs and outputs are **normalized** before model fitting.  
Normalization significantly improves **numerical stability** in covariance matrix computation and  
**conditioning** during hyperparameter optimization, especially when derivative observations are included.  

**If ``True``**, the following scaling is applied:

- **Inputs:**  
  Each input dimension :math:`x_j` is standardized to zero mean and unit variance:

  .. math::
     x'_j = \frac{x_j - \mu_{x,j}}{\sigma_{x,j}}

- **Outputs:**  
  Function values are standardized as

  .. math::
     y' = \frac{y - \mu_y}{\sigma_y}

- **Derivatives:**  
  Derivative observations are rescaled according to the chain rule,  
  ensuring consistent scaling between function and derivative values.  

  For the first derivative :math:`\frac{\partial y}{\partial x_j}`:

  .. math::
     \left(\frac{\partial y'}{\partial x'_j}\right)
     = \frac{\sigma_{x,j}}{\sigma_y}
     \left(\frac{\partial y}{\partial x_j}\right)

  For higher-order derivatives of total order :math:`m`:

  .. math::
     \left(\frac{\partial^m y'}{\partial x'_{j_1}\cdots \partial x'_{j_m}}\right)
     = \frac{\sigma_{x,j_1}\sigma_{x,j_2}\cdots\sigma_{x,j_m}}{\sigma_y}
     \left(\frac{\partial^m y}{\partial x_{j_1}\cdots \partial x_{j_m}}\right)

These transformations preserve the covariance structure between function values and derivatives,  
while keeping all data components on comparable scales.  

**If ``False``**, raw input and output values are used directly,  
which may lead to ill-conditioned covariance matrices if feature magnitudes vary widely.

**Directional Derivatives:**

When **directional derivatives** are used (e.g., derivatives evaluated along
specific direction vectors or rays), normalization involves a preliminary
step to ensure consistency between the original and normalized input spaces.


As above, inputs and outputs are normalized according to the following:

.. math::

   x'_j = \frac{x_j - \mu_{x,j}}{\sigma_{x,j}}, \qquad
   y' = \frac{y - \mu_y}{\sigma_y},

where :math:`\mu_{x,j}, \sigma_{x,j}` are the mean and standard deviation of
the inputs, and :math:`\mu_y, \sigma_y` are those of the output.

Directional vectors are scaled accordingly:

.. math::

   v'_j = \frac{v_j}{\sigma_{x,j}},

so that directional information is preserved in normalized coordinates.
Using the **chain rule**, the derivative with respect to normalized inputs is

.. math::

   \frac{\partial y'}{\partial x'_j} = 
   \frac{\partial y'}{\partial y} \frac{\partial y}{\partial x_j} \frac{\partial x_j}{\partial x'_j} 
   = \frac{1}{\sigma_y} \frac{\partial y}{\partial x_j} \sigma_{x,j}.

Multiplying by the normalized direction vector :math:`\mathbf{v}'` gives

.. math::

   \frac{\partial y'}{\partial \mathbf{v}'} 
   = \sum_{j=1}^{d} v'_j \frac{\partial y'}{\partial x'_j} 
   = \sum_{j=1}^{d} \frac{v_j}{\sigma_{x,j}} \frac{\sigma_{x,j}}{\sigma_y} \frac{\partial y}{\partial x_j} 
   = \frac{1}{\sigma_y} \sum_{j=1}^{d} v_j \frac{\partial y}{\partial x_j}.

The input scaling factors :math:`\sigma_{x,j}` cancel, leaving only a global
scaling by :math:`1/\sigma_y`. Therefore, the normalized directional derivative becomes:

.. math::

   \frac{\partial y'}{\partial \mathbf{v}'} = \frac{1}{\sigma_y} 
   \frac{\partial y}{\partial \mathbf{v}}.

Higher-Order Directional Derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same logic extends to second-order or higher derivatives:

.. math::

   \frac{\partial^n y'}{\partial \mathbf{v}'^n} 
   = \frac{1}{\sigma_y} 
     \sum_{j_1,\ldots,j_n} v_{j_1}\sigma_{x,j_1} \cdots v_{j_n}\sigma_{x,j_m} 
     \frac{\partial^n y}{\partial x_{j_1}\cdots\partial x_{j_n}}.

and the normalized higher order directional derivative becomes:

.. math::

   \frac{\partial^n y'}{\partial \mathbf{v}'^n} = \frac{1}{\sigma_y} 
   \frac{\partial^n y}{\partial \mathbf{v}^n}.

Notes
~~~~~

- Normalization statistics (means and standard deviations) are stored internally  
  and automatically applied when evaluating predictions on new test points.  
- When predictions are returned, JetGP automatically rescales them to the original physical units.

.. _kernel_argument:

**kernel**  
Specifies the kernel function used in the GP covariance structure.  
Available kernels include:

- ``"SE"`` – Squared Exponential (RBF)  
- ``"RQ"`` – Rational Quadratic  
- ``"Matern"`` – Matern family of kernels  
- ``"SineExp"`` – Sine-Exponential kernel  

The choice of kernel affects smoothness, differentiability, and correlation structure of the function being modeled.

**kernel_type**  
Determines whether the kernel is **isotropic** or **anisotropic**:

- ``"isotropic"`` → a single length scale :math:`\ell` shared across all input dimensions.  
- ``"anisotropic"`` → each input dimension :math:`j` has its own length scale :math:`\ell_j`.  
  Covariance is computed dimension-wise, allowing directional dependencies and varying smoothness.

**Kernel Formulas (Isotropic vs Anisotropic)**

.. list-table:: Kernel formulas for isotropic and anisotropic types
   :header-rows: 1
   :widths: 20 40 40

   * - Kernel
     - Isotropic
     - Anisotropic
   * - SE (RBF)
     - .. math:: k(x, x') = \sigma_f^2 \exp\Big(-\frac{\|x-x'\|^2}{2 \ell^2}\Big)
     - .. math:: k(x, x') = \sigma_f^2 \exp\Big(-\frac{1}{2} \sum_{j=1}^d \frac{(x_j - x'_j)^2}{\ell_j^2}\Big)
   * - RQ
     - .. math:: k(x, x') = \sigma_f^2 \Big(1 + \frac{\|x-x'\|^2}{2 \alpha \ell^2} \Big)^{-\alpha}
     - .. math:: k(x, x') = \sigma_f^2 \Big(1 + \frac{1}{2 \alpha} \sum_{j=1}^d \frac{(x_j - x'_j)^2}{\ell_j^2} \Big)^{-\alpha}
   * - Matern
     - .. math:: r = \frac{\|x-x'\|}{\ell}, \quad k(x,x') = \frac{2^{1-\nu}}{\Gamma(\nu)} (\sqrt{2\nu} r)^\nu K_\nu(\sqrt{2\nu} r)
     - .. math:: r = \sqrt{\sum_{j=1}^d \frac{(x_j - x'_j)^2}{\ell_j^2}}, \quad k(x,x') = \frac{2^{1-\nu}}{\Gamma(\nu)} (\sqrt{2\nu} r)^\nu K_\nu(\sqrt{2\nu} r)
   * - SineExp
     - .. math:: k(x, x') = \sigma_f^2 \exp\Big(-2 \frac{\sum_{j=1}^d \sin^2(\pi (x_j - x'_j) / p_j)}{\ell^2}\Big)
     - .. math:: k(x, x') = \sigma_f^2 \exp\Big(-2 \sum_{j=1}^d \frac{\sin^2(\pi (x_j - x'_j) / p_j)}{\ell_j^2}\Big)


**smoothness_parameter**  
Specifies the smoothness of the covariance function, primarily used for the **Matern** kernel.  
This parameter controls the differentiability of functions sampled from the Gaussian Process.

- Denoted as :math:`\nu = \text{smoothness\_parameter} + 0.5`.  
- Only **integer values** of ``smoothness_parameter`` are supported.  
- This parameter **must be provided** when ``kernel="Matern"``.  

**Interpretation of integer smoothness values:**

.. list-table:: Matern kernel smoothness interpretation
   :header-rows: 1
   :widths: 20 20 20

   * - smoothness_parameter
     - ν
     - Differentiability
   * - 0
     - 0.5
     - Continuous, non-differentiable
   * - 1
     - 1.5
     - Once differentiable
   * - 2
     - 2.5
     - Twice differentiable
   * - 3
     - 3.5
     - Three times differentiable

.. note::  
  - For kernels other than Matern, this parameter is ignored.  
  - When using **derivative-enhanced GPs**, it is recommended that  
    ``smoothness_parameter >= 2 * n_order``  
    to ensure that the kernel is sufficiently smooth to support the highest-order derivatives.

------------------------------------------------------------

.. _prediction_interface:

Prediction Interface
--------------------

All JetGP models share a common prediction interface via the ``predict()`` method.

**Basic Usage**
::

  y_pred = model.predict(X_test, params)

**Full Signature**
::

  y_pred, y_cov = model.predict(
      X_test, params,
      calc_cov=True,
      return_deriv=False,
      rays_predict=None  # GDDEGP only
  )

**Parameters**

``X_test``
  Test points at which to make predictions. Shape ``(n_test, d)``.

``params``
  Optimized hyperparameters returned by ``optimize_hyperparameters()``.

``calc_cov`` *(bool, default: True)*
  If ``True``, return the predictive variance/covariance.  
  If ``False``, only return the mean prediction (faster).

``return_deriv`` *(bool, default: False)*
  If ``True``, return predictions for both function values and derivatives.  
  The output shape becomes ``(n_outputs, n_test)`` where ``n_outputs = 1 + n_derivatives``.
  
  - Row 0: function value predictions
  - Rows 1, 2, ...: derivative predictions in the order specified by ``der_indices``

``rays_predict`` *(list of arrays, GDDEGP only)*
  Required for GDDEGP when ``return_deriv=True``.  
  A list of 2D arrays specifying directional vectors at each test point.  
  Structure: ``rays_predict[i]`` has shape ``(d, n_test)`` for direction ``i``.
  
  **Note:** For GDDEGP, you can specify **any directions** in ``rays_predict``—not limited  
  to the directions used during training. The only restriction is derivative **order**.

**Return Values**

- If ``calc_cov=False``: Returns ``y_pred`` only
- If ``calc_cov=True``: Returns ``(y_pred, y_cov)``

When ``return_deriv=False``:
  - ``y_pred``: shape ``(n_test,)`` – function value predictions
  - ``y_cov``: shape ``(n_test,)`` – predictive variances

When ``return_deriv=True``:
  - ``y_pred``: shape ``(n_outputs, n_test)`` – function and derivative predictions
  - ``y_cov``: shape ``(n_outputs, n_test)`` – predictive variances for each output

**Examples**

Basic prediction (function values only)::

  y_pred = model.predict(X_test, params, calc_cov=False)

Prediction with variance::

  y_pred, y_var = model.predict(X_test, params, calc_cov=True)

Prediction with derivatives (DEGP/DDEGP)::

  y_pred_full = model.predict(X_test, params, return_deriv=True, calc_cov=False)
  y_func = y_pred_full[0, :]      # Function values
  y_deriv_1 = y_pred_full[1, :]   # First derivative
  y_deriv_2 = y_pred_full[2, :]   # Second derivative (if included)

Prediction with derivatives (GDDEGP)::

  # Can use ANY directions at test points (not limited to training directions)
  rays_predict = [rays_dir1_test, rays_dir2_test]  # Each shape (d, n_test)
  
  y_pred_full = model.predict(
      X_test, params,
      rays_predict=rays_predict,
      return_deriv=True,
      calc_cov=False
  )

For detailed information on model-specific prediction behavior, derivative restrictions,  
and troubleshooting, see :doc:`predictions`.

------------------------------------------------------------

.. _wdegp_specific:

WDEGP-Specific Interface
------------------------

WDEGP extends the base interface with additional parameters for submodel configuration.

**Initialization**
::

  model = wdegp(
      X_train, y_train, n_order, n_bases,
      derivative_locations=derivative_locations,
      der_indices=der_indices,
      submodel_type='degp',  # 'degp', 'ddegp', or 'gddegp'
      rays=None,             # Required if submodel_type='ddegp'
      rays_list=None,        # Required if submodel_type='gddegp'
      normalize=True,
      kernel="SE",
      kernel_type="anisotropic"
  )

**WDEGP-Specific Parameters**

``submodel_type`` *(str, default: 'degp')*
  Specifies the type of GP model used for each submodel:
  
  - ``'degp'`` – Coordinate-aligned partial derivatives
  - ``'ddegp'`` – Global directional derivatives (requires ``rays``)
  - ``'gddegp'`` – Point-wise directional derivatives (requires ``rays_list``)

``rays`` *(ndarray, required if submodel_type='ddegp')*
  Global directional vectors shared by all submodels.  
  Shape: ``(d, n_directions)``

``rays_list`` *(list of lists, required if submodel_type='gddegp')*
  Point-wise directional vectors for each submodel.  
  Structure: ``rays_list[submodel_idx][direction_idx]`` has shape ``(d, n_points_in_submodel_with_direction)``.

**Data Structure for WDEGP**

In WDEGP, data is organized by **submodel**. Each list has one entry per submodel:

``y_train`` – List of lists::

  y_train = [
      [y_vals, dy1_sm1, dy2_sm1, ...],  # Submodel 1 data
      [y_vals, dy1_sm2, dy2_sm2, ...],  # Submodel 2 data
      ...
  ]

- The function values ``y_vals`` (shape ``(n_train, 1)``) are **shared** across all submodels
- Derivative arrays have shapes matching the points in each submodel's ``derivative_locations``

``der_indices`` – List of derivative specifications per submodel::

  der_indices = [
      [[[[1, 1]]], [[[1, 2]]]],  # Submodel 1: 1st and 2nd order
      [[[[1, 1]]], [[[1, 2]]]]   # Submodel 2: 1st and 2nd order
  ]

``derivative_locations`` – List of location lists per submodel::

  derivative_locations = [
      [sm1_deriv1_locs, sm1_deriv2_locs],  # Submodel 1 locations
      [sm2_deriv1_locs, sm2_deriv2_locs]   # Submodel 2 locations
  ]

.. warning::
   **Derivative locations must be DISJOINT across submodels.**
   
   Each training point's derivatives can only belong to **one** submodel.
   Overlapping derivative locations will cause incorrect covariance computations.
   
   Valid::
   
     derivative_locations = [
         [[0, 2, 4], [0, 2, 4]],  # Submodel 1: even indices
         [[1, 3, 5], [1, 3, 5]]   # Submodel 2: odd indices (disjoint!)
     ]
   
   Invalid::
   
     derivative_locations = [
         [[0, 1, 2], [0, 1, 2]],  # Submodel 1
         [[2, 3, 4], [2, 3, 4]]   # Submodel 2: index 2 overlaps!
     ]

**WDEGP Prediction**

WDEGP prediction supports an additional parameter to return submodel predictions:
::

  y_pred, y_cov, submodel_preds, submodel_covs = model.predict(
      X_test, params,
      calc_cov=True,
      return_submodels=True
  )

``return_submodels`` *(bool, default: False)*
  If ``True``, return individual submodel predictions in addition to the weighted combination.

**Return Values with return_submodels=True:**

- ``y_pred``: Combined (weighted) prediction, shape ``(n_test,)``
- ``y_cov``: Combined predictive variance, shape ``(n_test,)``
- ``submodel_preds``: List of predictions from each submodel
- ``submodel_covs``: List of variances from each submodel

**Derivative Prediction Flexibility (GDDEGP Submodels)**

When using ``submodel_type='gddegp'``, WDEGP inherits GDDEGP's prediction flexibility:  
you can predict directional derivatives along **any direction** via ``rays_predict``.  
The restriction is on derivative **order**, not direction—only orders present in **all**  
submodels can be predicted.

::

  # Predict in any direction (not limited to training directions)
  y_pred, y_cov = model.predict(
      X_test, params,
      rays_predict=rays_predict,  # Any directions
      calc_cov=True,
      return_deriv=True
  )

For detailed information on WDEGP prediction behavior and derivative restrictions  
by submodel type, see :doc:`predictions`.

**Complete WDEGP Example (DEGP Submodels)**
::

  from jetgp.wdegp.wdegp import wdegp
  
  # 10 training points, alternating submodel assignment
  X_train = np.linspace(0.5, 2.5, 10).reshape(-1, 1)
  
  sm1_indices = [0, 2, 4, 6, 8]  # Even indices → Submodel 1
  sm2_indices = [1, 3, 5, 7, 9]  # Odd indices → Submodel 2
  
  # Function and derivative values
  y_vals = f(X_train).reshape(-1, 1)                    # Shape (10, 1)
  dy_sm1 = df(X_train[sm1_indices]).reshape(-1, 1)      # Shape (5, 1)
  dy_sm2 = df(X_train[sm2_indices]).reshape(-1, 1)      # Shape (5, 1)
  
  # WDEGP data structure
  y_train = [
      [y_vals, dy_sm1],  # Submodel 1
      [y_vals, dy_sm2]   # Submodel 2
  ]
  
  der_indices = [
      [[[[1, 1]]]],  # Submodel 1: first-order derivative
      [[[[1, 1]]]]   # Submodel 2: first-order derivative
  ]
  
  derivative_locations = [
      [sm1_indices],  # Submodel 1 derivative locations
      [sm2_indices]   # Submodel 2 derivative locations (DISJOINT!)
  ]
  
  model = wdegp(
      X_train, y_train, n_order=1, n_bases=1,
      derivative_locations=derivative_locations,
      der_indices=der_indices,
      submodel_type='degp',
      normalize=True, kernel="SE", kernel_type="anisotropic"
  )
  
  params = model.optimize_hyperparameters(optimizer='jade', pop_size=100, n_generations=15)
  
  # Predict with submodel outputs
  y_pred, y_cov, sm_preds, sm_covs = model.predict(
      X_test, params, calc_cov=True, return_submodels=True
  )

**Complete WDEGP Example (GDDEGP Submodels)**
::

  from jetgp.wdegp.wdegp import wdegp
  
  # Partition by distance from origin
  distances = np.linalg.norm(X_train, axis=1)
  sm1_indices = [i for i in range(len(X_train)) if distances[i] < median_dist]
  sm2_indices = [i for i in range(len(X_train)) if distances[i] >= median_dist]
  
  # Build point-wise rays for each submodel
  # rays_list[submodel_idx][direction_idx] has shape (d, n_points_in_submodel)
  rays_list = [
      [rays_dir1_sm1, rays_dir2_sm1],  # Submodel 1 rays
      [rays_dir1_sm2, rays_dir2_sm2]   # Submodel 2 rays
  ]
  
  model = wdegp(
      X_train, y_train, n_order=1, n_bases=2,
      derivative_locations=derivative_locations,
      der_indices=der_indices,
      submodel_type='gddegp',
      rays_list=rays_list,
      normalize=True, kernel="SE", kernel_type="anisotropic"
  )

------------------------------------------------------------

Summary
-------
JetGP's modular design enables flexible experimentation with derivative-based Gaussian Process models.  
Each variant — DEGP, DDEGP, GDDEGP, and WDEGP — builds upon a shared foundation,  
with common kernel interfaces and optimization routines.  
This unified structure simplifies the exploration of increasingly rich GP formulations for  
sensitivity analysis, model calibration, and uncertainty quantification.

For detailed initialization instructions, see :doc:`initialization`.  
For prediction behavior and derivative restrictions, see :doc:`predictions`.