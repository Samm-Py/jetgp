JetGP Modules Overview
======================

Overview
--------
JetGP provides a unified framework for constructing **derivative-enhanced Gaussian Process (GP)** models.  
Each module corresponds to a distinct GP formulation, differing in how function and derivative information are incorporated.  
The core model families include:

- **DEGP** – Derivative-Enhanced Gaussian Process  
- **WDEGP** – Weighted Derivative-Enhanced Gaussian Process  
- **DDEGP** – Directional Derivative-Enhanced Gaussian Process  
- **GDDEGP** – Generalized Directional Derivative-Enhanced Gaussian Process  
- **WDDEGP** – Weighted Directional Derivative-Enhanced Gaussian Process  

All these models share a common structure and initialization interface.  
This page provides an overview of their usage and explains the shared arguments that appear in all implementations.

------------------------------------------------------------

Available Modules
-----------------

**1. full_degp**

- **File:** `full_degp/degp.py`  
- **Class:** `degp`  
- **Description:**  
  Implements a standard Derivative-Enhanced Gaussian Process (DEGP) where function values and their derivatives  
  (up to arbitrary order) are jointly modeled.

**2. wdegp**

- **File:** `wdegp/wdegp.py`  
- **Class:** `wdegp`  
- **Description:**  
  Implements the **Weighted Derivative-Enhanced Gaussian Process (WDEGP)** framework, inspired by the  
  *Weighted Gradient-Enhanced Kriging (WGEK)* method introduced by Han *et al.* :cite:`HanWeightedGEK`  
  and extended in :cite:`improved_gek`.  

  In this formulation, the training data are divided into smaller subsets to build multiple **submodels**,  
  each trained on a reduced set of derivatives. Predictions from all submodels are then  
  **combined through weighted summation**. This approach preserves the accuracy advantages of 
  derivative-enhanced models while **reducing the computational cost** associated with large matrix inversions. 

**3. full_ddegp**

- **File:** `full_ddegp/ddegp.py`  
- **Class:** `ddegp`  
- **Description:**  
  Constructs a **Directional Derivative-Enhanced Gaussian Process (DDEGP)** using directional derivative information  
  evaluated along **global directions**.  
  This means that all training points share the same derivative direction, allowing the model to capture variation  
  along specific, predefined axes of sensitivity.  
  Such an approach is well suited for problems where global sensitivity along known direction(s) is of interest.

**4. full_gddegp**

- **File:** `full_gddegp/gddegp.py`  
- **Class:** `gddegp`  
- **Description:**  
  Implements the **Generalized Directional Derivative-Enhanced Gaussian Process (GDDEGP)**, an extension of DDEGP  
  that allows **distinct directional derivative information at each training point**.  
  In contrast to DDEGP, the directional vectors need not be the same globally, enabling the model to represent  
  spatially varying directional sensitivities.  

**5. wddegp**

- **File:** `wddegp/wddegp.py`  
- **Class:** `wddegp`  
- **Description:**  
  Implements a **Weighted Directional Derivative-Enhanced Gaussian Process (WDDEGP)**,  
  extending DDEGP with a **weighted submodel framework**.  
  Here, the training data are partitioned into submodels, and **each submodel can use a different directional derivative configuration**,  
  allowing localized variation in derivative directions.

------------------------------------------------------------

.. _common_arguments:

Common Arguments
----------------

**der_indices**  
Specifies which derivatives are included in the model.  
This argument is a **nested list**, where each sublist contains all derivative components of a particular order.  
Each derivative component is itself a list specifying the variable indices and derivative order.  
Examples:

**1D function**  
::
  
  der_indices = [[[[1, 1]]]]        # first-order derivative
  der_indices = [[[[1, 1]], [[1, 2]]]]  # first- and second-order derivatives

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

  der_indices = [[[[1, 1]], [[1, 2]]]]
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

  der_indices = [[[[1, 1]], [[1, 2]]]]
  derivative_locations = [
      [0, 2, 4, 6, 8],  # df/dx at even indices
      [1, 3, 5, 7, 9]   # d²f/dx² at odd indices
  ]

**Note:** If ``derivative_locations`` is ``None`` or not provided, all derivatives are assumed to be available at all training points.
.. _normalize_argument:

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
------------------------------------

The same logic extends to second-order or higher derivatives:

.. math::

   \frac{\partial^n y'}{\partial \mathbf{v}'^n} 
   = \frac{1}{\sigma_y} 
     \sum_{j_1,\ldots,j_n} v_{j_1}\sigma_{x,j_1} \cdots v_{j_n}\sigma_{x,j_n} 
     \frac{\partial^n y}{\partial x_{j_1}\cdots\partial x_{j_n}}.

and the normalized higher order directional derivative becomes:

.. math::

   \frac{\partial^n y'}{\partial \mathbf{v}'^n} = \frac{1}{\sigma_y} 
   \frac{\partial^n y}{\partial \mathbf{v}^n}.

Notes
-----

- Normalization statistics (means and standard deviations) are stored internally  
  and automatically applied when evaluating predictions on new test points.  
- When predictions are returned, JetGP automatically rescales them to the original physical units.

.. _kernel_argument:

**kernel**  
Specifies the kernel function used in the GP covariance structure.  
Available kernels are include:

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
    ``smoothness_parameter = 2 × (max derivative order used for training)``  
    to ensure that the kernel is sufficiently smooth to support the highest-order derivatives.

------------------------------------------------------------

Summary
-------
JetGP’s modular design enables flexible experimentation with derivative-based Gaussian Process models.  
Each variant — DEGP, WDEGP, DDEGP, GDDEGP, and WDDEGP — builds upon a shared foundation,  
with common kernel interfaces and optimization routines.  
This unified structure simplifies the exploration of increasingly rich GP formulations for  
sensitivity analysis, model calibration, and uncertainty quantification.
