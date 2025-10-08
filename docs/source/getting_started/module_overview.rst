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
- **GDDEGP** – Gradient- and Directional-Derivative-Enhanced Gaussian Process  
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
  (up to arbitrary order) are jointly modeled. The model supports anisotropic kernels and hyperparameter optimization  
  using PSO or JADE.

**2. wdegp**

- **File:** `wdegp/wdegp.py`  
- **Class:** `wdegp`  
- **Description:**  
  Implements the **Weighted Derivative-Enhanced Gaussian Process (WDEGP)** framework, inspired by the  
  *Weighted Gradient-Enhanced Kriging (WGEK)* method introduced by Han *et al.* :cite:`HanWeightedGEK`  
  and extended in :cite:`improved_gek`.  

  In this formulation, the training data are divided into smaller subsets to build multiple **submodels**,  
  each trained on a reduced set of function values and derivatives. Predictions from all submodels are then  
  **combined through weighted summation**, where the weights depend on local variance, model likelihood,  
  or spatial proximity.  

  This approach preserves the accuracy advantages of derivative-enhanced models while **reducing the computational  
  cost** associated with large matrix inversions. As shown in high-dimensional aerodynamic optimization problems  
  involving up to 108 design variables :cite:`HanWeightedGEK`, the weighted formulation achieves accuracy comparable  
  to full gradient-enhanced kriging at a fraction of the training cost.

**3. full_ddegp**

- **File:** `full_ddegp/ddegp.py`  
- **Class:** `ddegp`  
- **Description:**  
  Directional DEGP model that extends DEGP by modeling **directional derivatives** along arbitrary directions.  
  This is particularly useful in problems with structured input domains (e.g., anisotropic surfaces).

**4. full_gddegp**

- **File:** `full_gddegp/gddegp.py`  
- **Class:** `gddegp`  
- **Description:**  
  Combines both gradient-based and directional derivative information into a single model.  
  This allows fine-grained control of spatial derivative information and improved learning in high-dimensional systems.

**5. wddegp**

- **File:** `wddegp/wddegp.py`  
- **Class:** `wddegp`  
- **Description:**  
  Weighted Directional DEGP model that introduces weights across derivative directions.  
  This model generalizes both **WDEGP** and **DDEGP**, allowing selective emphasis of certain derivative components.

------------------------------------------------------------

Common Arguments
----------------
All JetGP model classes share a set of key arguments during initialization.  
The following table summarizes these parameters and their purpose.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - **Argument**
     - **Description**

   * - ``der_indices``
     - Specifies which derivatives are included in the model and how they map to hypercomplex tags.  
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

       **Explanation:**  
       - Each top-level list corresponds to a derivative order (first, second, third, etc.).  
       - Each sublist contains all derivative components of that order.  
       - Each component is a list of pairs: ``[variable_index, derivative_order]``  
         - e.g., ``[[1, 2]]`` → second derivative of the first variable  
         - e.g., ``[[1, 1], [2, 1]]`` → mixed derivative ∂²/∂x₁∂x₂  
       - This structure allows the model to automatically construct the correct kernel derivatives for each input.
       - **Convenience function:** JetGP provides ``gen_OTI_indices`` in ``utils`` to automatically generate these indices for any dimension ``d`` and maximum order ``p``:
         ::
           
           from utils import gen_OTI_indices
           der_indices = gen_OTI_indices(d, p)

   * - ``normalize``
     - If ``True``, the training inputs and outputs are normalized before model fitting.  
       Normalization improves numerical stability during kernel evaluation and hyperparameter optimization.  
       When set to ``False``, raw data values are used directly.

   * - ``kernel``
     - Specifies the kernel function used in the GP covariance structure.  
       Available kernels typically include:
       - ``"SE"`` – Squared Exponential (RBF)  
       - ``"RQ"`` – Rational Quadratic  
       - ``"Matern"`` – Matern family of kernels  
       - ``"SineExp"`` – Sine-Exponential kernel  
       
       The choice of kernel affects smoothness and correlation structure.

   * - ``kernel_type``
     - Determines whether the kernel is **isotropic** or **anisotropic**:
       - ``"isotropic"`` → single length scale shared across all input dimensions  
       - ``"anisotropic"`` → distinct length scale per dimension  
       
       Anisotropic kernels provide flexibility in modeling directional dependencies in multivariate inputs.

------------------------------------------------------------

Additional Modules
------------------
JetGP also includes several helper modules supporting core functionality:

- **kernel_funcs** (`kernel_funcs/kernel_funcs.py`): Defines base covariance functions used across models.  
- **utils** (`utils.py`): Utility functions for normalization, matrix operations, and general helpers.  
- **plotting_helper** (`plotting_helper.py`): Tools for visualizing function predictions, residuals, and kernel surfaces.  
- **acquisition_functions** (`acquisition_functions/acquisition_funcs.py`): Acquisition strategies for active learning or Bayesian optimization.

------------------------------------------------------------

Summary
-------
JetGP’s modular design enables flexible experimentation with derivative-based Gaussian Process models.  
Each variant — DEGP, WDEGP, DDEGP, GDDEGP, and WDDEGP — builds upon a shared foundation,  
with common kernel interfaces and optimization routines.  
This unified structure simplifies the exploration of increasingly rich GP formulations for  
sensitivity analysis, model calibration, and uncertainty quantification.
