Initializing JetGP Modules
==========================

Overview
--------
This guide provides detailed instructions for initializing each JetGP module.
All modules share common parameters but differ in how they structure training data
and incorporate derivative information. This page walks through the initialization
signature for each module and explains the purpose and format of each argument.

------------------------------------------------------------

Common Parameters
-----------------

The following parameters appear across most or all JetGP modules:

**X_train**
  Training input locations.
  
  - **Type:** ``numpy.ndarray``
  - **Shape:** ``(num_training_points, dimension)``
  - **Description:** Array of input coordinates where function and derivative observations are available.

**n_order**
  Maximum derivative order used in training.
  
  - **Type:** ``int``
  - **Description:** Specifies the highest-order derivatives included in the model.
    For weighted models (WDEGP, WDDEGP), different submodels may use different derivative orders,
    but ``n_order`` represents the maximum across all submodels.

**n_bases**
  Problem dimension.
  
  - **Type:** ``int``
  - **Description:** Number of input variables (same as ``X_train.shape[1]``).

**der_indices**
  Derivative component specification.
  
  - **Type:** ``list`` (nested)
  - **Description:** Defines which derivatives are included, using the same format as described
    in the :ref:`Common Arguments <common_arguments>` section. Each sublist corresponds to
    derivatives of a particular order, with individual components specified as lists of
    variable indices and orders.

**normalize**
  Enable data normalization.
  
  - **Type:** ``bool``
  - **Default:** Typically ``True``
  - **Description:** When ``True``, inputs and outputs are standardized for improved numerical
    stability. See the :ref:`normalize <normalize_argument>` documentation for details.

**kernel**
  Kernel function selection.
  
  - **Type:** ``str``
  - **Options:** ``"SE"``, ``"RQ"``, ``"Matern"``, ``"SineExp"``
  - **Description:** Specifies the covariance kernel. See :ref:`kernel <kernel_argument>` for formulas.

**kernel_type**
  Kernel parameterization.
  
  - **Type:** ``str``
  - **Options:** ``"isotropic"``, ``"anisotropic"``
  - **Description:** Determines whether a single length scale (isotropic) or dimension-specific
    length scales (anisotropic) are used.

------------------------------------------------------------

Module-Specific Initialization
-------------------------------

1. DEGP (Derivative-Enhanced Gaussian Process)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``full_degp/degp.py``

**Signature:**

.. code-block:: python

    from jetgp.full_degp.degp import degp
    
    gp = degp(
        X_train,
        y_train_list,
        n_order,
        n_bases,
        der_indices,
        normalize=True,
        kernel="SE",
        kernel_type="anisotropic"
    )

**Module-Specific Parameters:**

**y_train_list**
  Training observations for function and derivatives.
  
  - **Type:** ``list`` of ``numpy.ndarray``
  - **Description:** Ordered list of training data arrays. The first element contains function values,
    and subsequent elements contain derivative observations. The order must match ``der_indices``.
  - **Example structure:**
  
    .. code-block:: python
    
        y_train_list = [
            y_vals,      # shape: (num_training_points,)
            dy_dx1,      # first-order derivatives
            dy_dx2,
            d2y_dx1dx1,  # second-order derivatives
            d2y_dx1dx2,
            d2y_dx2dx2
        ]

**Usage Notes:**

- DEGP incorporates all specified derivatives at all training points.
- Each array in ``y_train_list`` must have length ``num_training_points``.
- The correspondence between ``y_train_list`` and ``der_indices`` must be exact.

------------------------------------------------------------

2. WDEGP (Weighted Derivative-Enhanced Gaussian Process)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``wdegp/wdegp.py``

**Signature:**

.. code-block:: python

    from jetgp.wdegp.wdegp import wdegp
    
    gp = wdegp(
        X_train,
        submodel_data,
        n_order,
        n_bases,
        submodel_indices,
        derivative_specifications,
        normalize=True,
        kernel="SE",
        kernel_type="anisotropic"
    )

**Module-Specific Parameters:**

**submodel_data**
  Training data for each submodel.
  
  - **Type:** ``list`` of ``list`` of ``numpy.ndarray``
  - **Description:** Outer list contains one entry per submodel. Each submodel entry is itself
    a list of arrays (function values followed by derivatives) similar to ``y_train_list`` in DEGP.
    **Important:** All submodels share the same function values array (i.e., ``y_vals_sub1 = y_vals_sub2 = ... = y_vals_subn``),
    but may have different derivative observations depending on ``derivative_specifications``.
  - **Example:**
  
    .. code-block:: python
    
        # All submodels use the same function values
        y_vals = np.random.randn(num_points)
        
        submodel_data = [
            [y_vals, dy_dx1_sub1, dy_dx2_sub1, ...],  # Submodel 1 data
            [y_vals, dy_dx1_sub2, dy_dx2_sub2, ...],  # Submodel 2 data
            ...
        ]

**submodel_indices**
  Training point assignments for submodels.
  
  - **Type:** ``list`` of ``list`` of ``int``
  - **Description:** Specifies which training point(s) are associated with each submodel.
    Each inner list contains indices corresponding to rows in ``X_train``.
    
    **CRITICAL REQUIREMENT:** Indices for each submodel **must be contiguous**. For example,
    ``[0, 1, 2]`` is valid, but ``[0, 2, 4]`` is not. If you want to use non-contiguous
    points from your original data, you must **reorder** ``X_train`` so that each submodel's
    points become contiguous before initialization.
    
  - **Example:**
  
    .. code-block:: python
    
        # Valid: contiguous indices for each submodel
        submodel_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
        
        # Invalid: non-contiguous indices
        # submodel_indices = [[0, 2, 4], [1, 3, 5]]  # This will NOT work!
        
        # To use non-contiguous points, reorder X_train first:
        # Original points at indices [0, 2, 4, 6, 8] and [1, 3, 5, 7, 9]
        original_indices_sub1 = [0, 2, 4, 6, 8]
        original_indices_sub2 = [1, 3, 5, 7, 9]
        
        # Reorder training data
        reorder = original_indices_sub1 + original_indices_sub2
        X_train = X_train_original[reorder]
        
        # Now use contiguous indices
        submodel_indices = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

**derivative_specifications**
  Derivative components for each submodel.
  
  - **Type:** ``list`` of ``list``
  - **Description:** Specifies which derivatives each submodel incorporates. Each entry
    corresponds to one submodel and contains a subset of derivative labels.
    These labels typically come from a helper function like ``gen_OTI_indices``.
  - **Example:**
  
    .. code-block:: python
    
        derivative_specifications = [
            [[[1, 1]], [[2, 1]]],                      # Submodel 1: first-order only
            [[[1, 2]], [[1, 1], [2, 1]], [[2, 2]]],   # Submodel 2: second-order
            ...
        ]

**Usage Notes:**

- WDEGP partitions training data into submodels to reduce computational cost.
- **All submodels share the same function values** but may incorporate different derivative information.
- Each submodel uses derivatives only at its designated subset of training points.
- **Critical:** Submodel indices must be contiguous. If you want to assign non-contiguous points
  to a submodel, you must reorder ``X_train`` and all associated data before initialization.
- Predictions are combined via weighted averaging across submodels.

------------------------------------------------------------

3. DDEGP (Directional Derivative-Enhanced Gaussian Process)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``full_ddegp/ddegp.py``

**Signature:**

.. code-block:: python

    from jetgp.full_ddegp.ddegp import ddegp
    
    gp = ddegp(
        X_train,
        Y_train,
        n_order,
        der_indices,
        rays,
        normalize=True,
        kernel="SE",
        kernel_type="anisotropic"
    )

**Module-Specific Parameters:**

**Y_train**
  Training observations for function and directional derivatives.
  
  - **Type:** ``list`` of ``numpy.ndarray``
  - **Description:** Same format as ``y_train_list`` in DEGP. Contains function values
    followed by directional derivative observations. Order must match ``der_indices``.

**rays**
  Global directional derivative directions.
  
  - **Type:** ``numpy.ndarray``
  - **Shape:** ``(dimension, rays_per_point)``
  - **Description:** Defines the direction vectors along which derivatives are evaluated.
    All training points share the same directional information (global directions).
    **Direction vectors should be normalized to unit length (norm = 1) for proper interpretation.**
  - **Example for 2D problem with 3 directions:**
  
    .. code-block:: python
    
        rays = np.array([
            [1.0, 0.5, 0.0],   # x-components of 3 directions
            [0.0, 0.5, 1.0]    # y-components of 3 directions
        ])
        
        # Normalize each direction to unit length
        for i in range(rays.shape[1]):
            rays[:, i] = rays[:, i] / np.linalg.norm(rays[:, i])

**Usage Notes:**

- DDEGP uses **global** directional derivatives: all training points use the same direction vectors.
- **Direction vectors should be normalized to unit length** for proper interpretation of directional derivatives.
- This approach is suited for problems with known global sensitivity directions.

------------------------------------------------------------

4. GDDEGP (Generalized Directional Derivative-Enhanced Gaussian Process)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``full_gddegp/gddegp.py``

**Signature:**

.. code-block:: python

    from jetgp.full_gddegp.gddegp import gddegp
    
    gp = gddegp(
        X_train,
        y_train_list,
        n_order,
        rays_array,
        der_indices,
        normalize=True,
        kernel="SE",
        kernel_type="anisotropic"
    )

**Module-Specific Parameters:**

**y_train_list**
  Training observations for function and directional derivatives.
  
  - **Type:** ``list`` of ``numpy.ndarray``
  - **Description:** Same format as in DEGP. Contains function values followed by
    directional derivative observations in order matching ``der_indices``.

**rays_array**
  Point-specific directional derivative directions.
  
  - **Type:** ``list`` of ``numpy.ndarray``
  - **Description:** A list where each element is an array of shape ``(dimension, num_training_points)``.
    The length of the list specifies how many directional derivatives are used per point.
    Unlike DDEGP, each training point can have **different** direction vectors.
    **Direction vectors should be normalized to unit length (norm = 1) for proper interpretation.**
  - **Example for 2D problem, 10 training points, 2 directions per point:**
  
    .. code-block:: python
    
        rays_array = [
            np.random.randn(2, 10),  # First directional derivative at each point
            np.random.randn(2, 10)   # Second directional derivative at each point
        ]
        
        # Normalize each direction vector to unit length
        for ray_set in rays_array:
            for i in range(ray_set.shape[1]):
                ray_set[:, i] = ray_set[:, i] / np.linalg.norm(ray_set[:, i])
        
        # Note: wrapped in list when passed to gddegp
        gp = gddegp(X_train, y_train_list, n_order, 
                    rays_array=[rays_array], ...)

**Usage Notes:**

- GDDEGP allows **spatially varying** directional derivatives.
- Each training point can encode different sensitivity directions.
- **Direction vectors should be normalized to unit length** for proper interpretation of directional derivatives.
- The ``rays_array`` parameter is passed as ``[rays_array]`` (wrapped in a list) in the initialization.
- This formulation is useful when local sensitivity varies across the input space.

------------------------------------------------------------

5. WDDEGP (Weighted Directional Derivative-Enhanced Gaussian Process)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``wddegp/wddegp.py``

**Signature:**

.. code-block:: python

    from jetgp.wddegp.wddegp import wddegp
    
    gp = wddegp(
        X_train,
        y_train_data,
        n_order,
        n_bases,
        submodel_indices,
        der_indices,
        rays_data,
        normalize=True,
        kernel="SE",
        kernel_type="anisotropic"
    )

**Module-Specific Parameters:**

**y_train_data**
  Training data for each submodel.
  
  - **Type:** ``list`` of ``list`` of ``numpy.ndarray``
  - **Description:** Similar to ``submodel_data`` in WDEGP. Outer list has one entry per submodel,
    each containing function values and directional derivative observations.

**submodel_indices**
  Training point assignments for submodels.
  
  - **Type:** ``list`` of ``list`` of ``int``
  - **Description:** Same as in WDEGP. Specifies which training points are associated with each submodel.

**rays_data**
  Directional derivative directions for each submodel.
  
  - **Type:** ``list`` of ``numpy.ndarray``
  - **Description:** Each element specifies the global directions used by one submodel.
    Shape of each array is ``(dimension, rays_per_point)``.
    Unlike GDDEGP, directions are global within each submodel but **can vary across submodels**.
    **Direction vectors should be normalized to unit length (norm = 1) for proper interpretation.**
  - **Example for 3 submodels in 2D:**
  
    .. code-block:: python
    
        rays_data = [
            np.array([[1.0, 0.0], [0.0, 1.0]]),      # Submodel 1: axis-aligned
            np.array([[0.707, 0.707], [-0.707, 0.707]]),  # Submodel 2: diagonal (normalized)
            np.array([[1.0], [0.0]])                 # Submodel 3: single direction
        ]

**Usage Notes:**

- WDDEGP combines the weighted submodel framework (WDEGP) with directional derivatives (DDEGP).
- Each submodel uses global directional derivatives, but different submodels can use different directions.
- **Direction vectors should be normalized to unit length** for proper interpretation of directional derivatives.
- This enables localized directional sensitivity while maintaining computational efficiency.
- Particularly useful when different regions of input space have different dominant sensitivity directions.

------------------------------------------------------------

Best Practices
--------------

1. **Always enable normalization** (``normalize=True``) when using derivative information
   to ensure numerical stability.

2. **Match data order to derivative indices**: The order of arrays in ``y_train_list``,
   ``Y_train``, or ``submodel_data`` must exactly correspond to the structure defined
   in ``der_indices``.

3. **Choose appropriate smoothness**: When using the Matern kernel with derivative-enhanced GPs,
   set ``smoothness_parameter ≥ 2 × n_order`` to ensure the kernel is sufficiently smooth.

4. **Submodel design**: For WDEGP and WDDEGP, balance the number of submodels against
   computational cost. More submodels reduce individual matrix sizes but increase
   the overhead of combining predictions. **Remember:** Submodel indices must be contiguous,
   so reorder your training data appropriately before initialization.

5. **Direction vector scaling**: Direction vectors in ``rays``, ``rays_array``, and ``rays_data``
   **should be normalized to unit length (norm = 1)** for proper interpretation of directional derivatives.
   Normalization ensures consistent scaling and meaningful comparison of sensitivities across different directions.

6. **Anisotropic kernels**: Use ``kernel_type="anisotropic"`` when input dimensions
   have different characteristic length scales or when derivative information reveals
   directional dependencies.

------------------------------------------------------------

Summary
-------

JetGP provides five module variants for incorporating derivative information into
Gaussian Process models:

- **DEGP**: Full derivative information at all points
- **WDEGP**: Weighted submodels with flexible derivative specifications
- **DDEGP**: Global directional derivatives
- **GDDEGP**: Point-specific directional derivatives
- **WDDEGP**: Weighted submodels with directional derivatives

Each module follows a consistent initialization pattern while accommodating different
use cases for derivative-enhanced modeling. Choose the module based on your data
availability, computational constraints, and the spatial characteristics of your problem.