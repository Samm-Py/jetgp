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

------------------------------------------------------------

**derivative_locations**  
Specifies which training points have each derivative defined in ``der_indices``.  
This argument is a **list of lists**, where each sublist contains the training point indices 
for the corresponding derivative. The structure must match ``der_indices`` exactly—one entry 
per derivative. Indices can be **non-contiguous** (e.g., ``[0, 2, 5, 7]`` is valid).

For **weighted models** (WDEGP, WDDEGP), an additional outer list is added for submodels:
``derivative_locations[submodel_idx][deriv_idx] = [list of point indices]``.

Examples:

**DEGP – 1D function, first-order derivative at subset of points**  
::

  der_indices = [[[[1, 1]]]]
  derivative_locations = [[2, 3, 4, 5]]  # 1 derivative → 1 entry

**DEGP – 1D function, first and second-order at different locations**  
::

  der_indices = [[[[1, 1]], [[1, 2]]]]
  derivative_locations = [
      [0, 1, 2, 3, 4, 5],  # df/dx at all 6 points
      [2, 3, 4]            # d²f/dx² at middle 3 points only
  ]

**DEGP – 2D function, different coverage per derivative**  
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

**WDEGP – 2 submodels with different derivative coverage**  
::

  # Submodel 0: boundary points with 1st order only
  # Submodel 1: interior points with 1st and 2nd order
  derivative_specs = [
      [
          [[[1, 1]], [[2, 1]]]   # Submodel 0: 1st order derivatives
      ],
      [
          [[[1, 1]], [[2, 1]]],                    # Submodel 1: 1st order
          [[[1, 2]], [[1,1],[2,1]], [[2, 2]]]      # Submodel 1: 2nd order
      ]
  ]
  derivative_locations = [
      # Submodel 0: 1 derivative type
      [
          [0, 1, 2, 3, 4, 5, 6, 7]   # 1st order at boundary points
      ],
      # Submodel 1: 2 derivative types
      [
          [8, 9, 10, 11, 12],        # 1st order at interior points
          [8, 9, 10, 11, 12]         # 2nd order at interior points
      ]
  ]

**WDEGP – single submodel, heterogeneous derivative coverage**  
::

  # Boundary: 1st order only, Interior: 1st and 2nd order
  derivative_specs = [
      [
          [[[1, 1]], [[2, 1]]],  # 1st order derivatives
          [[[1, 2]], [[2, 2]]]   # 2nd order derivatives
      ]
  ]
  derivative_locations = [
      [
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # 1st order at all points
          [8, 9, 10, 11, 12]                            # 2nd order at interior only
      ]
  ]

.. note::

   When WDEGP is configured with a single submodel, it reduces to the standard DEGP case.
   The only structural difference is the additional outer list for the submodel dimension.
   For example, these configurations are functionally equivalent:
   
   **DEGP:**
   ::
   
     derivative_locations = [
         [0, 1, 2, 3, 4],  # df/dx1
         [0, 1, 2, 3, 4]   # df/dx2
     ]
   
   **WDEGP (single submodel):**
   ::
   
     derivative_locations = [
         [                     # Submodel 0
             [0, 1, 2, 3, 4],  # df/dx1
             [0, 1, 2, 3, 4]   # df/dx2
         ]
     ]
   
   In single-submodel mode, WDEGP also supports ``return_deriv=True`` for derivative 
   predictions, just like DEGP.



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
        derivative_locations=derivative_locations,
        normalize=True,
        kernel="SE",
        kernel_type="anisotropic"
    )

**Module-Specific Parameters:**

**y_train_list**
  Training observations for function and derivatives.
  
  - **Type:** ``list`` of ``numpy.ndarray``
  - **Description:** Ordered list of training data arrays. The first element contains function values
    at all training points, and subsequent elements contain derivative observations. The order of
    derivative arrays must match ``der_indices``. When using ``derivative_locations``, each
    derivative array length must match the corresponding entry in ``derivative_locations``.
  - **Example structure (all derivatives at all points):**
  
    .. code-block:: python
    
        y_train_list = [
            y_vals,      # shape: (num_training_points,)
            dy_dx1,      # shape: (num_training_points,)
            dy_dx2,      # shape: (num_training_points,)
        ]

  - **Example structure (derivatives at subset of points):**
  
    .. code-block:: python
    
        # derivative_locations = [[2, 3, 4, 5], [2, 3, 4, 5]]
        y_train_list = [
            y_vals,      # shape: (num_training_points,) - all 6 points
            dy_dx1,      # shape: (4,) - only at points 2, 3, 4, 5
            dy_dx2,      # shape: (4,) - only at points 2, 3, 4, 5
        ]

**derivative_locations**
  Training point indices for each derivative.
  
  - **Type:** ``list`` of ``list`` of ``int``
  - **Description:** Specifies which training points have each derivative defined in ``der_indices``.
    The structure must match ``der_indices`` exactly—one entry per derivative. Indices can be
    non-contiguous (e.g., ``[0, 2, 5, 7]`` is valid).
  - **Example (1D, first-order derivative at subset of points):**
  
    .. code-block:: python
    
        der_indices = [[[[1, 1]]]]
        derivative_locations = [[2, 3, 4, 5]]  # df/dx at points 2, 3, 4, 5 only

  - **Example (2D, different coverage per derivative):**
  
    .. code-block:: python
    
        der_indices = [[ [[1, 1]], [[2, 1]] ]]
        derivative_locations = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],  # ∂f/∂x₁ at all 9 points
            [4, 5, 7, 8]                   # ∂f/∂x₂ at interior 4 points only
        ]

**Usage Notes:**

- DEGP incorporates specified derivatives at the training points defined by ``derivative_locations``.
- The first element of ``y_train_list`` (function values) always has length ``num_training_points``.
- Each subsequent array in ``y_train_list`` must have length matching the corresponding entry
  in ``derivative_locations``.
- The correspondence between ``y_train_list``, ``der_indices``, and ``derivative_locations`` must be exact.
------------------------------------------------------------

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
        derivative_locations,
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

**derivative_locations**
  Training point assignments for submodels.
  
  - **Type:** ``list`` of ``list`` of ``list`` of ``int``
  - **Description:** Specifies which training point(s) are associated with each derivative type
    for each submodel. The structure is nested three levels deep:
    
    - Outermost list: one entry per submodel
    - Middle list: one entry per derivative type in that submodel
    - Inner list: indices of training points that have this derivative type
    
    Indices correspond to rows in ``X_train`` and can be non-contiguous (e.g., ``[0, 2, 4]`` is valid).
    
  - **Example:**
  
    .. code-block:: python
    
        # Example 1: Simple case with contiguous indices
        submodel_indices = [
            [[0, 1, 2], [3, 4, 5]],      # Submodel 1: deriv1 at points 0-2, deriv2 at points 3-5
            [[6, 7, 8], [9, 10, 11]]     # Submodel 2: deriv1 at points 6-8, deriv2 at points 9-11
        ]
        
        # Example 2: Non-contiguous indices (now supported!)
        submodel_indices = [
            [[0, 2, 4], [1, 3, 5]],      # Submodel 1: alternating point assignments
            [[6, 8, 10], [7, 9, 11]]     # Submodel 2: alternating point assignments
        ]
        
        # Example 3: Different numbers of points per derivative type
        submodel_indices = [
            [[0, 1, 2, 3], [5, 7, 9]],   # Submodel 1: 4 points for deriv1, 3 for deriv2
            [[10, 11], [12, 13, 14, 15]] # Submodel 2: 2 points for deriv1, 4 for deriv2
        ]

**derivative_specifications**
  Derivative components for each submodel.
  
  - **Type:** ``list`` of ``list`` of ``list`` of ``list`` of ``int``
  - **Description:** Specifies which derivatives each submodel incorporates. The structure is nested:
    
    - Outermost list: one entry per submodel
    - Middle list: one entry per derivative type in that submodel  
    - Inner list: derivative specifications for that type (can include multiple derivatives)
    - Innermost: ``[[dim, order]]`` for pure derivatives or ``[[dim1, order1], [dim2, order2]]`` for mixed
    
  - **Notation:** 
    - ``[[dim, order]]`` = derivative of order ``order`` w.r.t. dimension ``dim``
    - ``[[dim1, order1], [dim2, order2]]`` = mixed derivative (e.g., ∂²f/∂x₁∂x₂)
    
  - **Example:**
  
    .. code-block:: python
    
        # 2D example: Submodel 0 with 1st order only, Submodel 1 with 1st and 2nd order
        derivative_specs = [
            # Submodel 0: only 1st order derivatives
            [
                [[[1, 1]], [[2, 1]]]   # ∂f/∂x₁ and ∂f/∂x₂
            ],
            # Submodel 1: 1st and 2nd order derivatives  
            [
                [[[1, 1]], [[2, 1]]],                    # 1st order: ∂f/∂x₁, ∂f/∂x₂
                [[[1, 2]], [[1,1],[2,1]], [[2, 2]]]      # 2nd order: ∂²f/∂x₁², ∂²f/∂x₁∂x₂, ∂²f/∂x₂²
            ]
        ]
        
        # Matching submodel_indices structure:
        submodel_indices = [
            [boundary_indices],                    # Submodel 0: 1 derivative type
            [interior_indices, interior_indices]   # Submodel 1: 2 derivative types
        ]
        

**Usage Notes:**

- WDEGP partitions training data into submodels to reduce computational cost.
- **All submodels share the same function values** but may incorporate different derivative information.
- Each submodel uses derivatives only at its designated subset of training points.
- Indices in ``submodel_indices`` can be non-contiguous, allowing flexible assignment of training points.
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
        derivative_locations,
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
  - **Shape correspondence:** ``Y_train[0]`` has shape ``(n_train, 1)`` for function values.
    ``Y_train[i+1]`` has shape ``(len(derivative_locations[i]), 1)`` for direction ``i``.

**rays**
  Global directional derivative directions.
  
  - **Type:** ``numpy.ndarray``
  - **Shape:** ``(dimension, n_directions)``
  - **Description:** Defines the direction vectors along which derivatives are evaluated.
    All training points that have a given direction share the same ray vector (global directions).
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

**derivative_locations**
  Specifies which training points have which directional derivatives.
  
  - **Type:** ``list`` of ``list`` of ``int``, or ``None``
  - **Default:** ``None`` (all directions at all training points)
  - **Description:** Each inner list contains the training point indices where that 
    direction's derivative is available. Indices do NOT need to be contiguous.
  - **Length:** Must match the number of directions (columns in ``rays``).
  - **Correspondence:** ``derivative_locations[i]`` specifies which points have direction ``i``,
    and ``Y_train[i+1]`` must have length ``len(derivative_locations[i])``.
  - **Example:**
  
    .. code-block:: python
    
        # 10 training points, 3 directional derivatives
        # Direction 0: at all points
        # Direction 1: at points 0-4 only
        # Direction 2: at points 5-9 only
        derivative_locations = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Direction 0
            [0, 1, 2, 3, 4],                  # Direction 1
            [5, 6, 7, 8, 9]                   # Direction 2
        ]
        
        # Y_train shapes:
        # Y_train[0]: (10, 1) - function values at all points
        # Y_train[1]: (10, 1) - direction 0 derivatives
        # Y_train[2]: (5, 1)  - direction 1 derivatives
        # Y_train[3]: (5, 1)  - direction 2 derivatives

**Usage Notes:**

- DDEGP uses **global** directional derivatives: the ray vectors are the same at all points 
  that have a given direction.
- **Direction vectors should be normalized to unit length** for proper interpretation of directional derivatives.
- Use ``derivative_locations`` for **selective coverage**: different points can have different 
  subsets of the available directions.
- When ``derivative_locations=None``, all directions are assumed to be available at all training points.
- This approach is suited for problems with known global sensitivity directions, such as 
  fixed sensor orientations or wind directions.

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
        rays_list,
        der_indices,
        derivative_locations,
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
  - **Shape correspondence:** ``y_train_list[0]`` has shape ``(n_train, 1)`` for function values.
    ``y_train_list[i+1]`` has shape ``(len(derivative_locations[i]), 1)`` for direction ``i``.

**rays_list**
  Point-specific directional derivative directions.
  
  - **Type:** ``list`` of ``numpy.ndarray``
  - **Length:** Number of directional derivative types (must match length of ``derivative_locations``)
  - **Description:** Each element ``rays_list[i]`` is an array of shape 
    ``(dimension, len(derivative_locations[i]))``. Unlike DDEGP where all points share the same 
    ray vectors, GDDEGP allows each training point to have **unique** direction vectors.
    **Direction vectors should be normalized to unit length (norm = 1) for proper interpretation.**
  - **Correspondence:** ``rays_list[i][:, j]`` is the ray vector for training point 
    ``derivative_locations[i][j]``.
  - **Example for 2D problem, 12 training points, 2 directions at interior points only:**
  
    .. code-block:: python
    
        # Suppose interior_indices = [2, 3, 5, 7, 8, 10] (6 interior points)
        interior_indices = [2, 3, 5, 7, 8, 10]
        n_interior = len(interior_indices)
        
        # Build point-specific rays (e.g., gradient-aligned and perpendicular)
        rays_dir1 = np.zeros((2, n_interior))  # Direction 1 at interior points
        rays_dir2 = np.zeros((2, n_interior))  # Direction 2 at interior points
        
        for j, idx in enumerate(interior_indices):
            # Compute gradient direction at this point
            gradient = compute_gradient(X_train[idx])
            grad_norm = np.linalg.norm(gradient)
            
            # Direction 1: normalized gradient
            rays_dir1[:, j] = gradient / grad_norm
            
            # Direction 2: perpendicular (90° rotation in 2D)
            rays_dir2[:, j] = np.array([-rays_dir1[1, j], rays_dir1[0, j]])
        
        rays_list = [rays_dir1, rays_dir2]

**derivative_locations**
  Specifies which training points have which directional derivatives.
  
  - **Type:** ``list`` of ``list`` of ``int``
  - **Description:** Each inner list contains the training point indices where that 
    direction's derivative is available. Indices do NOT need to be contiguous.
    Different directions can have different point sets.
  - **Length:** Must match the length of ``rays_list``.
  - **Correspondence:** 
    
    - ``derivative_locations[i][j]`` is the training point index
    - ``rays_list[i][:, j]`` is the ray vector for that point
    - ``y_train_list[i+1][j]`` is the derivative value at that point
    
  - **Example:**
  
    .. code-block:: python
    
        # 12 training points total
        # Direction 1 (gradient): at interior points [2, 3, 5, 7, 8, 10]
        # Direction 2 (perpendicular): at same interior points
        derivative_locations = [
            [2, 3, 5, 7, 8, 10],  # Direction 1
            [2, 3, 5, 7, 8, 10]   # Direction 2
        ]
        
        # rays_list shapes:
        # rays_list[0]: (2, 6) - unique ray at each of 6 interior points
        # rays_list[1]: (2, 6) - unique ray at each of 6 interior points
        
        # y_train_list shapes:
        # y_train_list[0]: (12, 1) - function values at all points
        # y_train_list[1]: (6, 1)  - direction 1 derivatives at interior points
        # y_train_list[2]: (6, 1)  - direction 2 derivatives at interior points

  - **Mixed coverage example** (different points have different directions):
  
    .. code-block:: python
    
        # Direction 1: at points [0, 1, 2, 3, 4]
        # Direction 2: at points [3, 4, 5, 6, 7]
        # Points 3, 4 have BOTH directions
        derivative_locations = [
            [0, 1, 2, 3, 4],  # Direction 1
            [3, 4, 5, 6, 7]   # Direction 2
        ]
        
        # rays_list[0]: (d, 5) - rays for points 0, 1, 2, 3, 4
        # rays_list[1]: (d, 5) - rays for points 3, 4, 5, 6, 7

**Usage Notes:**

- GDDEGP allows **spatially varying** directional derivatives: each point can have unique ray directions.
- **Direction vectors should be normalized to unit length** for proper interpretation of directional derivatives.
- Use ``derivative_locations`` to specify **selective coverage**: not all points need derivatives, 
  and different points can have different subsets of directions.
- **Key difference from DDEGP:** In DDEGP, ``rays.shape = (d, n_directions)`` and all points share 
  the same rays. In GDDEGP, ``rays_list[i].shape = (d, len(derivative_locations[i]))`` with unique 
  rays per point.
- This formulation is useful when local sensitivity varies across the input space, such as 
  gradient-aligned directions or adaptive directional sampling.

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