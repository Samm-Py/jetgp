"""
Slice-partition helper for Sliced DEGP (option C: shared full function-value
block, slice-restricted gradient blocks with signed weights).

Implements the sample-space partitioning of Cheng & Zimmermann (2024)
following eqs. 11-12, and builds the submodel specs for the 2-appendant
likelihood factorization (eq. 17). Both the function-value block and
derivative locations are sliced per submodel, matching the fully-sliced
R̃_{2*i} formulation from the paper.
"""
import numpy as np
import jetgp.utils as utils


def partition_indices(X_train, grads, m):
    """
    Partition training sample indices into m balanced slices along the
    coordinate with the largest derivative-based sensitivity index
    (eq. 11: S_k = mean(grads[:, k]**2)). Slice sizes follow eq. 12.

    The sensitivity index is computed from the first-order gradients only
    (the first ``d`` columns of ``grads``), regardless of whether
    higher-order derivatives are also present.

    Parameters
    ----------
    X_train : ndarray of shape (N, d)
    grads   : ndarray of shape (N, n_deriv_cols)
        Derivative observations.  Only the first ``d`` columns (first-order
        coordinate gradients) are used for the sensitivity index.
    m       : int, number of slices (m >= 3 is required by the 2-appendant
              factorization so that at least one single-slice correction
              block exists).

    Returns
    -------
    slices : list of list[int] of length m
        slices[i] is the list of point indices (into X_train) in slice i.
    slice_dim : int
        Coordinate chosen for slicing (argmax of S_k).
    """
    if m < 3:
        raise ValueError(f"m must be >= 3 for 2-appendant sliced likelihood, got {m}")
    N, D = X_train.shape
    if N < m:
        raise ValueError(f"N={N} must be >= m={m}")

    # Use only first-order gradients (first D columns) for sensitivity
    first_order_grads = grads[:, :D]
    S = np.mean(first_order_grads ** 2, axis=0)
    slice_dim = int(np.argmax(S))

    sort_idx = np.argsort(X_train[:, slice_dim])
    N_floor = N // m
    N_r = N % m
    sizes = [N_floor + 1 if i < N_r else N_floor for i in range(m)]

    slices = []
    cursor = 0
    for sz in sizes:
        slices.append(sort_idx[cursor:cursor + sz].tolist())
        cursor += sz
    return slices, slice_dim


def build_sliced_submodels(X_train, y_vals, grads, m, n_order=1):
    """
    Build the submodel specs for a 2-appendant sliced SDEGP (option C).

    Each submodel shares the full y function-value block and restricts
    gradient observations to the slice's (or slice pair's) points. The
    signed weights form the pair-minus-single combination from eq. 17
    (and its negated log form, eq. 20) — pair blocks get weight +1 and
    single-slice correction blocks get weight -1.

    Parameters
    ----------
    X_train : ndarray (N, d)
    y_vals  : ndarray (N,) or (N, 1)
    grads   : ndarray (N, n_deriv_types)
        All derivative observations stacked column-wise in the order
        produced by ``gen_OTI_indices(d, n_order)``.  For ``n_order=1``
        this is ``(N, d)``; for ``n_order=2`` in 2D this is ``(N, 5)``.
    m       : int, number of slices (>= 3)
    n_order : int, default=1
        Maximum derivative order.

    Returns
    -------
    submodel_data : list
        Ready to pass as `y_train` to sdegp(...).
    derivative_specs_list : list
        Ready to pass as `der_indices` to sdegp(...).
    derivative_locations_list : list
    function_locations_list : list of list[int]
        Per-submodel global point indices for the function-value block.
    submodel_weights : ndarray of shape (2m-3,)
    slice_dim : int
    slices : list[list[int]]
    """
    N, D = X_train.shape
    slices, slice_dim = partition_indices(X_train, grads, m)

    der_specs = utils.gen_OTI_indices(D, n_order)
    n_deriv_types = sum(len(group) for group in der_specs)
    y_col = y_vals.reshape(-1, 1)

    submodel_data = []
    derivative_specs_list = []
    derivative_locations_list = []
    function_locations_list = []
    weights = []

    # Pair blocks f(ỹ_i, ỹ_{i+1}): i = 1..m-1 -> 0-indexed 0..m-2, weight +1.
    for i in range(m - 1):
        pair_idx = sorted(slices[i] + slices[i + 1])
        grad_cols = [grads[pair_idx, j:j + 1] for j in range(n_deriv_types)]
        submodel_data.append([y_col[pair_idx].reshape(-1, 1)] + grad_cols)
        derivative_specs_list.append(der_specs)
        derivative_locations_list.append([pair_idx for _ in range(n_deriv_types)])
        function_locations_list.append(pair_idx)
        weights.append(+1.0)

    # Single-slice correction blocks f(ỹ_i): i = 2..m-1 -> 0-indexed 1..m-2,
    # weight -1 (denominator of f̂ in eq. 17).
    for i in range(1, m - 1):
        slice_idx = sorted(slices[i])
        grad_cols = [grads[slice_idx, j:j + 1] for j in range(n_deriv_types)]
        submodel_data.append([y_col[slice_idx].reshape(-1, 1)] + grad_cols)
        derivative_specs_list.append(der_specs)
        derivative_locations_list.append([slice_idx for _ in range(n_deriv_types)])
        function_locations_list.append(slice_idx)
        weights.append(-1.0)

    return (
        submodel_data,
        derivative_specs_list,
        derivative_locations_list,
        function_locations_list,
        np.asarray(weights, dtype=np.float64),
        slice_dim,
        slices,
    )
