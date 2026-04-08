"""
Sparse Cholesky utilities using the paper's geometric sparsity criterion.

Sparsity pattern:
    dist(x_P(i), x_P(j)) <= rho * l(j)

where P is the MMD (maximin) ordering and l(j) is the fill-distance at step j.

Key property: the sparsity pattern depends only on the training points and rho,
NOT on the kernel hyperparameters. It is therefore computed once at model
initialisation and reused across all NLML evaluations during optimisation.

References
----------
Schäfer et al. (2021), "Sparse Cholesky factorization by Kullback-Leibler
minimization", SIAM Journal on Scientific Computing.
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve, cho_factor, cho_solve, solve_triangular
from scipy.spatial.distance import cdist
from line_profiler import profile


# =============================================================================
# Maximin (MMD) ordering
# =============================================================================

def mmd_ordering(X):
    """
    Compute the maximin ordering of training points.

    The first point is fixed at index 0. Each subsequent point is chosen as
    the one farthest from all already-selected points (maximum of minimum
    distances). Also returns the fill-distances l[j], where l[j] is the
    minimum distance from X[P[j]] to all previously selected points.

    Parameters
    ----------
    X : ndarray of shape (N, d)
        Training input points.

    Returns
    -------
    P : ndarray of int, shape (N,)
        Permutation indices (MMD ordering). P[0] is always 0.
    l : ndarray of float, shape (N,)
        Fill-distances. l[0] = 0 by convention; l[j] > 0 for j >= 1.
    """
    N = len(X)
    P = np.zeros(N, dtype=int)
    l = np.zeros(N)
    rem = list(range(1, N))
    for q in range(1, N):
        dists = cdist(X[rem], X[P[:q]]).min(axis=1)
        best = int(np.argmax(dists))
        P[q] = rem[best]
        l[q] = dists[best]
        rem.pop(best)
    return P, l


# =============================================================================
# Sparsity pattern (paper's geometric criterion)
# =============================================================================

def build_sparsity_pattern(X_ord, l, rho):
    """
    Build the sparsity pattern using the paper's criterion.

    For column j, the neighbour set S[j] contains j itself plus all i < j
    such that  dist(X_ord[i], X_ord[j]) <= rho * l[j].

    Because the criterion depends only on (X_ord, l, rho) and NOT on the
    kernel hyperparameters, this can be computed once and reused.

    Parameters
    ----------
    X_ord : ndarray of shape (N, d)
        Training points reordered by MMD (i.e. X[P]).
    l : ndarray of float, shape (N,)
        Fill-distances from mmd_ordering.
    rho : float
        Sparsity radius multiplier. Larger rho → denser pattern.

    Returns
    -------
    S : dict
        S[j] = sorted list of row indices (neighbours) for column j,
        always including j.
    """
    N = len(X_ord)
    S = {}
    for j in range(N):
        nb = [j]
        if l[j] > 0:
            for i in range(j):
                if np.linalg.norm(X_ord[i] - X_ord[j]) <= rho * l[j]:
                    nb.append(i)
        S[j] = sorted(nb)
    return S


# =============================================================================
# Sparse inverse-Cholesky factor U (column-by-column)
# =============================================================================

@profile
def build_U(K_ord, S, N, block_size=0, out=None):
    """
    Build the sparse inverse-Cholesky factor U column by column.

    For each column j, we solve the small local system K_sub * v = e_j
    where K_sub = K_ord[np.ix_(S[j], S[j])] and e_j is the unit vector
    for the diagonal entry. The column of U is then v / sqrt(v[jj]).

    When block_size > 0, consecutive columns are grouped into blocks.
    Within each block the sparsity patterns are nested (S[j] ⊂ S[j+1]),
    so the last column's pattern is the union.  We factor that union
    K_sub once via Cholesky and batch-solve all RHS vectors at once.

    Parameters
    ----------
    K_ord : ndarray of shape (N, N)
        Kernel matrix in MMD order.
    S : dict
        Sparsity pattern from build_sparsity_pattern.
    N : int
        Number of training points.
    block_size : int
        If > 0, batch columns in groups of this size (typically n_dim + 1).
        Set to 0 to use the original column-by-column path.
    out : ndarray of shape (N, N), optional
        Pre-allocated output buffer.  Zeroed before use.

    Returns
    -------
    U : ndarray of shape (N, N)
        Sparse upper-triangular inverse-Cholesky factor.
    """
    if out is not None:
        U = out
        U[:] = 0.0
    else:
        U = np.zeros((N, N))

    if block_size > 0:
        for start in range(0, N, block_size):
            end = min(start + block_size, N)
            cols = list(range(start, end))
            n_cols = len(cols)

            # Union pattern = last column's pattern (nested property)
            nb = S[end - 1] if isinstance(S[end - 1], np.ndarray) else np.asarray(S[end - 1])
            m = len(nb)

            K_sub = K_ord[np.ix_(nb, nb)]
            try:
                L, low = cho_factor(K_sub, lower=True)
                use_cho = True
            except np.linalg.LinAlgError:
                L, low = lu_factor(K_sub)
                use_cho = False

            # Build RHS matrix: each column k has a 1 at the position of cols[k] in nb
            positions = np.searchsorted(nb, cols)
            E = np.zeros((m, n_cols))
            E[positions, np.arange(n_cols)] = 1.0

            if use_cho:
                V = cho_solve((L, low), E)
            else:
                V = lu_solve((L, low), E)

            # Normalise each column: v /= sqrt(v[diag_pos])
            diag_vals = V[positions, np.arange(n_cols)]
            np.maximum(diag_vals, 1e-30, out=diag_vals)
            np.sqrt(diag_vals, out=diag_vals)
            V /= diag_vals

            U[np.ix_(nb, cols)] = V
    else:
        for j in range(N):
            nb = S[j] if isinstance(S[j], np.ndarray) else np.asarray(S[j])
            K_sub = K_ord[np.ix_(nb, nb)]
            dp = int(np.searchsorted(nb, j))
            e = np.zeros(len(nb))
            e[dp] = 1.0
            v = np.linalg.solve(K_sub, e)
            u_jj = np.sqrt(max(v[dp], 1e-30))
            U[nb, j] = v / u_jj

    return U


@profile
def build_U_from_phi(phi_exp_3d, S, N, block_size,
                     k_type, k_phys, deriv_lookup, sign_lookup,
                     P_full, sigma_n_sq, sigma_data_diag,
                     out=None):
    """
    Build sparse U directly from phi_exp_3d, skipping full K construction.

    Instead of building the full K matrix and extracting K_sub via fancy
    indexing, this assembles each K_sub on the fly from the kernel's
    intermediate phi_exp_3d representation.

    Parameters
    ----------
    phi_exp_3d : ndarray of shape (n_derivs, n_rows_func, n_cols_func)
    S : dict
        Sparsity pattern (in MMD-ordered indices).
    N : int
        Total number of rows/columns.
    block_size : int
        Batch columns in groups of this size.
    k_type, k_phys : int64 arrays of shape (N_total,)
        Maps from original K index to (derivative type, physical point).
    deriv_lookup : int64 array of shape (n_types, n_types)
    sign_lookup : float64 array of shape (n_types,)
    P_full : int64 array of shape (N_total,)
        MMD permutation: P_full[mmd_idx] = original K index.
    sigma_n_sq : float
        Noise variance.
    sigma_data_diag : float64 array of shape (N_total,)
        Diagonal of sigma_data**2 in MMD order.
    out : ndarray, optional
        Pre-allocated output buffer of shape (N, N).

    Returns
    -------
    U : ndarray of shape (N, N)
    """
    from jetgp.full_degp_sparse.optimizer import _extract_K_sub

    if out is not None:
        U = out
        # No need to zero: the sparsity pattern S is fixed, so the loop
        # overwrites exactly the same entries every call.  Non-pattern
        # entries stay at zero from the initial np.zeros allocation.
    else:
        U = np.zeros((N, N))

    for start in range(0, N, block_size):
        end = min(start + block_size, N)

        # Union pattern = last column's pattern (nested property)
        nb_union = S[end - 1] if isinstance(S[end - 1], np.ndarray) else np.asarray(S[end - 1])
        m_union = len(nb_union)

        # Map MMD-ordered union neighbourhood to original K indices
        orig_nb_union = P_full[nb_union]
        nb_type_union = k_type[orig_nb_union]
        nb_phys_union = k_phys[orig_nb_union]
        sd_diag_union = sigma_data_diag[nb_union]

        # Assemble K_sub for the union neighbourhood once
        K_sub_union = np.empty((m_union, m_union))
        _extract_K_sub(phi_exp_3d, nb_type_union, nb_phys_union,
                       deriv_lookup, sign_lookup,
                       sigma_n_sq, sd_diag_union, m_union, K_sub_union)

        # Batch solve: factor K_sub_union once, solve all columns at once.
        # Within each block, earlier columns condition on the union
        # neighbourhood (slightly larger than their own S[j]), which gives
        # a tighter Vecchia approximation at minimal accuracy cost.
        n_cols = end - start
        positions = np.searchsorted(nb_union, np.arange(start, end))
        E = np.zeros((m_union, n_cols))
        E[positions, np.arange(n_cols)] = 1.0
        try:
            L_u, low_u = cho_factor(K_sub_union, lower=True)
            V = cho_solve((L_u, low_u), E)
        except np.linalg.LinAlgError:
            V = np.linalg.solve(K_sub_union, E)
        diag_vals = V[positions, np.arange(n_cols)]
        np.maximum(diag_vals, 1e-30, out=diag_vals)
        np.sqrt(diag_vals, out=diag_vals)
        V /= diag_vals
        U[nb_union, start:end] = V

    return U


# =============================================================================
# Supernodes
# =============================================================================

def build_supernodes(X_ord, l, S, lam=1.5):
    """
    Aggregate adjacent columns into supernodes.

    Columns j-1 and j are merged into the same supernode if
        dist(X_ord[P(j-1)], X_ord[P(j)]) <= lam * l[j].

    The children of a supernode are the union of all neighbour sets of its
    parent columns. This allows a single LU factorisation to be shared across
    all parent columns in the supernode.

    Parameters
    ----------
    X_ord : ndarray of shape (N, d)
    l : ndarray of float, shape (N,)
    S : dict
        Sparsity pattern.
    lam : float
        Merging threshold (should be >= 1).

    Returns
    -------
    supernodes : list of dict
        Each dict has keys 'parents' (list of column indices) and
        'children' (sorted list of row indices = union of S[p]).
    """
    N = len(X_ord)
    supernodes = []
    current_parents = [0]
    for j in range(1, N):
        prev = current_parents[-1]
        dist_prev = np.linalg.norm(X_ord[prev] - X_ord[j])
        if l[j] > 0 and dist_prev <= lam * l[j]:
            current_parents.append(j)
        else:
            children = set()
            for p in current_parents:
                children.update(S[p])
            supernodes.append({
                'parents': list(current_parents),
                'children': sorted(children)
            })
            current_parents = [j]
    children = set()
    for p in current_parents:
        children.update(S[p])
    supernodes.append({
        'parents': list(current_parents),
        'children': sorted(children)
    })
    return supernodes


@profile
def build_U_supernodes(K_ord, supernodes, N):
    """
    Build sparse U using supernode structure.

    Each supernode factorises K_sub (children x children) once via LU and
    solves for all parent columns.  LU is used instead of Cholesky because
    derivative-enhanced kernel submatrices can be poorly conditioned and
    Cholesky may fail on them even when the full K is PD.

    Parameters
    ----------
    K_ord : ndarray of shape (N, N)
    supernodes : list of dict
        Output of build_supernodes or expand_supernodes_to_blocks.
    N : int

    Returns
    -------
    U : ndarray of shape (N, N)
    n_factorizations : int
        Number of LU factorisations performed.

    Raises
    ------
    np.linalg.LinAlgError
        If a supernode's K_sub is singular.
    """
    U = np.zeros((N, N))
    n_factorizations = 0
    for sn in supernodes:
        # Use pre-computed numpy array and position lookup if available
        ch = sn.get('children_arr')
        if ch is None:
            ch = np.asarray(sn['children'])
        ch_pos = sn.get('ch_pos')
        if ch_pos is None:
            ch_pos = {c: i for i, c in enumerate(sn['children'])}

        m = len(ch)
        K_sub = K_ord[np.ix_(ch, ch)]
        try:
            L, low = cho_factor(K_sub, lower=True)
            use_cho = True
        except np.linalg.LinAlgError:
            L, low = lu_factor(K_sub)
            if np.any(np.abs(np.diag(L)) < 1e-30):
                raise np.linalg.LinAlgError("Singular submatrix in supernode")
            use_cho = False
        n_factorizations += 1

        # Solve all parent columns in one batch
        parents = sn['parents']
        n_parents = len(parents)
        parent_positions = sn.get('parent_positions')
        if parent_positions is None:
            parent_positions = np.array([ch_pos[p] for p in parents])

        E = np.zeros((m, n_parents))
        E[parent_positions, np.arange(n_parents)] = 1.0
        if use_cho:
            V = cho_solve((L, low), E)
        else:
            V = lu_solve((L, low), E)  # (m, n_parents)

        # Vectorised normalisation: extract diagonal entries, clamp, sqrt
        diag_vals = V[parent_positions, np.arange(n_parents)]
        np.maximum(diag_vals, 1e-30, out=diag_vals)
        np.sqrt(diag_vals, out=diag_vals)
        V /= diag_vals  # broadcast: (m, n_parents) / (n_parents,)

        # Place all columns at once via fancy indexing
        U[np.ix_(ch, parents)] = V
    return U, n_factorizations


def build_deriv_supernodes(X_ord, l, S_phys, lam=1.5):
    """
    Expand physical supernodes to derivative index pairs.

    Each physical index p maps to derivative indices [2p, 2p+1].

    Parameters
    ----------
    X_ord : ndarray of shape (N, d)
    l : ndarray of float, shape (N,)
    S_phys : dict
        Physical sparsity pattern.
    lam : float

    Returns
    -------
    der_sns : list of dict
        Supernodes with 'parents' and 'children' in derivative index space.
    """
    phys_sns = build_supernodes(X_ord, l, S_phys, lam)
    der_sns = []
    for sn in phys_sns:
        der_parents = []
        for p in sn['parents']:
            der_parents.extend([2 * p, 2 * p + 1])
        der_children = []
        for c in sn['children']:
            der_children.extend([2 * c, 2 * c + 1])
        der_sns.append({
            'parents': der_parents,
            'children': sorted(set(der_children))
        })
    return der_sns


# =============================================================================
# NLML via sparse U
# =============================================================================

# @profile
def nlml_from_U(U, f):
    """
    Compute the negative log marginal likelihood from the sparse U factor.

    NLML = 0.5 * ||U.T f||^2 - sum(log|diag(U)|) + 0.5 * N * log(2pi)

    This is equivalent to the standard NLML when U.T @ U = K^{-1} exactly.
    For sparse U it is an approximation.

    Parameters
    ----------
    U : ndarray of shape (N, N)
        Sparse inverse-Cholesky factor (upper triangular in construction,
        stored as a full matrix with zeros outside the sparsity pattern).
    f : ndarray of shape (N,)
        Training targets (in MMD order).

    Returns
    -------
    float
        Approximate NLML value.
    """
    N = len(f)
    Ut_f = U.T @ f
    log_det_term = -np.sum(np.log(np.abs(np.diag(U)) + 1e-300))
    return 0.5 * np.dot(Ut_f, Ut_f) + log_det_term + 0.5 * N * np.log(2 * np.pi)


# =============================================================================
# Expansion utilities: physical ordering → full K-matrix ordering
# =============================================================================

def expand_mmd_permutation(P_phys, N, derivative_locations):
    """
    Expand the physical MMD ordering (size N) to a full K-matrix permutation.

    JetGP DEGP stores K in a block layout:
        rows 0..N-1             : function values  f(x_0), ..., f(x_{N-1})
        rows N..N+|dl[0]|-1     : 1st deriv at points derivative_locations[0]
        rows N+|dl[0]|..        : 2nd deriv at points derivative_locations[1]
        ...

    We build an interleaved ordering: for each physical point P_phys[q] (in
    MMD order), we first emit its function-value row, then each derivative row
    (if P_phys[q] appears in that derivative block).  This groups each point
    with all its observations before moving to the next MMD point.

    Parameters
    ----------
    P_phys : ndarray of int, shape (N,)
        Physical MMD permutation from mmd_ordering.
    N : int
        Number of physical training points.
    derivative_locations : list of list of int
        derivative_locations[k] = list of physical indices with derivative k.

    Returns
    -------
    P_full : ndarray of int, shape (N_total,)
        Permutation of {0, ..., N_total-1} for use with K[np.ix_(P_full, P_full)].
    phys_to_rows : list of list of int
        phys_to_rows[q] = all K-row indices belonging to physical point P_phys[q],
        in the order they appear in P_full.  Needed to build S_full.
    """
    # Build mapping: physical index p → list of K row indices
    p_to_k_rows = {p: [p] for p in range(N)}   # function-value block first

    cum = N
    for k, dl in enumerate(derivative_locations):
        for pos, p in enumerate(dl):
            p_to_k_rows[p].append(cum + pos)
        cum += len(dl)

    P_full = []
    phys_to_rows = []
    for p in P_phys:
        rows = p_to_k_rows[p]
        phys_to_rows.append(rows)
        P_full.extend(rows)

    return np.array(P_full, dtype=int), phys_to_rows


def expand_sparsity_to_blocks(S_phys, phys_to_rows):
    """
    Expand a physical sparsity pattern to the full K-matrix index space.

    S_phys[q] gives the physical MMD-order neighbours of column q (indices ≤ q).
    S_full[j] gives the K_ord_full row indices (in P_full-indexed space) that
    are neighbours of column j (indices ≤ j).

    Because the interleaved ordering places all rows of physical point q at
    P_full positions  [n_rows_before_q .. n_rows_before_q + n_blocks_q - 1],
    column j (= P_full position of some row of physical point q) has the same
    physical neighbours as q.

    Parameters
    ----------
    S_phys : dict
        S_phys[q] = sorted list of physical MMD indices, 0 ≤ i ≤ q, for q in range(N).
    phys_to_rows : list of list of int
        phys_to_rows[q] = P_full positions belonging to physical point q
        (output of expand_mmd_permutation, already in P_full index space
        since they are consecutive starting from the offset for q).

    Returns
    -------
    S_full : dict
        S_full[j] = sorted list of P_full-space indices ≤ j that are neighbours
        of column j.
    """
    # Build a flat lookup: P_full position j → physical MMD index q
    j_to_q = {}
    P_full_offset = 0
    for q, rows in enumerate(phys_to_rows):
        for _ in rows:
            j_to_q[P_full_offset] = q
            P_full_offset += 1

    N_total = P_full_offset

    # For each physical point q, compute the starting P_full index
    q_start = {}
    offset = 0
    for q, rows in enumerate(phys_to_rows):
        q_start[q] = offset
        offset += len(rows)

    S_full = {}
    for j in range(N_total):
        q_j = j_to_q[j]
        nb = []
        for q_nb in S_phys[q_j]:          # physical neighbours (≤ q_j)
            start = q_start[q_nb]
            for b in range(len(phys_to_rows[q_nb])):
                row = start + b
                if row <= j:
                    nb.append(row)
        S_full[j] = sorted(nb) if nb else [j]

    return S_full


def expand_supernodes_to_blocks(supernodes, phys_to_rows):
    """
    Expand physical-level supernodes to cover all derivative rows.

    Each physical parent p maps to all P_full positions belonging to p.
    Children are expanded similarly.

    Parameters
    ----------
    supernodes : list of dict
        Physical supernodes from build_supernodes.
    phys_to_rows : list of list of int
        phys_to_rows[q] = P_full positions for physical point q.

    Returns
    -------
    full_supernodes : list of dict
        Supernodes in P_full-indexed space.
    """
    # Compute starting P_full offset for each physical point q
    q_start = {}
    offset = 0
    for q, rows in enumerate(phys_to_rows):
        q_start[q] = offset
        offset += len(rows)

    full_supernodes = []
    for sn in supernodes:
        full_parents = []
        for p in sn['parents']:
            start = q_start[p]
            full_parents.extend(range(start, start + len(phys_to_rows[p])))

        full_children = []
        for c in sn['children']:
            start = q_start[c]
            full_children.extend(range(start, start + len(phys_to_rows[c])))

        full_supernodes.append({
            'parents': full_parents,
            'children': sorted(set(full_children)),
        })

    return full_supernodes


# @profile
def alpha_from_U(U, f):
    """
    Compute K^{-1} f using the sparse U factor.

    Since U.T @ U ≈ K^{-1}, we have K^{-1} f ≈ U @ (U.T @ f).

    Parameters
    ----------
    U : ndarray of shape (N, N)
    f : ndarray of shape (N,)

    Returns
    -------
    ndarray of shape (N,)
    """
    return U @ (U.T @ f)
