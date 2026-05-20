import numpy as np

from jetgp.full_gddegp.gddegp import gddegp
from posterior_queries import _make_ray


def _optimizer_kwargs_with_warm_start(optimizer_kwargs, initial_params, n_params):
    """Add a previous optimum as the first JADE candidate when dimensions match."""
    opt_kwargs = {
        "optimizer": "pso",
        "pop_size": 60,
        "n_generations": 20,
        "local_opt_every": 20,
        "debug": False,
    }
    if optimizer_kwargs is not None:
        opt_kwargs.update(optimizer_kwargs)
    if initial_params is None:
        return opt_kwargs

    initial_params = np.asarray(initial_params, dtype=float).reshape(-1)
    if initial_params.size == n_params and np.all(np.isfinite(initial_params)):
        opt_kwargs["initial_positions"] = np.atleast_2d(initial_params)
    return opt_kwargs


def fit_function_only_gp(X_train, y_train, n_dir_types=None,
                         kernel="SE", kernel_type="anisotropic",
                         optimizer_kwargs=None, normalize=True,
                         initial_params=None, max_directions=None):
    """
    n_order=0 (pure GP). Reserve enough basis directions so that the
    coordinate derivative covariance can be queried at prediction time.
    """
    d = X_train.shape[1]
    cap = max_directions if max_directions is not None else d
    cap_eff = max(1, min(d, cap))
    if n_dir_types is None:
        n_dir_types = cap_eff
    n_bases = 2 * cap_eff
    n_bases = max(n_bases, 4, 2 * max(1, n_dir_types))
    gp_model = gddegp(
        X_train,
        [y_train],
        n_order=0,
        rays_list=[],
        der_indices=[],
        derivative_locations=[],
        n_bases=n_bases,
        normalize=normalize,
        kernel=kernel,
        kernel_type=kernel_type,
    )
    opt_kwargs = _optimizer_kwargs_with_warm_start(
        optimizer_kwargs, initial_params, len(gp_model.bounds))
    params = gp_model.optimize_hyperparameters(**opt_kwargs)
    return gp_model, params


def _construct_directional_gp(X_train, y_train, directional_observations,
                              second_order_observations=None,
                              kernel="SE", kernel_type="anisotropic",
                              max_directions=None):
    """Construct a GDDEGP with first- and optional second-order observations."""
    first_slots = {}
    for obs in directional_observations:
        first_slots.setdefault(obs["slot"], []).append(obs)

    n_first_slots = max(first_slots.keys()) + 1 if first_slots else 0
    y_blocks = [y_train]
    rays_list = []
    derivative_locations = []

    for s in range(n_first_slots):
        slot_obs = first_slots.get(s, [])
        if not slot_obs:
            continue
        values = np.array([[o["value"]] for o in slot_obs])
        rays = np.hstack([_make_ray(o["direction"], 1) for o in slot_obs])
        locs = [o["x_index"] for o in slot_obs]
        y_blocks.append(values)
        rays_list.append(rays)
        derivative_locations.append(locs)

    n_first_types = len(rays_list)

    n_second_types = 0
    second_der = []
    if second_order_observations:
        second_slots = {}
        for obs in second_order_observations:
            second_slots.setdefault(obs["slot"], []).append(obs)

        n_second_slots = max(second_slots.keys()) + 1 if second_slots else 0
        for s in range(n_second_slots):
            slot_obs = second_slots.get(s, [])
            if not slot_obs:
                continue
            values = np.array([[o["value"]] for o in slot_obs])
            rays = np.hstack([_make_ray(o["direction"], 1) for o in slot_obs])
            locs = [o["x_index"] for o in slot_obs]
            y_blocks.append(values)
            rays_list.append(rays)
            derivative_locations.append(locs)
            second_der.append([[s + 1, 2]])
            n_second_types += 1

    first_der = [[[i + 1, 1]] for i in range(n_first_types)]
    total_types = n_first_types + n_second_types
    der_indices = [first_der + second_der] if total_types > 0 else []

    n_order = 2 if n_second_types > 0 else (1 if n_first_types > 0 else 0)
    n_unique_slots = max(n_first_types, n_second_types)
    d_dim = X_train.shape[1]
    cap = max_directions if max_directions is not None else d_dim
    n_bases = 2 * min(d_dim, max(1, cap))
    n_bases = max(n_bases, 4, 2 * max(1, n_unique_slots))

    return gddegp(
        X_train, y_blocks,
        n_order=n_order,
        rays_list=rays_list,
        der_indices=der_indices,
        derivative_locations=derivative_locations,
        n_bases=n_bases,
        normalize=True,
        kernel=kernel,
        kernel_type=kernel_type,
    )


def fit_directional_gp(X_train, y_train, directional_observations,
                       second_order_observations=None,
                       kernel="SE", kernel_type="anisotropic",
                       optimizer_kwargs=None, initial_params=None,
                       max_directions=None):
    """Fit a GDDEGP with first- and optional second-order observations."""
    gp_model = _construct_directional_gp(
        X_train,
        y_train,
        directional_observations,
        second_order_observations=second_order_observations,
        kernel=kernel,
        kernel_type=kernel_type,
        max_directions=max_directions,
    )
    opt_kwargs = _optimizer_kwargs_with_warm_start(
        optimizer_kwargs, initial_params, len(gp_model.bounds))
    params = gp_model.optimize_hyperparameters(**opt_kwargs)
    return gp_model, params


def _build_fantasy_gp(X_train, y_train, x_new, f_new,
                      directions, deriv_values,
                      kernel, kernel_type, max_directions=None):
    """Build a non-optimized fantasy model with an added function site."""
    X_aug = np.vstack([X_train, np.atleast_2d(x_new)])
    y_func_aug = np.vstack([y_train, np.array([[f_new]])])

    y_blocks = [y_func_aug]
    rays_list = []
    der_locs = []
    new_idx = X_aug.shape[0] - 1

    for val, v in zip(deriv_values, directions):
        y_blocks.append(np.array([[val]]))
        rays_list.append(_make_ray(v, 1))
        der_locs.append([new_idx])

    n_obs = len(directions)
    der_indices = [[[[i + 1, 1]] for i in range(n_obs)]] if n_obs > 0 else []
    d_dim = X_aug.shape[1]
    cap = max_directions if max_directions is not None else d_dim
    n_bases = 2 * min(d_dim, max(1, cap))
    n_bases = max(n_bases, 4, 2 * max(1, n_obs))

    return gddegp(
        X_aug, y_blocks,
        n_order=1 if n_obs > 0 else 0,
        rays_list=rays_list,
        der_indices=der_indices,
        derivative_locations=der_locs,
        n_bases=n_bases,
        normalize=True,
        kernel=kernel,
        kernel_type=kernel_type,
    )
