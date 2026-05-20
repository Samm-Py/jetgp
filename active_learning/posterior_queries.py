import numpy as np


def _make_ray(direction, n_points):
    v = np.asarray(direction, dtype=float).reshape(-1)
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("direction must be a finite nonzero vector")
    v = v / norm
    return np.tile(v.reshape(-1, 1), (1, n_points))


def query_function_posterior(gp_model, params, X_test):
    """Returns (mean, var) each shape (n_test,)."""
    mean, var = gp_model.predict(
        np.atleast_2d(X_test), params, calc_cov=True, return_deriv=False)
    return mean[0], var[0]


def query_function_posterior_batched(gp_model, params, X_test, batch_size=250):
    """Evaluate the function posterior without one large covariance call."""
    X_test = np.atleast_2d(X_test)
    if batch_size is None or batch_size <= 0:
        return query_function_posterior(gp_model, params, X_test)

    means = []
    variances = []
    for start in range(0, X_test.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), X_test.shape[0])
        mean_batch, var_batch = query_function_posterior(
            gp_model, params, X_test[start:stop])
        means.append(mean_batch)
        variances.append(var_batch)
    return np.concatenate(means), np.concatenate(variances)


def query_directional_variance(gp_model, params, X_test, direction):
    """
    Posterior variance of the directional derivative along `direction` at X_test.
    Uses derivs_to_predict=[[[1,1]]] -- direction-type 1.
    Returns var shape (n_test,).
    """
    ray = _make_ray(direction, X_test.shape[0])
    _, var = gp_model.predict(
        X_test, params,
        rays_predict=[ray],
        calc_cov=True,
        return_deriv=True,
        derivs_to_predict=[[[1, 1]]],
    )
    return var[1]


def query_directional_mean(gp_model, params, X_test, direction):
    """Posterior mean of directional derivative along `direction`."""
    ray = _make_ray(direction, X_test.shape[0])
    mean, _ = gp_model.predict(
        X_test, params,
        rays_predict=[ray],
        calc_cov=True,
        return_deriv=True,
        derivs_to_predict=[[[1, 1]]],
    )
    return mean[1]
