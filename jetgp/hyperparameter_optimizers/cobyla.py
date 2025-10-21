import numpy as np
from scipy.optimize import minimize

def cobyla(func, lb, ub, **kwargs):
    """
    COBYLA optimizer with bounds as inequality constraints and multi-start.

    Parameters
    ----------
    func : callable
        Function to minimize.
    lb, ub : array-like
        Lower and upper bounds.
    kwargs : dict
        Optional arguments:
        - x0 : initial guess for first restart
        - n_restart_optimizer : number of random restarts (default=10)
        - debug : bool, print intermediate results (default=False)
        - Any COBYLA options: maxiter, rhobeg, catol, f_target
    """
    x0 = kwargs.pop("x0", None)
    n_restart_optimizer = kwargs.pop("n_restart_optimizer", 10)
    debug = kwargs.pop("debug", False)

    # Extract COBYLA options
    options = {}
    for key in ["maxiter", "rhobeg", "catol", "f_target", "disp"]:
        if key in kwargs:
            options[key] = kwargs.pop(key)

    lb = np.array(lb)
    ub = np.array(ub)
    best_x = None
    best_val = np.inf

    # Inequality constraints for bounds
    def make_constraints(lb, ub):
        cons = []
        for i in range(len(lb)):
            cons.append(lambda x, i=i: x[i] - lb[i])  # x[i] >= lb[i]
            cons.append(lambda x, i=i: ub[i] - x[i])  # x[i] <= ub[i]
        return [{"type": "ineq", "fun": c} for c in cons]

    constraints = make_constraints(lb, ub)

    for i in range(n_restart_optimizer):
        if x0 is not None and i == 0:
            x_init = np.array(x0)
        else:
            x_init = np.random.uniform(lb, ub)

        res = minimize(
            func,
            x_init,
            method="COBYLA",
            constraints=constraints,
            options=options,
            **kwargs  # any extra kwargs
        )

        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x

        if debug:
            print(f"[COBYLA] Restart {i+1}/{n_restart_optimizer} -> best_val={best_val}")

    return best_x, best_val
