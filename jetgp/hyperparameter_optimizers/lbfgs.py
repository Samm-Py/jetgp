import numpy as np
from scipy.optimize import minimize

def lbfgs(func, lb, ub, **kwargs):
    """
    L-BFGS-B optimizer with bounds and multi-start.

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
        - maxiter : max iterations per restart (default=100)
        - ftol : function tolerance (default=1e-8)
        - gtol : gradient tolerance (default=1e-8)
        - debug : bool, print intermediate results (default=False)
        - disp : bool, print optimization info (default=False)
    """
    x0 = kwargs.pop("x0", None)
    num_restart_optimizer = kwargs.pop("n_restart_optimizer", 10)
    maxiter = kwargs.pop("maxiter", 100)
    ftol = kwargs.pop("ftol", 1e-8)
    gtol = kwargs.pop("gtol", 1e-8)
    debug = kwargs.pop("debug", False)
    disp = kwargs.pop("disp", False)

    lb = np.array(lb)
    ub = np.array(ub)

    best_x = None
    best_val = np.inf

    for i in range(num_restart_optimizer):
        if x0 is not None and i == 0:
            x_init = np.array(x0)
        else:
            x_init = np.random.uniform(lb, ub)

        res = minimize(
            func,
            x_init,
            method="L-BFGS-B",
            bounds=list(zip(lb, ub)),
            options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol, "disp": disp}
        )

        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x

        if debug:
            print(f"[L-BFGS-B] Restart {i+1}/{num_restart_optimizer} -> best_val={best_val}")

    return best_x, best_val
