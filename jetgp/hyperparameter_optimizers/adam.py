import numpy as np

def adam(func, lb, ub, **kwargs):
    """
    ADAM optimizer - pure NumPy implementation for performance.
    Works with custom types like sparse matrices.
    """
    x0 = kwargs.pop("x0", None)
    num_restart_optimizer = kwargs.pop("n_restart_optimizer", 10)
    maxiter = kwargs.pop("maxiter", 1000)
    learning_rate = kwargs.pop("learning_rate", 0.001)
    beta1 = kwargs.pop("beta1", 0.9)
    beta2 = kwargs.pop("beta2", 0.999)
    epsilon = kwargs.pop("epsilon", 1e-8)
    ftol = kwargs.pop("ftol", 1e-8)
    gtol = kwargs.pop("gtol", 1e-8)
    debug = kwargs.pop("debug", False)
    disp = kwargs.pop("disp", False)

    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    n_dim = len(lb)

    def forward_gradient(f, x, f_x, h=1e-7):
        """
        Forward differences - reuses f(x) from current evaluation.
        Only n function evaluations instead of 2n for central differences.
        """
        grad = np.empty(n_dim)
        x_pert = x.copy()
        for i in range(n_dim):
            x_pert[i] += h
            f_plus = f(x_pert)
            if isinstance(f_plus, tuple):
                f_plus = f_plus[0]
            grad[i] = (f_plus - f_x) / h
            x_pert[i] = x[i]  # Reset in-place
        return grad

    best_x = None
    best_val = np.inf

    for restart in range(num_restart_optimizer):
        # Initialize starting point
        if x0 is not None and restart == 0:
            x = np.array(x0, dtype=np.float64)
        else:
            x = np.random.uniform(lb, ub)

        # Adam state variables (pure NumPy)
        m = np.zeros(n_dim)  # First moment estimate
        v = np.zeros(n_dim)  # Second moment estimate

        prev_val = np.inf

        for t in range(1, maxiter + 1):  # Start at 1 for bias correction
            # Evaluate function
            result = func(x)
            if isinstance(result, tuple):
                f_val, grad = result[0], result[1]
            else:
                f_val = result
                grad = forward_gradient(func, x, f_val)

            # Convergence check
            grad_norm = np.linalg.norm(grad)
            if t > 1:
                f_diff = abs(prev_val - f_val)
                if f_diff < ftol and grad_norm < gtol:
                    if disp:
                        print(f"Converged at iteration {t}")
                    break

            prev_val = f_val

            # Adam update (pure NumPy implementation)
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad * grad)

            # Bias-corrected estimates
            m_hat = m / (1.0 - beta1 ** t)
            v_hat = v / (1.0 - beta2 ** t)

            # Parameter update
            x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            # Project onto bounds (in-place)
            np.clip(x, lb, ub, out=x)

            if disp and t % 100 == 0:
                print(f"Iteration {t}: f(x) = {f_val:.6e}, ||grad|| = {grad_norm:.6e}")

        # Final evaluation
        result = func(x)
        final_val = result[0] if isinstance(result, tuple) else result

        if final_val < best_val:
            best_val = final_val
            best_x = x.copy()

        if debug:
            print(f"[ADAM] Restart {restart+1}/{num_restart_optimizer} -> best_val={best_val}")

    return best_x, best_val