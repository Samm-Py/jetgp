import numpy as np

def rprop(func, lb, ub, **kwargs):
    """
    RProp optimizer - pure NumPy implementation for performance.
    Resilient backpropagation with adaptive step sizes per parameter.
    """
    x0 = kwargs.pop("x0", None)
    num_restart_optimizer = kwargs.pop("n_restart_optimizer", 10)
    maxiter = kwargs.pop("maxiter", 1000)
    initial_step = kwargs.pop("learning_rate", 0.01)
    etas = kwargs.pop("etas", (0.5, 1.2))  # (etaminus, etaplus)
    step_sizes = kwargs.pop("step_sizes", (1e-6, 50))  # (min_step, max_step)
    ftol = kwargs.pop("ftol", 1e-8)
    gtol = kwargs.pop("gtol", 1e-8)
    debug = kwargs.pop("debug", False)
    disp = kwargs.pop("disp", False)

    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    n_dim = len(lb)
    
    eta_minus, eta_plus = etas
    min_step, max_step = step_sizes

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

        # RProp state variables (pure NumPy)
        step = np.full(n_dim, initial_step)  # Per-parameter step sizes
        prev_grad = np.zeros(n_dim)          # Previous gradient for sign comparison

        prev_val = np.inf

        for t in range(maxiter):
            # Evaluate function
            result = func(x)
            if isinstance(result, tuple):
                f_val, grad = result[0], result[1]
            else:
                f_val = result
                grad = forward_gradient(func, x, f_val)

            # Convergence check
            grad_norm = np.linalg.norm(grad)
            if t > 0:
                f_diff = abs(prev_val - f_val)
                if f_diff < ftol and grad_norm < gtol:
                    if disp:
                        print(f"Converged at iteration {t}")
                    break

            prev_val = f_val

            # RProp update (pure NumPy implementation)
            if t > 0:
                # Compute sign agreement: grad * prev_grad
                sign_product = grad * prev_grad
                
                # Where signs agree (positive product): increase step
                increase_mask = sign_product > 0
                step[increase_mask] *= eta_plus
                
                # Where signs disagree (negative product): decrease step
                decrease_mask = sign_product < 0
                step[decrease_mask] *= eta_minus
                
                # For sign changes, zero out gradient to skip update
                # (iRProp- variant: don't update if sign changed)
                grad[decrease_mask] = 0.0
                
                # Clamp step sizes
                np.clip(step, min_step, max_step, out=step)

            # Parameter update: x = x - sign(grad) * step
            x -= np.sign(grad) * step

            # Project onto bounds (in-place)
            np.clip(x, lb, ub, out=x)

            # Store gradient for next iteration (only non-zeroed values)
            prev_grad = grad.copy()

            if disp and t % 100 == 0:
                print(f"Iteration {t}: f(x) = {f_val:.6e}, ||grad|| = {grad_norm:.6e}")

        # Final evaluation
        result = func(x)
        final_val = result[0] if isinstance(result, tuple) else result

        if final_val < best_val:
            best_val = final_val
            best_x = x.copy()

        if debug:
            print(f"[RProp] Restart {restart+1}/{num_restart_optimizer} -> best_val={best_val}")

    return best_x, best_val