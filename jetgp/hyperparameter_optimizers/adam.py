import numpy as np
import torch

def adam(func, lb, ub, **kwargs):
    """
    ADAM optimizer using PyTorch's optimizer but NumPy-compatible functions.
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
    
    lb = np.array(lb)
    ub = np.array(ub)
    
    def numerical_gradient(f, x, h=1e-8):
        """Compute numerical gradient using central differences."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            
            f_plus = f(x_plus)
            f_minus = f(x_minus)
            if isinstance(f_plus, tuple):
                f_plus = f_plus[0]
            if isinstance(f_minus, tuple):
                f_minus = f_minus[0]
                
            grad[i] = (f_plus - f_minus) / (2 * h)
        return grad
    
    best_x = None
    best_val = np.inf
    
    for restart in range(num_restart_optimizer):
        # Initialize starting point
        if x0 is not None and restart == 0:
            x = np.array(x0, dtype=float)
        else:
            x = np.random.uniform(lb, ub)
        
        # Convert to torch tensor for optimizer
        x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=False)
        
        # Create optimizer
        optimizer = torch.optim.Adam([x_tensor], lr=learning_rate, 
                                     betas=(beta1, beta2), eps=epsilon)
        
        prev_val = np.inf
        
        for t in range(maxiter):
            # Evaluate function with NumPy array
            x_np = x_tensor.detach().numpy()
            
            result = func(x_np)
            if isinstance(result, tuple):
                f_val, grad = result[0], result[1]
            else:
                f_val = result
                grad = numerical_gradient(func, x_np)
            
            # Check convergence
            if t > 0:
                f_diff = abs(prev_val - f_val)
                grad_norm = np.linalg.norm(grad)
                
                if f_diff < ftol and grad_norm < gtol:
                    if disp:
                        print(f"Converged at iteration {t}")
                    break
            
            prev_val = f_val
            
            # Set gradient manually
            optimizer.zero_grad()
            x_tensor.grad = torch.tensor(grad, dtype=torch.float64)
            
            # Optimizer step
            optimizer.step()
            
            # Project onto bounds
            with torch.no_grad():
                x_tensor.clamp_(torch.tensor(lb), torch.tensor(ub))
            
            if disp and t % 100 == 0:
                print(f"Iteration {t}: f(x) = {f_val:.6e}, ||grad|| = {grad_norm:.6e}")
        
        # Final evaluation
        x_final = x_tensor.detach().numpy()
        result = func(x_final)
        final_val = result[0] if isinstance(result, tuple) else result
        
        if final_val < best_val:
            best_val = final_val
            best_x = x_final.copy()
        
        if debug:
            print(f"[ADAM-Hybrid] Restart {restart+1}/{num_restart_optimizer} -> best_val={best_val}")
    
    return best_x, best_val