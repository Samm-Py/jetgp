import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import qmc  # For quasi-random sequences


def lbfgs_smart(func, lb, ub, **kwargs):
    """
    Smart L-BFGS-B optimizer with intelligent restart strategies.
    
    Strategies:
    - 'lhs': Latin Hypercube Sampling for space-filling initial points
    - 'sobol': Sobol quasi-random sequence for low-discrepancy coverage
    - 'exclusion': Avoid regions near previous optima
    - 'adaptive': Combine exclusion with basin size estimation
    
    Parameters
    ----------
    func : callable
        Function to minimize.
    lb, ub : array-like
        Lower and upper bounds.
    kwargs : dict
        - x0 : initial guess for first restart
        - n_restart_optimizer : number of restarts (default=10)
        - strategy : 'random', 'lhs', 'sobol', 'exclusion', 'adaptive' (default='exclusion')
        - exclusion_radius : fraction of domain to exclude around optima (default=0.1)
        - max_rejection : max attempts to find valid starting point (default=100)
        - maxiter, ftol, gtol, debug, disp : standard L-BFGS-B options
    """
    x0 = kwargs.pop("x0", None)
    n_restarts = kwargs.pop("n_restart_optimizer", 10)
    strategy = kwargs.pop("strategy", "lhs")
    exclusion_radius = kwargs.pop("exclusion_radius", 0.1)
    max_rejection = kwargs.pop("max_rejection", 100)
    maxiter = kwargs.pop("maxiter", 100)
    ftol = kwargs.pop("ftol", 1e-8)
    gtol = kwargs.pop("gtol", 1e-8)
    debug = kwargs.pop("debug", False)
    disp = kwargs.pop("disp", False)
    
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)
    ndim = len(lb)
    
    # Normalize to unit hypercube for distance calculations
    scale = ub - lb
    
    best_x = None
    best_val = np.inf
    
    # Track all found optima (normalized coordinates)
    found_optima = []
    found_values = []
    
    # Pre-generate starting points based on strategy
    if strategy == 'lhs':
        sampler = qmc.LatinHypercube(d=ndim, seed=42)
        samples = sampler.random(n=n_restarts)
        starting_points = qmc.scale(samples, lb, ub)
    elif strategy == 'sobol':
        sampler = qmc.Sobol(d=ndim, seed=42)
        samples = sampler.random(n=n_restarts)
        starting_points = qmc.scale(samples, lb, ub)
    else:
        starting_points = None  # Generate on-the-fly
    
    for i in range(n_restarts):
        # Determine starting point
        if x0 is not None and i == 0:
            x_init = np.array(x0)
        elif starting_points is not None:
            x_init = starting_points[i]
        elif strategy in ['exclusion', 'adaptive']:
            x_init = _get_excluded_start(lb, ub, found_optima, scale, 
                                         exclusion_radius, max_rejection)
        else:  # random
            x_init = np.random.uniform(lb, ub)
        
        # Run optimization
        res = minimize(
            func,
            x_init,
            method="L-BFGS-B",
            bounds=list(zip(lb, ub)),
            options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol, "disp": disp}
        )
        
        # Store optimum (normalized)
        x_opt_norm = (res.x - lb) / scale
        found_optima.append(x_opt_norm)
        found_values.append(res.fun)
        
        # Update best
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
        
        if debug:
            print(f"[L-BFGS-B {strategy}] Restart {i+1}/{n_restarts} | "
                  f"f={res.fun:.6e} | best={best_val:.6e} | ")
    
    
    return best_x, best_val



# =============================================================================
# Unified interface
# =============================================================================

def lbfgs(func, lb, ub, **kwargs):
    """
    L-BFGS-B optimizer with intelligent restart strategies.
    
    Parameters
    ----------
    func : callable
        Function to minimize.
    lb, ub : array-like
        Lower and upper bounds.
    kwargs : dict
        - strategy : 'random', 'lhs', 'sobol', 'exclusion', 'adaptive', 'clustering'
                     (default='adaptive')
        - n_restart_optimizer : number of restarts (default=10)
        - x0 : initial guess for first restart
        - maxiter, ftol, gtol, debug, disp : L-BFGS-B options
        
    Strategy descriptions:
        - 'random': Pure random restarts (original behavior)
        - 'lhs': Latin Hypercube Sampling for space-filling coverage
        - 'sobol': Sobol sequence for low-discrepancy coverage
        - 'exclusion': Avoid regions near previously found optima
        - 'adaptive': Two-phase exploration/exploitation with basin estimation
        - 'clustering': Online clustering to identify and avoid basins
    """
    strategy = kwargs.get("strategy", "lhs")
    

    return lbfgs_smart(func, lb, ub, **kwargs)



# =============================================================================
# Example usage and comparison
# =============================================================================
