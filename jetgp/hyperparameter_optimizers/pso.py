from jetgp.utils import pso as core_pso

def pso(func, lb, ub, **kwargs):
    """
    Wrapper for PSO with unified interface.
    """
    # Extract PSO-specific args or set defaults
    pop_size = kwargs.pop("pop_size", 20)
    n_generations = kwargs.pop("n_generations", 50)
    local_opt_every = kwargs.pop("local_opt_every", 15)
    initial_positions = kwargs.pop("initial_positions", None)
    omega = kwargs.pop("omega", 0.5)
    phip = kwargs.pop("phip", 0.5)
    phig = kwargs.pop("phig", 0.5)
    seed = kwargs.pop("seed", 42)
    debug = kwargs.pop("debug", False)

    # Remove unrelated keys
    for key in ["n_restart_optimizer", "optimizer", "debug", "x0"]:
        kwargs.pop(key, None)

    return core_pso(
        func,
        lb,
        ub,
        pop_size=pop_size,
        n_generations=n_generations,
        local_opt_every=local_opt_every,
        initial_positions=initial_positions,
        omega=omega,
        phip=phip,
        phig=phig,
        seed=seed,
        debug=debug,
        **kwargs
    )
