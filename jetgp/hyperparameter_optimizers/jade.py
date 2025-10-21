from utils import jade as core_jade

def jade(func, lb, ub, **kwargs):
    """
    Wrapper for JADE with unified interface.
    """
    pop_size = kwargs.pop("pop_size", 20)
    n_generations = kwargs.pop("n_generations", 50)
    local_opt_every = kwargs.pop("local_opt_every", 15)
    initial_positions = kwargs.pop("initial_positions", None)
    p = kwargs.pop("p", 0.1)
    c = kwargs.pop("c", 0.1)
    seed = kwargs.pop("seed", 42)
    debug = kwargs.pop("debug", False)

    for key in ["n_restart_optimizer", "optimizer", "debug", "x0"]:
        kwargs.pop(key, None)

    return core_jade(
        func,
        lb,
        ub,
        pop_size=pop_size,
        n_generations=n_generations,
        p=p,
        c=c,
        local_opt_every=local_opt_every,
        initial_positions=initial_positions,
        seed=seed,
        debug=debug,
        **kwargs
    )
