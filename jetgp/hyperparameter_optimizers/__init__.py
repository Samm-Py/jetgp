from .pso import pso
from .jade import jade
from .lbfgs import lbfgs
from .powell import powell
from .cobyla import cobyla

__all__ = ["pso", "jade", "lbfgs", 'powell', 'cobyla']

# Create a dictionary for easy lookup by string
OPTIMIZERS = {
    "pso": pso,
    "jade": jade,
    "lbfgs": lbfgs,
    "powell": powell, 
    "cobyla": cobyla
}
