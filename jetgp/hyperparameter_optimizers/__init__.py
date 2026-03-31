from .pso import pso
from .jade import jade
from .lbfgs import lbfgs
from .powell import powell
from .cobyla import cobyla
from .adam import adam
from .rprop import rprop

__all__ = ["pso", "jade", "lbfgs", "adam", "rprop", "powell", "cobyla"]

# Create a dictionary for easy lookup by string
OPTIMIZERS = {
    "pso": pso,
    "jade": jade,
    "lbfgs": lbfgs,
    "adam": adam,
    "rprop": rprop,
    "powell": powell,
    "cobyla": cobyla,
}
