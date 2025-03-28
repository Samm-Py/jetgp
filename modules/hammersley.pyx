"""Create samples from the Halton sequence."""
import numpy


import cython
cimport cython
import  numpy as np
cimport numpy as np
import modules.primes
import modules.vdc as vdc
cimport libc.math as cmath # Import c- math libraries.
import math 
from Sequences.sobol_constants import DIM_MAX, LOG_MAX, POLY, SOURCE_SAMPLES
from cython.parallel import prange

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def create_hammersley_samples(int order, int dim, int burnin, np.ndarray[long, ndim=1] primes):
    """
    Create samples from the Halton sequence.

    In statistics, Halton sequences are sequences used to generate points in
    space for numerical methods such as Monte Carlo simulations. Although these
    sequences are deterministic, they are of low discrepancy, that is, appear
    to be random for many purposes. They were first introduced in 1960 and are
    an example of a quasi-random number sequence. They generalise the
    one-dimensional van der Corput sequences. For ``dim == 1`` the sequence
    falls back to Van Der Corput sequence.

    Args:
        order (int):
            The order of the Halton sequence. Defines the number of samples.
        dim (int):
            The number of dimensions in the Halton sequence.
        burnin (int):
            Skip the first ``burnin`` samples. If negative, the maximum of
            ``primes`` is used.
        primes (tuple):
            The (non-)prime base to calculate values along each axis. If
            empty, growing prime values starting from 2 will be used.

    Returns:
        (numpy.ndarray):
            Halton sequence with ``shape == (dim, order)``.

    Examples:
        >>> distribution = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(0, 1))
        >>> samples = distribution.sample(3, rule="halton")
        >>> samples.round(4)
        array([[0.125 , 0.625 , 0.375 ],
               [0.4444, 0.7778, 0.2222]])
        >>> samples = distribution.sample(4, rule="halton")
        >>> samples.round(4)
        array([[0.125 , 0.625 , 0.375 , 0.875 ],
               [0.4444, 0.7778, 0.2222, 0.5556]])

    """
    
    cdef double [:,:] out = np.empty((dim,order))
    
    if burnin < 0:
        burnin = max(primes)
    cdef np.ndarray [int] indices = np.arange(0,order,1, dtype = 'int32') + burnin
    cdef  double [:] vdc_samples = np.empty((order,), dtype = 'float64') 
    cdef int j
    cdef int k
    cdef int max_val = max(dim - 1, 1)
    cdef double [:] grid = numpy.linspace(0, 1, order + 2)[1:-1]
    for dim_ in range(max_val):
        vdc_samples = vdc.create_van_der_corput_samples(indices, number_base=primes[dim_])
        for j in prange(order, nogil = True):
            out[dim_,j] = vdc_samples[j]
    if dim > 1:
        grid = numpy.linspace(0, 1, order + 2)[1:-1]
        for k in prange(order, nogil = True):
            out[dim-1, k] = grid[k]
    return np.asarray(out)