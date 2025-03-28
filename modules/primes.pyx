"""
Create all primes bellow a certain threshold.

Examples::

    >>> create_primes(1)
    []
    >>> create_primes(2)
    [2]
    >>> create_primes(3)
    [2, 3]
    >>> create_primes(20)
    [2, 3, 5, 7, 11, 13, 17, 19]
"""
import cython
cimport cython
import  numpy as np
cimport numpy as np
cimport libc.math as cmath # Import c- math libraries.
import math 
from Sequences.sobol_constants import DIM_MAX, LOG_MAX, POLY, SOURCE_SAMPLES
from cython.parallel import prange


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def create_primes(int threshold):
    """
    Generate prime values using sieve of Eratosthenes method.

    Args:
        threshold (int):
            The upper bound for the size of the prime values.

    Returns (List[int]):
        All primes from 2 and up to ``threshold``.
    """
    cdef int i, j, p, limit
    cdef int threshold_c = threshold + 1
    
    if threshold < 2:
        return []
    elif threshold == 2:
        return np.array([2], dtype='int32')

    # Array to hold numbers, where True indicates a prime
    cdef np.ndarray[int, ndim=1] is_prime = np.ones(threshold_c, dtype='int32')
    
    # Mark 0 and 1 as non-prime
    is_prime[0] = is_prime[1] = 0
    
    # Loop over the numbers and apply the sieve
    limit = int(threshold ** 0.5) + 1
    for i in range(2, limit):
        if is_prime[i]:
            for j in range(i * i, threshold_c, i):
                is_prime[j] = 0

    # Collect primes
    primes = np.nonzero(is_prime)[0]
    
    return primes