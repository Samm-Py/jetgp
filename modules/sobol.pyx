# cython: wraparound=False
# cython: boundscheck=False
# cython: profile=True
# cython: initializedcheck=False
import cython
cimport cython
import  numpy as np
cimport numpy as np
cimport libc.math as cmath # Import c- math libraries.
import math 
from SOBOL_SEQUENCES import DIM_MAX, LOG_MAX, POLY, SOURCE_SAMPLES
from cython.parallel import prange
cimport openmp
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)

def create_sobol_samples(int order, int dim, int seed):
    
    """
    Generates samples from the Sobol sequence.

    Sobol sequences (also called LP_T sequences or (t, s) sequences in base 2)
    are an example of quasi-random low-discrepancy sequences. They were first
    introduced by the Russian mathematician Ilya M. Sobol in 1967.

    These sequences use a base of two to form successively finer uniform
    partitions of the unit interval and then reorder the coordinates in each
    dimension.

    Args:
        order (int):
            Number of unique samples to generate.
        dim (int):
            Number of spacial dimensions. Must satisfy ``0 < dim < 1111``.
        seed (int):
            Starting seed. Non-positive values are treated as 1. If omitted,
            consecutive samples are used.

    Returns:
        (numpy.ndarray):
            Quasi-random vector with ``shape == (dim, order)``.

    Notes:
        Implementation based on the initial work of Sobol
        :cite:`sobol_distribution_1967`. This implementation is based on the
        work of `Burkardt
        <https://people.sc.fsu.edu/~jburkardt/py_src/sobol/sobol.html>`_.

    Examples:
        >>> distribution = chaospy.Iid(chaospy.Uniform(0, 1), 2)
        >>> samples = distribution.sample(3, rule="sobol")
        >>> samples.round(4)
        array([[0.5 , 0.75, 0.25],
                [0.5 , 0.25, 0.75]])
        >>> samples = distribution.sample(4, rule="sobol")
        >>> samples.round(4)
        array([[0.5  , 0.75 , 0.25 , 0.375],
                [0.5  , 0.25 , 0.75 , 0.375]])

    """

    # Initialize row 1 of V.
    samples = np.copy(SOURCE_SAMPLES)
    samples[0, 0:LOG_MAX] = 1

    # Initialize the remaining rows of V.
    for idx in range(1, dim):

        # The bits of the integer POLY(I) gives the form of polynomial:
        degree = int(np.log2(POLY[idx]))

        # Expand this bit pattern to separate components:
        includ = np.array([val == "1" for val in bin(POLY[idx])[-degree:]])

        # Calculate the remaining elements of row as explained
        # in Bratley and Fox, section 2.
        for idy in range(degree + 1, LOG_MAX + 1):
            newv = samples[idx, idy - degree - 1].item()
            base = 1
            for idz in range(1, degree + 1):
                base *= 2
                if includ[idz - 1]:
                    newv = newv ^ base * samples[idx, idy - idz - 1].item()
            samples[idx, idy - 1] = newv

    samples = samples[:dim]

    # Multiply columns of V by appropriate power of 2.
    samples *= 2 ** (np.arange(LOG_MAX, 0, -1, dtype=int))


    # RECIPD is 1/(common denominator of the elements in V).
    recipd = 0.5 ** (LOG_MAX + 1)
    seed = int(seed) if seed > 1 else 1
    
    cdef np.ndarray [long, ndim = 2] lastq = np.zeros((dim,order), dtype = 'int64')
    # Calculate the new components of QUASI.
    cdef double[:,:] quasi = np.empty((dim, order))
    cdef int seed_c = seed
    cdef int lowbit_c = 0
    cdef int c 
    cdef int ndarray
    cdef int idx_c
    cdef int n
    cdef double recipd_c = recipd
    cdef int j
    cdef np.ndarray [long, ndim = 2] samples_c = samples
    cdef int seed_
    cdef openmp.omp_lock_t lock
    openmp.omp_init_lock(&lock)
    for seed_ in range(seed_c):
        if (seed_)%2== 0:
            lowbit_c = 0
        else:
            c = 0
            n = seed_ + 1
            while ((n % 2) == 0):
                n = n//2
                c = c+ 1
            lowbit_c = c
        lastq[:,0] = lastq[:,0] ^ samples_c[:, lowbit_c]
    for idx_c in range(0,order-1):
        if (seed_c+idx_c)%2== 0:
            lowbit_c = 0
        else:
            c = 0
            n = seed_c+idx_c + 1
            while ((n % 2) == 0):
                n = n//2
                c = c+ 1
            lowbit_c = c
        for j in range(0,dim):
            # quasi[j, idx_c] = lastq[j] * recipd_c
            lastq[j, idx_c+1] = lastq[j, idx_c] ^ samples_c[j, lowbit_c]
    for idx_c in prange(order, nogil = True):
         for j in range(0,dim):
              quasi[j, idx_c] = lastq[j, idx_c] * recipd_c
             # lastq[j, idx_c] = lastq[j, idx_c] ^ samples_c[j, lowbit_c]
                
        

    return np.asarray(quasi)