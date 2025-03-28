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
from Sequences.sobol_constants import DIM_MAX, LOG_MAX, POLY, SOURCE_SAMPLES
from cython.parallel import prange

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def create_latin_hypercube_samples(int order, int dim):
   

   cdef np.ndarray [double, ndim = 2] randoms = np.empty((dim, order))
   cdef int dim_
   cdef np.ndarray [long, ndim = 1] perm = np.zeros((order,), dtype = 'int64')

   """
   Latin Hypercube sampling.

   Args:
       order (int):
           The order of the latin hyper-cube. Defines the number of samples.
       dim (int):
           The number of dimensions in the latin hyper-cube.

   Returns (numpy.ndarray):
       Latin hyper-cube with ``shape == (dim, order)``.
   """
   randoms = np.random.random(order * dim).reshape((dim, order))
   for dim_ in range(dim):
       perm = np.random.permutation(order) 
       randoms[dim_] = (perm + randoms[dim_]) / order
   return randoms
