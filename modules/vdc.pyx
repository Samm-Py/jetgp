"""Create Van Der Corput low discrepancy sequence samples."""
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
def create_van_der_corput_samples(np.ndarray[np.int32_t, ndim=1] idx, int number_base):
    """
    Create Van Der Corput low discrepancy sequence samples.

    A van der Corput sequence is an example of the simplest one-dimensional
    low-discrepancy sequence over the unit interval; it was first described in
    1935 by the Dutch mathematician J. G. van der Corput. It is constructed by
    reversing the base-n representation of the sequence of natural numbers
    :math:`(1, 2, 3, ...)`.

    In practice, use Halton sequence instead of Van Der Corput, as it is the
    same, but generalized to work in multiple dimensions.

    Args:
        idx (int, numpy.ndarray):
            The index of the sequence. If array is provided, all values in
            array is returned.
        number_base (int):
            The numerical base from where to create the samples from.

    Returns:
        (numpy.ndarray):
            Van der Corput samples.

    Examples:
        #>>> chaospy.create_van_der_corput_samples(range(11), number_base=10)
        #array([0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 , 0.01, 0.11])
        #>>> chaospy.create_van_der_corput_samples(range(8), number_base=2)
        #array([0.5   , 0.25  , 0.75  , 0.125 , 0.625 , 0.375 , 0.875 , 0.0625])

    """
    cdef int number_base_c = number_base

    idx = idx + 1

    cdef double [:] out = np.zeros(len(idx), dtype=float)
    cdef np.ndarray [double, ndim = 1] base = np.zeros((len(idx),), dtype = 'float64') + number_base
    cdef int i 

    
    cdef int [: :] active = np.ones(len(idx), dtype='int32')
    cdef int flag = 1
    while flag==1:
        for i in prange(active.shape[0], nogil = True):
            if active[i]!=0:
                out[i] = out[i] + ((idx[i] % number_base_c) / base[i])
                idx[i] = idx[i]// number_base_c
                base[i] = base[i]* number_base_c
                if idx[i] > 0:
                    active[i] = idx[i]
                else:
                    active[i] = 0
        if(np.any(np.asarray(active)>0)):
            pass
        else:
            flag = 0
    return np.asarray(out)


