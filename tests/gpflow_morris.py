"""
Benchmark: GPflow DEGP on the Morris function (20D)
Custom derivative-enhanced SE kernel with analytical derivative blocks.
CPU only, single thread for fair comparison.

Follows the methodology of Erickson et al. (2018) with sample sizes
n = d, 5d, 10d, using 5 macroreplicates.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

import time
import sys
sys.path.insert(0, '.')

from benchmark_functions import morris, morris_gradient
from gpflow_degp import run_gpflow_benchmark

DIM = 20
SAMPLE_SIZES = [DIM, 5 * DIM, 10 * DIM]

if __name__ == "__main__":
    run_gpflow_benchmark(
        func=morris,
        grad_func=morris_gradient,
        dim=DIM,
        function_name="morris",
        sample_sizes=SAMPLE_SIZES,
    )
