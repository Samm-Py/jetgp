import numpy as np
import os

os.chdir('../../modules')
DIM_MAX = 1111
LOG_MAX = 30
POLY = np.load('poly.npy')
SOURCE_SAMPLES = np.load('source_samples.npy')
