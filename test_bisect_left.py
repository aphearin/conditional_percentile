"""
"""
import numpy as np
from bisect import bisect_left as python_bisect_left
from conditional_percentile_kernels import exposed_bisect_left as cython_bisect_left
from astropy.utils.misc import NumpyRNGContext


fixed_seed = 43


def test_bisect_left1():
    num_tests = int(1e3)
    num_arr = 11*13
    for i in range(num_tests):
        with NumpyRNGContext(i):
            arr = np.sort(np.random.random(num_arr))
            x = np.random.rand()
        idx_cython = cython_bisect_left(arr, x)
        idx_python = python_bisect_left(arr, x)
        assert idx_cython == idx_python
