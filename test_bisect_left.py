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


def test_bisect_left_endpoint_behavior1():
    idx = cython_bisect_left(np.array((0, 1., 2)), -1)
    assert idx == 0


def test_bisect_left_endpoint_behavior2():
    idx = cython_bisect_left(np.array((0, 1., 2)), 0)
    assert idx == 0


def test_bisect_right_endpoint_behavior1():
    idx = cython_bisect_left(np.array((0, 1., 2)), 3)
    assert idx == 3


def test_bisect_right_endpoint_behavior2():
    idx = cython_bisect_left(np.array((0, 1., 2)), 2)
    assert idx == 2
