"""
"""
from time import time
import pytest
import numpy as np
from bisect import bisect_left as python_bisect_left
from conditional_percentile_kernels import exposed_bisect_left as cython_bisect_left
from conditional_percentile import rank_order_function, conditional_window_ranks
from astropy.utils.misc import NumpyRNGContext


fixed_seed = 43


def python_insert_pop(arr0, idx_in, value_in, idx_out):
    arr = np.copy(arr0).tolist()
    if idx_in < idx_out:
        arr.insert(idx_in, value_in)
        arr.pop(idx_out+1)
    elif idx_in == idx_out:
        arr[idx_in] = value_in
    else:
        arr.insert(idx_in+1, value_in)
        arr.pop(idx_out)
    return np.array(arr, dtype=int)


def test_bisect_left():
    num_tests = int(1e3)
    num_arr = 11*13
    for i in range(num_tests):
        with NumpyRNGContext(i):
            arr = np.sort(np.random.random(num_arr))
            x = np.random.rand()
        idx_cython = cython_bisect_left(arr, x)
        idx_python = python_bisect_left(arr, x)
        assert idx_cython == idx_python


def test_conditional_window_ranks1a():
    npts = int(5e4)
    property1 = np.linspace(1, 0, npts)

    with NumpyRNGContext(fixed_seed):
        property2 = np.random.rand(npts)

    num_window = 101
    result = conditional_window_ranks(property1, property2, num_window=num_window,
            endpoint_fill_value=-1)

    assert np.all(result[:num_window/2] == -1)
    assert np.all(result[-num_window/2+1:] == -1)
    assert not np.any(result[num_window/2: -num_window/2+1] == -1)


def test_conditional_window_ranks1b():
    npts = int(5e4)
    with NumpyRNGContext(fixed_seed):
        property1 = np.random.rand(npts)
        property2 = np.random.rand(npts)

    num_window = 101
    result = conditional_window_ranks(property1, property2, num_window=num_window,
            endpoint_fill_value=-1)

    idx_property1_sorted = np.argsort(-property1)
    sorted_result = result[idx_property1_sorted]

    assert np.all(sorted_result[:num_window/2] == -1)
    assert np.all(sorted_result[-num_window/2+1:] == -1)
    assert not np.any(sorted_result[num_window/2: -num_window/2+1] == -1)


def test_conditional_window_ranks2():
    npts = int(5e4)
    property1 = np.linspace(1, 0, npts)

    with NumpyRNGContext(fixed_seed):
        property2 = np.random.rand(npts)

    num_window = 101
    result = conditional_window_ranks(property1, property2, num_window=num_window,
            endpoint_fill_value='auto')
    msg1 = "Minimum value of conditional_window_ranks must be non-negative"
    assert result.min() >= 0, msg1
    msg2 = "Minimum value of conditional_window_ranks cannot be exceed num_window"
    assert result.max() <= num_window, msg2


@pytest.mark.xfail
def test_conditional_window_ranks_sensible_quantiles():
    npts = int(5e4)
    property1 = np.linspace(1, 0, npts)
    with NumpyRNGContext(fixed_seed):
        property2 = np.random.rand(npts)

    num_window = 501
    result = conditional_window_ranks(property1, property2, num_window=num_window,
            endpoint_fill_value='auto')
    median_result = np.median(result)
    correct_median = num_window/2.
    assert np.allclose(median_result, correct_median, rtol=0.2)


def test_rank_order_function1():
    x = [0.1, 0.95, 0.5, -100]
    result = rank_order_function(x)
    correct_result = [1, 3, 2, 0]
    assert np.all(result == correct_result)


def test_conditional_window_ranks_speed():
    npts = int(1e6)
    with NumpyRNGContext(fixed_seed):
        property1 = np.random.rand(npts)
        property2 = np.random.rand(npts)

    num_window = 1001
    start = time()
    result = conditional_window_ranks(property1, property2, num_window=num_window,
            endpoint_fill_value='auto')
    end = time()
    runtime = end - start
    assert runtime < 5, "Runtime for 10^6 points should be less than 5 seconds"

