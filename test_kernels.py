"""
"""
import pytest
import numpy as np
from bisect import bisect_left as python_bisect_left
from conditional_percentile_kernels import exposed_bisect_left as cython_bisect_left
from conditional_percentile_kernels import calculate_percentile_loop
from conditional_percentile import rank_order_function, conditional_window_ranks


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
        arr = np.sort(np.random.random(num_arr))
        x = np.random.rand()
        idx_cython = cython_bisect_left(arr, x)
        idx_python = python_bisect_left(arr, x)
        assert idx_cython == idx_python


def test_conditional_window_ranks1():
    npts = 1000
    property1 = np.random.rand(npts)
    property2 = np.random.rand(npts)

    num_window = 101
    result = conditional_window_ranks(property1, property2, num_window=num_window,
            endpoint_fill_value=-1)

    assert np.all(result[:num_window/2] == -1)
    assert np.all(result[-num_window/2+1:] == -1)
    assert not np.any(result[num_window/2: -num_window/2+1] == -1)


@pytest.mark.xfail
def test_conditional_window_ranks2():
    npts = 1000
    property1 = np.random.rand(npts)
    property2 = np.random.rand(npts)

    num_window = 101
    result = conditional_window_ranks(property1, property2, num_window=num_window,
            endpoint_fill_value='auto')
    assert result.min() >= 0
    assert result.max() <= num_window - 1


def test_rank_order_function1():
    x = [0.1, 0.95, 0.5, -100]
    result = rank_order_function(x)
    correct_result = [1, 3, 2, 0]
    assert np.all(result == correct_result)
