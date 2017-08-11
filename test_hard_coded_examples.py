"""
"""
import numpy as np
from conditional_percentile import conditional_window_ranks


fixed_seed = 43


def test1():
    property1 = np.array((5., 4, 3, 2, 1, 0))
    property2 = np.array((5., 4, 3, 2, 1, 0))
    num_window = 3
    endpoint_fill_value = -9
    result = conditional_window_ranks(property1, property2, num_window, endpoint_fill_value)
    assert np.all(result == [-9, 1, 1, 1, 1, -9])
