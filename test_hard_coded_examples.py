"""
"""
import numpy as np
from conditional_percentile import conditional_window_ranks
from conditional_percentile_kernels import exposed_correspondence_indices_update


fixed_seed = 43


def test1():
    property1 = np.array((5., 4, 3, 2, 1, 0))
    property2 = np.array((5., 4, 3, 2, 1, 0))
    num_window = 3
    endpoint_fill_value = -9
    result = conditional_window_ranks(property1, property2, num_window, endpoint_fill_value)
    assert np.all(result == [-9, 1, 1, 1, 1, -9])


def test2():
    correspondence_indices = np.array([2, 1, 0])
    n = len(correspondence_indices)
    idx_in, idx_out = 0, 0
    result = np.array(exposed_correspondence_indices_update(
                correspondence_indices, n, idx_in, idx_out))
    assert np.all(result == [0, 2, 1])
