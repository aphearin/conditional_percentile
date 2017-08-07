"""
"""
import numpy as np
from conditional_percentile_kernels import calculate_percentile_loop
from halotools.utils import unsorting_indices


def conditional_window_ranks(property1, property2, num_window=501,
            endpoint_fill_value='auto'):
    """
    """
    property1 = np.copy(np.atleast_1d(property1))
    property2 = np.copy(np.atleast_1d(property2))

    #  Bounds checks
    assert num_window % 2 == 1, "num_window = {0} must be odd".format(num_window)
    assert property1.shape == property2.shape, "``property1`` and ``property2`` must have same shape"
    assert len(property1.shape) == 1
    msg = "Length of ``property1`` = {0} must exceed num_window = {1}"
    assert property1.shape[0] > num_window, msg.format(property1.shape[0], num_window)

    idx_property1_sorted = np.argsort(-property1)
    sorted_property2 = property2[idx_property1_sorted]

    cdf_value_table_init = sorted_property2[:num_window]
    correspondence_indices_init = rank_order_function(cdf_value_table_init)

    window_ranks = np.array(calculate_percentile_loop(cdf_value_table_init,
            correspondence_indices_init, sorted_property2))

    if endpoint_fill_value == 'auto':
        low_end_result = rank_order_function(sorted_property2[:num_window])
        high_end_result = rank_order_function(sorted_property2[-num_window:])
        window_ranks[:num_window/2] = low_end_result[:num_window/2]
        window_ranks[-num_window/2+1:] = high_end_result[-num_window/2+1:]
    else:
        window_ranks[:num_window/2] = endpoint_fill_value
        window_ranks[-num_window/2+1:] = endpoint_fill_value

    return window_ranks[unsorting_indices(idx_property1_sorted)]


def rank_order_function(arr):
    """
    """
    idx_sorted = np.argsort(arr)
    return unsorting_indices(idx_sorted)

