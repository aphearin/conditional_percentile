"""
"""
import numpy as np
from conditional_percentile_kernels import calculate_percentile_loop
from halotools.utils import unsorting_indices


def conditional_window_ranks(property1, property2, num_window=501,
            endpoint_fill_value='auto'):
    """
    """
    property1, property2, num_window, endpoint_fill_value = _process_args(
            property1, property2, num_window, endpoint_fill_value)

    idx_property1_sorted = np.argsort(-property1)
    sorted_property2 = property2[idx_property1_sorted]

    correspondence_indices_init, cdf_value_table_init = get_initial_windows(
            sorted_property2, num_window)

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


def _process_args(property1, property2, num_window, endpoint_fill_value):
    """
    """
    property1 = np.atleast_1d(property1)
    property2 = np.atleast_1d(property2)

    #  Bounds checks
    assert num_window % 2 == 1, "num_window = {0} must be odd".format(num_window)
    msg1 = "``property1`` and ``property2`` must be 1-d arrays"
    assert len(property1.shape) == len(property2.shape) == 1, msg1

    msg2 = "``property1`` and ``property2`` must have same shape"
    assert property1.shape == property2.shape, msg2

    msg3 = "Length of ``property1`` = {0} must exceed num_window = {1}"
    assert property1.shape[0] > num_window, msg3.format(property1.shape[0], num_window)

    return property1, property2, num_window, endpoint_fill_value


def get_initial_windows(sorted_property2, num_window):
    """
    """
    correspondence_indices_init = rank_order_function(sorted_property2[:num_window])
    cdf_value_table_init = np.sort(sorted_property2[:num_window])

    return correspondence_indices_init, cdf_value_table_init

