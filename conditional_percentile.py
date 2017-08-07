"""
"""
import numpy as np
from conditional_percentile_kernels import calculate_percentile_loop
from halotools.utils import unsorting_indices


def conditional_percentile_function(property1, property2, num_window=500):
    """
    """
    #  Bounds checks
    assert num_window % 2 == 1, "num_window = {0} must be odd".format(num_window)
    assert property1.shape == property2.shape, "``property1`` and ``property2`` must have same shape"


    idx_property1_sorted = np.argsort(-property1)
    sorted_property2 = property2[idx_property1_sorted]

    cdf_value_table_init = sorted_property2[:num_window]
    remaining_cdf_values = sorted_property2[num_window:]
    correspondence_indices_init = rank_order_function(cdf_value_table_init)

    window_ranks = calculate_percentile_loop(cdf_value_table_init,
            correspondence_indices_init, remaining_cdf_values)


def rank_order_function(arr):
    """
    """
    idx_sorted = np.argsort(arr)
    return unsorting_indices(idx_sorted)

