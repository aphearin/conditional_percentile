"""
"""
import numpy as np
from conditional_percentile_kernels import calculate_percentile_loop
from halotools.utils import unsorting_indices


def conditional_percentile_function(property1, property2, num_window=500):
    raise NotImplementedError

    idx_property1_sorted = np.argsort(-property1)
    sorted_property2 = property2[idx_property1_sorted]

    cdf_value_table_init = sorted_property2[-num_window:]
    remaining_cdf_values = sorted_property2[:-num_window]

    idx_init_values_sorted = np.argsort(cdf_value_table_init)

    correspondence_indices = np.arange(0, num_window)

    calculate_percentile_loop(cdf_value_table_init, correspondence_indices, remaining_cdf_values)
