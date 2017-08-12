"""
"""
import numpy as np
from conditional_percentile import get_initial_windows


def test1():
    sorted_property2 = [5., 4, 3, 2, 1, 0]
    num_window = 3
    correspondence_indices_init, cdf_value_table_init = get_initial_windows(
            sorted_property2, num_window)
    assert np.all(cdf_value_table_init == [3, 4, 5])
    assert np.all(correspondence_indices_init == [2, 1, 0])


def test2():
    sorted_property2 = [0, 1, 2, 5., 4, 3, 2, 1, 0]
    num_window = 5
    correspondence_indices_init, cdf_value_table_init = get_initial_windows(
            sorted_property2, num_window)

    assert np.all(cdf_value_table_init == (0, 1, 2, 4, 5))
    assert np.all(correspondence_indices_init == (0, 1, 2, 4, 3))


