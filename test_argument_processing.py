"""
"""
import numpy as np
from conditional_percentile import get_initial_windows, conditional_window_ranks



# def test1():
#     sorted_property2 = [5., 4, 3, 2, 1, 0]
#     num_window = 3
#     correspondence_indices_init, cdf_value_table_init = get_initial_windows(
#             sorted_property2, num_window)
#     assert np.all(cdf_value_table_init == [3, 4, 5])
#     assert np.all(correspondence_indices_init == [2, 1, 0])


# def test2():
#     sorted_property2 = [0, 1, 2, 5., 4, 3, 2, 1, 0]
#     num_window = 5
#     correspondence_indices_init, cdf_value_table_init = get_initial_windows(
#             sorted_property2, num_window)

#     assert np.all(cdf_value_table_init == (0, 1, 2, 4, 5))
#     assert np.all(correspondence_indices_init == (0, 1, 2, 4, 3))


def test_simple_hard_coded_result1():
    """ Initial property2 array is (0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
    So every rank should be 2, except for the endpoints.
    """
    npts = 10
    property1 = np.arange(npts, 0., -1)
    property2 = np.arange(0., npts)
    num_window = 5
    ranks = conditional_window_ranks(property1, property2, num_window)
    print(ranks)
    assert np.all(ranks == (0, 1, 2, 2, 2, 2, 2, 2, 3, 4))
