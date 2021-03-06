"""
"""
import numpy as np
from libc.math cimport fmod
cimport cython


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef long bisect_left(double* arr, double value, long n):
    """ Return the index where to insert ``value`` in list ``arr`` of length ``n``,
    assuming ``arr`` is sorted.
    """
    cdef long ifirst_subarr = 0
    cdef long ilast_subarr = n
    cdef long imid_subarr

    while ilast_subarr-ifirst_subarr >= 2:
        imid_subarr = (ifirst_subarr + ilast_subarr)/2
        if value > arr[imid_subarr]:
            ifirst_subarr = imid_subarr
        else:
            ilast_subarr = imid_subarr
    if value > arr[ifirst_subarr]:
        return ilast_subarr
    else:
        return ifirst_subarr


def exposed_bisect_left(double[:] arr, double value):
    """
    """
    cdef long n = arr.shape[0]
    return bisect_left(&arr[0], value, n)


cdef void correspondence_indices_update(long* arr, long n, long idx_in, long idx_out):
    """
    The ``correspondence_indices_update`` pops out the last value of ``arr`` and
    inserts a new element at the beginning, updating the other values to help maintain
    the order of the sorting window.

    The array ``arr`` stores the indices that are to be popped out of the
    ``property2`` array input to the calculate_percentile_loop function,
    sorted such that the last element of ``arr`` stores the index of
    the next elemement of ``property2`` to be popped out.

    """
    cdef long i
    cdef long increment = 0

    for i in range(1, n-1):
        arr[i] = arr[i-1]

    if idx_in < idx_out:
        for i in range(idx_in, idx_out):
            arr[i] += 1
    elif idx_in > idx_out + 1:
        for i in range(idx_out+1, idx_in):
            arr[i] -= 1

    arr[0] = idx_in
    if idx_in > idx_out:
        arr[0] -= 1


cdef void cython_insert_pop(double* arr, long idx_in, long idx_out, double value_in, long n):
    """ Pop out the value stored in index ``idx_out`` of array ``arr``,
    and insert ``value_in`` at index ``idx_in`` of the final array.
    """
    cdef int i

    if idx_in <= idx_out:
        for i in range(idx_out-1, idx_in-1, -1):
            arr[i+1] = arr[i]
    else:
        for i in range(idx_out, idx_in):
            arr[i] = arr[i+1]
    arr[idx_in] = value_in


cdef long update_tables(double* cdf_value_table, long* correspondence_indices,
            double cdf_value_in, long n):
    """ Insert ``cdf_value_in`` into ``cdf_value_table``, maintaining its sorted order.
    Remove the value from ``cdf_value_table`` corresponding to
    the most massive element, whose index is stored in the
    final element in ``correspondence_indices``. Update the ``correspondence_indices``
    array after popping out the most massive element, so that the next-most-massive
    element now appears at the end of ``correspondence_indices``.
    """
    cdef long idx_in = bisect_left(&cdf_value_table[0], cdf_value_in, n)
    cdef long idx_out = correspondence_indices[n-1]

    correspondence_indices_update(&correspondence_indices[0], n, idx_in, idx_out)
    cython_insert_pop(&cdf_value_table[0], idx_in, idx_out, cdf_value_in, n)
    return correspondence_indices[n/2]


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def calculate_percentile_loop(double[:] cdf_value_table, long[:] correspondence_indices,
            double[:] cdf_values):
    """
    """
    cdef long nwindow = cdf_value_table.shape[0]
    cdef long num_loop = cdf_values.shape[0]

    assert nwindow % 2 == 1, "Window size = len(cdf_value_table) = {0} must be odd".format(nwindow)
    cdef long i, new_idx
    cdef double new_cdf_value
    cdef long[:] result = np.zeros(num_loop, dtype='i8') - 1

    cdef long ifirst = nwindow/2
    cdef long ilast = num_loop - nwindow/2

    for i in range(ifirst, ilast):
        new_cdf_value = cdf_values[i]
        result[i] = update_tables(&cdf_value_table[0], &correspondence_indices[0], new_cdf_value, nwindow)
    return result




