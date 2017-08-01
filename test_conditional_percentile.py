"""
"""
import numpy as np


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
