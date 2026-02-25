import numpy as np


def safe_value(val):
    """This function ensures that a value that we put into a pandas DataFrame
    is safe, i.e doesn't have mismatched datatypes.

    :param val: value to store in df
    :type val: string or float or list or array
    :return: safe value
    """
    if isinstance(val, np.ndarray):
        # Ensure native byte order
        if hasattr(val, "dtype") and val.dtype.byteorder not in ("=", "|"):
            val = val.astype(val.dtype.newbyteorder("="))
        # If array has one element, convert to float
        if val.size == 1:
            return float(val)
    elif isinstance(val, np.generic):
        return float(val)
    return val
