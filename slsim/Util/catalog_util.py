import numpy as np
from lenstronomy.Util.param_util import ellipticity2phi_q


def match_source(
    angular_size,
    physical_size,
    e1,
    e2,
    n_sersic,
    processed_catalog,
    max_scale=1,
    match_n_sersic=False,
):
    """This function matches the parameters in source_dict to find a
    corresponding source in a source catalog. The parameters being
    matched are:

    1. physical size: the tolerance starts at 0.5 kPc and increases by 0.2 until at least one match
    2. axis ratio: the tolerance starts at 0.1 and increases by 0.05 until at least one match
    3. n_sersic: if match_n_sersic is True, finally selects the source with the best matching n_sersic

    :param angular_size: desired angular size of the source [arcsec]
    :param physical_size: desired physical size of the source [kpc]
    :param e1: desired eccentricity modulus
    :param e2: desired eccentricity modulus
    :param processed_catalog: the returned object from calling process_catalog()
        See e.g. slsim/Sources/SourceCatalogues HSTCosmosCatalog or CosmosWebCatalog
    :param max_scale: The matched image will be scaled to have the desired angular size. Scaling up
        results in a more pixelated image. This input determines what the maximum up-scale factor is.
    :type max_scale: int or float
    :param match_n_sersic: determines whether to match based off of the sersic index as well.
        Since n_sersic is usually undefined and set to 1 in SLSim, this is set to False by default.
    :type match_n_sersic: bool
    :return: tuple(ndarray, float, float, int)
        This is the raw image matched from the catalog, the scale that the image needs to
        match angular size, the angle of rotation needed to match the desired e1 and e2, and the galaxy ID.
    """


    processed_catalog = processed_catalog[
        angular_size <= processed_catalog["angular_size"].data * max_scale
    ]
    if len(processed_catalog) == 0:
        return None, None, None, None

    # Keep sources within the physical size tolerance, all units in kPc
    size_tol = 0.5
    size_difference = np.abs(
        physical_size - processed_catalog["physical_size"].data
    )
    matched_catalog = processed_catalog[size_difference < size_tol]
    # If no sources, relax the matching condition and try again
    while len(matched_catalog) == 0:
        size_tol += 0.2
        matched_catalog = processed_catalog[size_difference < size_tol]

    phi, q = ellipticity2phi_q(e1, e2)
    # Keep sources within the axis ratio tolerance
    q_tol = 0.1
    q_matched_catalog = matched_catalog[
        np.abs(matched_catalog["axis_ratio"].data - q) <= q_tol
    ]
    # If no sources, relax the tolerance and try again
    while len(q_matched_catalog) == 0:
        q_tol += 0.05
        q_matched_catalog = matched_catalog[
            np.abs(matched_catalog["axis_ratio"].data - q) <= q_tol
        ]

    if match_n_sersic:
        # Select source based off of best matching n_sersic
        index = np.argsort(np.abs(q_matched_catalog["sersic_index"].data - n_sersic))
        matched_source = q_matched_catalog[index][0]
    else:
        # Select source based off of best matching axis ratio
        index = np.argsort(np.abs(q_matched_catalog["axis_ratio"].data - q))
        matched_source = q_matched_catalog[index][0]

    return matched_source


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
