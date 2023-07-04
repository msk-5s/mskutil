# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains functions for scaling data.

Functions in the module must adhere to the following signature:
def function(
    data: NDArray[Shape["*,*"], Float], **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*,*"], Float]
"""

from typing import Any, Dict, Callable, Mapping
from nptyping import Float, NDArray, Shape

import sklearn.preprocessing

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_valid_scalers() -> Dict[
    str, Callable[[NDArray[Shape["*,*"], Float], Mapping[str, Any]], NDArray[Shape["*,*"], Float]]
]:
    """
    Make a dictionary of valid scalers that are in this module.

    Returns
    -------
    dict of {str: callable}
        A new dictionary of valid scalers as defined in this module.
    """
    valid_scalers = {
        "none": run_none,
        "normalize": run_normalize,
        "standardize": run_standardize,
    }

    return valid_scalers

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_none(
    data: NDArray[Shape["*,*"], Float], **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*,*"], Float]:
    # pylint: disable=unused-argument
    """
    Pass the data through with no scaling.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The data to reduce.
    kwargs : dict of {str: any}
        Not used.

    Returns
    -------
    Result
        The result of the reducer.
    """
    return data

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_normalize(
    data: NDArray[Shape["*,*"], Float], **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*,*"], Float]:
    # pylint: disable=unused-argument
    """
    Pass the data through with no scaling.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The data to reduce.
    kwargs : dict of {str: any}
        See scikit-learn's documentation for `MinMaxScaler`.

    Returns
    -------
    Result
        The result of the reducer.
    """
    result = sklearn.preprocessing.MinMaxScaler(**kwargs).fit_transform(data)

    return result

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_standardize(
    data: NDArray[Shape["*,*"], Float], **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*,*"], Float]:
    # pylint: disable=unused-argument
    """
    Pass the data through with no scaling.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The data to reduce.
    kwargs : dict of {str: any}
        See scikit-learn's documentation for `StandardScaler`.

    Returns
    -------
    Result
        The result of the reducer.
    """
    result = sklearn.preprocessing.StandardScaler(**kwargs).fit_transform(data)

    return result
