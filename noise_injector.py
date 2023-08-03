# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains functions for injecting noise.

Functions in the module must adhere to the following signature:
def function(
    data: NDArray[Shape["*,*"], Float], random_state: int, **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*,*"], Float]
"""

from typing import Any, Dict, Callable, Mapping
from nptyping import Float, NDArray, Shape

import numpy as np
import scipy.stats

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_valid_injectors() -> Dict[
    str,
    Callable[
        [NDArray[Shape["*,*"], Float], int, Mapping[str, Any]], NDArray[Shape["*,*"], Float]
    ]
]:
    """
    Make a dictionary of valid injectors that are in this module.

    Returns
    -------
    dict of {str: callable}
        A new dictionary of valid injectors as defined in this module.
    """
    valid_injectors = {
        "cauchy": inject_cauchy,
        "gauss": inject_gauss,
        "gmm": inject_gmm,
        "laplace": inject_laplace,
        "mixed": inject_mixed,
        "none": inject_none,
    }

    return valid_injectors

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def inject_none(
    data: NDArray[Shape["*,*"], Float], random_state: int, **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*,*"], Float]:
    # pylint: disable=unused-argument
    """
    Pass the data through.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The time series data to inject noise into.
    random_state : int
        The state to use for rng.
    kwargs : dict of {str: any}
        Not used.

    Returns
    -------
    numpy.ndarray, (n_timestep, n_series)
        The noisy time series data.
    """
    return data

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def inject_cauchy(
    data: NDArray[Shape["*,*"], Float], random_state: int, **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*,*"], Float]:
    # pylint: disable=too-many-locals
    """
    Inject additive Cauchy noise into the `data`.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_series)
        The time series data to inject noise into.
    random_state : int
        The state to use for rng.
    kwargs : dict of {str: any}
        base_kv_partials : numpy.ndarray of float, (n_meters,)
            The partial base KV for each meter. This should be a percentage of the original base kv.
        channel_map : dict of {str: list of int}
            The list of channel indices for each meter. These mappings are used to fuse multiple
            meter channels. Each item in the mapping will be {meter_name: list of channel indices},
            where the indices are the columns in the returned data.
        percent : float
            The probability of observing the max magnitude via Cauchy distribution.

    Returns
    -------
    numpy.ndarray, (n_timestep, n_series)
        The noisy time series data.
    """
    base_kv_partials = kwargs["base_kv_partials"]
    channel_map = kwargs["channel_map"]
    percent = kwargs["percent"]

    max_magnitudes = []
    for (i, (_, channel_indices)) in enumerate(list(channel_map.items())):
        base_kv_partial = base_kv_partials[i]
        max_magnitude = 1000 * base_kv_partial

        for _ in range(len(channel_indices)):
            max_magnitudes.append(max_magnitude)

    scales = np.abs(max_magnitudes / np.tan((np.pi / 2) * (percent - 3)))

    # Each element of `scales` will be applied to its respective column.
    cauchy = scipy.stats.cauchy.rvs(
        loc=0, scale=scales, size=data.shape, random_state=random_state
    )

    # Ensure that the impulses never exceed `max_magnitude`.
    for (i, magnitude) in enumerate(max_magnitudes):
        high_indices = np.where(cauchy[:, i] > magnitude)[0]
        low_indices = np.where(cauchy[:, i] < -magnitude)[0]

        cauchy[high_indices, i] = magnitude
        cauchy[low_indices, i] = -magnitude

    data_n = data + cauchy

    return data_n

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def inject_gmm(
    data: NDArray[Shape["*,*"], Float], random_state: int, **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*,*"], Float]:
    # pylint: disable=unused-argument
    """
    Inject additive noise using a Gaussian Mixture Model (GMM).

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The time series data to inject noise into.
    random_state : int
        The state to use for rng.
    kwargs : dict of {str: any}
        centers_list : list of [list of float]
            The centers of the Gaussians for each time series. Lists are used to allow for varying
            numbers of distributions for each series.
        percents_list : list of [list of float]
            The percentage of noise to add for each Gaussian for each time series. Lists are used to
            allow for varying numbers of distributions for each series.

    Returns
    -------
    numpy.ndarray, (n_timestep, n_series)
        The noisy time series data.
    """
    centers_list = kwargs["centers_list"]
    percents_list = kwargs["percents_list"]

    rng = np.random.default_rng(seed=random_state)
    gmm = np.zeros(shape=data.shape, dtype=float)

    for (i, (centers, percents)) in enumerate(zip(centers_list, percents_list)):
        for (center, percent) in zip(centers, percents):
            random_state_local = int(rng.integers(np.iinfo(np.int32).max))

            gmm[:, i] += scipy.stats.norm.rvs(
                loc=center, scale=(data[:, i] * percent) / 3, size=data.shape[0],
                random_state=random_state_local
            )

    data_n = data + gmm

    return data_n

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def inject_gauss(
    data: NDArray[Shape["*,*"], Float], random_state: int, **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*,*"], Float]:
    # pylint: disable=unused-argument
    """
    Inject additive Gaussian noise into the `data`.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The time series data to inject noise into.
    random_state : int
        The state to use for rng.
    kwargs : dict of {str: any}
        percent : float
            The percentage of Gaussian noise to add.

    Returns
    -------
    numpy.ndarray, (n_timestep, n_series)
        The noisy time series data.
    """
    percent = kwargs["percent"]

    gauss = scipy.stats.norm.rvs(
        loc=0, scale=(data * percent) / 3, size=data.shape, random_state=random_state
    )

    data_n = data + gauss

    return data_n

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def inject_laplace(
    data: NDArray[Shape["*,*"], Float], random_state: int, **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*,*"], Float]:
    # pylint: disable=unused-argument
    """
    Inject additive Laplacian noise into the `data`.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The time series data to inject noise into.
    random_state : int
        The state to use for rng.
    kwargs : dict of {str: any}
        percent : float
            The percentage of Laplacian noise to add.

    Returns
    -------
    numpy.ndarray, (n_timestep, n_series)
        The noisy time series data.
    """
    percent = kwargs["percent"]

    laplace = scipy.stats.laplace.rvs(
        loc=0, scale=(data * percent) / 3, size=data.shape, random_state=random_state
    )

    data_n = data + laplace

    return data_n

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def inject_mixed(
    data: NDArray[Shape["*,*"], Float], random_state: int, **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*,*"], Float]:
    # pylint: disable=unused-argument
    """
    Inject additive mixed Gaussian and Cauchy noise into the `data`.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The time series data to inject noise into.
    random_state : int
        The state to use for rng.
    kwargs : dict of {str: any}
        base_kv_partials : numpy.ndarray of float, (n_meters,)
            The partial base KV for each meter. This should be a percentage of the original base kv.
        channel_map : dict of {str: list of int}
            The list of channel indices for each meter. These mappings are used to fuse multiple
            meter channels. Each item in the mapping will be {meter_name: list of channel indices},
            where the indices are the columns in the returned data.
        percent_cauchy : float
            The probability of observing the max magnitude via Cauchy distribution.
        percent_gauss : float
            The percentage of Gaussian noise to add.

    Returns
    -------
    numpy.ndarray, (n_timestep, n_series)
        The noisy time series data.
    """
    percent_gauss = kwargs["percent_gauss"]

    temp = kwargs.copy()
    temp["percent"] = kwargs["percent_cauchy"]

    cauchy = inject_cauchy(
        data=np.zeros(shape=data.shape, dtype=float), random_state=random_state, **temp
    )

    gauss = scipy.stats.norm.rvs(
        loc=0, scale=(data * percent_gauss) / 3, size=data.shape, random_state=random_state
    )

    data_n = data + cauchy + gauss

    return data_n
