# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains functions for aggregating multiple channels of a meter into a single channel.

Functions in the module must adhere to the following signature:
def function(
    data: NDArray[Shape["*,*"], Float], channel_map: Mapping[str, Sequence[int]],
    **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*,*"], Float]
"""

from typing import Any, Dict, Callable, Mapping, Sequence
from nptyping import Float, NDArray, Shape

import numpy as np

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_valid_aggregators() -> Dict[
    str, Callable[
        [NDArray[Shape["*,*"], Float], Mapping[str, Sequence[int]], Mapping[str, Any]],
        NDArray[Shape["*,*"], Float]
    ]
]:
    """
    Make a dictionary of valid aggregators that are in this module.

    Returns
    -------
    dict of {str: callable}
        A new dictionary of valid aggregators as defined in this module.
    """
    valid_aggregators = {
        "cross_func": run_cross_func,
        "cross_time": run_cross_time,
        "mean": run_mean
    }

    return valid_aggregators

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_cross_func(
    data: NDArray[Shape["*,*"], Float], channel_map: Mapping[str, Sequence[int]],
    **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*,*"], Float]:
    # pylint: disable=too-many-locals,
    """
    Aggregate the data by using the cross product between the channels and some new data generated
    by applying a nonlinear function to the original channels.

    The output of the function should generate vectors that are not collinear with the original
    data. This can be easily achieved by using a nonlinear function.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_meter_channel)
        The data to aggregate.
    channel_map : dict of {str: list of int}
        The list of channel indices for each meter. These mappings are used to aggregate multiple
        meter channels. Each item in the mapping will be {meter_name: list of channel indices},
        where the indices are the columns in the returned data.
    kwargs : dict of {str: any}
        run_func : callable, default=lambda data: np.exp(-data**2)
            The function to apply to the measurment channels. This should be a nonlinear function.
            The default function is the standard Gaussian kernel. It should be of the form:
            (data: NDArray[Shape["*,*"], Float]) -> NDArray[Shape["*,*"], Float]
        will_flip : bool, default=True
            Whether to flip the direction of the single channel meters in time. This can allow for
            more distinct seperation between phases.

    Returns
    -------
    numpy.ndarray of float, (n_timestep, n_meter)
        The new aggregated data.
    """
    run_func = kwargs.get("run_func", lambda data: np.exp(-data**2))
    will_flip = kwargs.get("will_flip", True)

    result = np.zeros(shape=(data.shape[0], len(channel_map)), dtype=float)

    # Estimate the expected channel. This channel is used as an addition piece of information for
    # the single channel case.
    expected_channel = np.mean(data, axis=1)

    for (i, (_, channel_indices)) in enumerate(list(channel_map.items())):
        data_channel = data[:, channel_indices]

        series = None

        if len(channel_indices) == 1:
            data_x = np.column_stack([data_channel, expected_channel])
            data_f = np.flip(data_x, axis=0) if will_flip else data_x
            data_y = run_func(data=data_f)
            series = np.cross(data_f, data_y)

        elif len(channel_indices) == 2:
            data_y = run_func(data=data_channel)
            series = np.cross(data_channel, data_y)

        elif len(channel_indices) == 3:
            data_y = run_func(data=data_channel)
            data_cross = np.cross(data_channel, data_y)
            series = np.linalg.norm(data_cross, ord=2, axis=1)

        else:
            raise ValueError("Only up to 3 meter channels is supported.")

        result[:, i] = series.reshape(-1,)

    return result

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_cross_time(
    data: NDArray[Shape["*,*"], Float], channel_map: Mapping[str, Sequence[int]],
    **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*,*"], Float]:
    # pylint: disable=too-many-locals,
    """
    Aggregate the data by using the cross product between the vectors v(t) and v(t - 1), where v(*)
    is a row vector in the channel data.

    Since the cross product is occuring between two adjacent timesteps, there will be an inherent
    high-pass filtering equivalent to a first-order time difference transformation (first-order
    difference filter). If this filtering is not desired, then `run_cross_func` should be used
    instead.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_meter_channel)
        The data to aggregate.
    channel_map : dict of {str: list of int}
        The list of channel indices for each meter. These mappings are used to aggregate multiple
        meter channels. Each item in the mapping will be {meter_name: list of channel indices},
        where the indices are the columns in the returned data.
    kwargs : dict of {str: any}
        will_flip : bool, default=True
            Whether to flip the direction of the single channel meters in time. This can allow for
            more distinct seperation between phases.

    Returns
    -------
    numpy.ndarray of float, (n_timestep - 1, n_meter)
        The new aggregated data.
    """
    will_flip = kwargs.get("will_flip", True)

    result = np.zeros(shape=(data.shape[0] - 1, len(channel_map)), dtype=float)

    # Estimate the expected channel. This channel is use as an addition piece of information for the
    # single channel case.
    expected_channel = np.mean(data, axis=1)

    # Aggregate the channels.
    for (i, (_, channel_indices)) in enumerate(list(channel_map.items())):
        data_channel = data[:, channel_indices]

        series = None

        if len(channel_indices) == 1:
            data_x = np.column_stack([data_channel, expected_channel])
            data_f = np.flip(data_x, axis=0) if will_flip else data_x
            series = np.cross(data_f[1:, :], data_f[:-1, :])

        elif len(channel_indices) == 2:
            series = np.cross(data_channel[1:, :], data_channel[:-1, :])

        elif len(channel_indices) == 3:
            data_cross = np.cross(data_channel[1:, :], data_channel[:-1, :])
            series = np.linalg.norm(data_cross, ord=2, axis=1)

        else:
            raise ValueError("Only up to 3 meter channels is supported.")

        result[:, i] = series.reshape(-1,)

    return result

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_mean(
    data: NDArray[Shape["*,*"], Float], channel_map: Mapping[str, Sequence[int]],
    **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*,*"], Float]:
    # pylint: disable=unused-argument
    """
    Aggregate the data by using the mean of the channels.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_meter_channel)
        The data to aggregate.
    channel_map : dict of {str: list of int}
        The list of channel indices for each meter. These mappings are used to aggregate multiple
        meter channels. Each item in the mapping will be {meter_name: list of channel indices},
        where the indices are the columns in the returned data.
    kwargs : dict of {str: any}
        Not used.

    Returns
    -------
    numpy.ndarray of float, (n_timestep, n_meter)
        The new aggregated data.
    """
    result = np.zeros(shape=(data.shape[0], len(channel_map)), dtype=float)

    for (i, (_, channel_indices)) in enumerate(list(channel_map.items())):
        data_channel = data[:, channel_indices]
        series = np.mean(data_channel, axis=1)

        result[:, i] = series.reshape(-1,)

    return result
