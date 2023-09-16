# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains functions for linear filtering.

Functions in the module must adhere to the following signature:
def function(
    data: NDArray[Shape["*,*"], Float], **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*,*"], Float]
"""

from typing import Any, Dict, Callable, Mapping
from nptyping import Float, NDArray, Shape

import numpy as np

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_valid_filterers() -> Dict[
    str, Callable[[NDArray[Shape["*,*"], Float], Mapping[str, Any]], NDArray[Shape["*,*"], Float]]
]:
    """
    Make a dictionary of valid filterers that are in this module.

    Returns
    -------
    dict of {str: callable}
        A new dictionary of valid filterers as defined in this module.
    """
    valid_filterers = {
        "butterworth": run_butterworth,
        "difference": run_difference,
        "ideal": run_ideal,
        "none": run_none,
        "unit_magnitude": run_unit_magnitude
    }

    return valid_filterers

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_none(
    data: NDArray[Shape["*,*"], Float], **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*,*"], Float]:
    # pylint: disable=unused-argument
    """
    Apply no filtering.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The time series data to filter.
    kwargs : dict of {str: any}
        Not used.

    Returns
    -------
    numpy.ndarray, (n_timestep, n_series)
        The filtered time series data.
    """
    return data

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_butterworth(
    data: NDArray[Shape["*,*"], Float], **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*,*"], Float]:
    """
    Apply a butterworth filter to each column (time series) of `data`.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The time series data to filter.
    kwargs : dict of {str: any}
        cutoff : float
            The cutoff frequency in cycles per sample.
        order : int
            The order of the butterworth filter.
        filter_type : str, ["lowpass", "highpass"], default="highpass"
            The filter type to use.

    Returns
    -------
    numpy.ndarray, (n_timestep, n_series)
        The filtered time series data.
    """
    cutoff = kwargs["cutoff"]
    order = kwargs["order"]
    filter_type = kwargs.get("filter_type", "highpass")

    valid_types = ["lowpass", "highpass"]

    if filter_type not in valid_types:
        ValueError(f"Invalid filter type: {filter_type} - Valid Types: {valid_types}")

    # We add a small number to the frequencies to prevent division by zero.
    frequencies = np.fft.fftfreq(data.shape[0]) + 1e-6

    filter_b = {
        "highpass": (1 / np.sqrt(1 + (cutoff / frequencies)**(2 * order))).astype(complex),
        "lowpass": (1 / np.sqrt(1 + (frequencies / cutoff)**(2 * order))).astype(complex)
    }[filter_type]

    filtered_data = np.zeros(shape=data.shape)

    for (i, series) in enumerate(data.T):
        dft_series = np.fft.fft(series)
        dft_filtered_series = dft_series * filter_b

        filtered_data[:, i] = np.fft.ifft(dft_filtered_series).real

    return filtered_data

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_difference(
    data: NDArray[Shape["*,*"], Float], **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*,*"], Float]:
    """
    Apply a (discrete) difference high pass filter to each column (time series) of `data`. The
    filtered data will have `order` less timesteps and the original `data`.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The time series data to filter.
    kwargs : dict of {str: any}
        order : int
            The order of the difference filter.

    Returns
    -------
    numpy.ndarray, (n_timestep - order, n_series)
        The filtered time series data.
    """
    order = kwargs["order"]

    filtered_data = np.diff(a=data, n=order, axis=0)

    return filtered_data

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_ideal(
    data: NDArray[Shape["*,*"], Float], **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*,*"], Float]:
    """
    Apply an ideal filter to each column (time series) of `data`.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The time series data to filter.
    kwargs : dict of {str: any}
        cutoff : float
            The cutoff frequency in cycles per sample.
        filter_type : str, ["lowpass", "highpass"], default="highpass"
            The filter type to use.

    Returns
    -------
    numpy.ndarray, (n_timestep, n_series)
        The filtered time series data.
    """
    cutoff = kwargs["cutoff"]
    filter_type = kwargs.get("filter_type", "highpass")

    valid_types = ["lowpass", "highpass"]

    if filter_type not in valid_types:
        ValueError(f"Invalid filter type: {filter_type} - Valid Types: {valid_types}")

    # We add a small number to the frequencies to prevent division by zero.
    frequencies = np.fft.fftfreq(data.shape[0]) + 1e-6

    filter_b = {
        "highpass": np.array([1 if abs(frequency) > cutoff else 0 for frequency in frequencies]),
        "lowpass": np.array([1 if abs(frequency) <= cutoff else 0 for frequency in frequencies])
    }[filter_type]

    filtered_data = np.zeros(shape=data.shape)

    for (i, series) in enumerate(data.T):
        dft_series = np.fft.fft(series)
        dft_filtered_series = dft_series * filter_b

        filtered_data[:, i] = np.fft.ifft(dft_filtered_series).real

    return filtered_data

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_unit_magnitude(
    data: NDArray[Shape["*,*"], Float], **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*,*"], Float]:
    # pylint: disable=unused-argument
    """
    Apply a filter that sets the frequency magnitude of the data to unity. The filtered data will
    only contain phase information.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The time series data to filter.
    kwargs : dict of {str: any}
        will_zero_first : bool, default=True
            Whether to zero the first row of the filtered result. The first element of each filtered
            result will be an anomalous value due to the FFT.

    Returns
    -------
    numpy.ndarray, (n_timestep, n_series)
        The filtered time series data.
    """
    will_zero_first = kwargs.get("will_zero_first", True)

    filtered_data = np.zeros(shape=data.shape)
    data_unit = np.zeros(shape=data.shape, dtype=complex)

    data_dft = np.fft.fft(data, axis=0)
    data_dft_angles = np.arctan(data_dft.imag / data_dft.real)
    data_unit.real = np.cos(data_dft_angles)
    data_unit.imag = np.sin(data_dft_angles)

    filtered_data = np.fft.ifft(data_unit, axis=0).real

    if will_zero_first:
        filtered_data[0, :] = 0

    return filtered_data
