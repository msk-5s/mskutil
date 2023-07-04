# SPDX-License-Identifier: BSD-3-Clause

"""
The module contains functions for creating different types of matrices.
"""

import math

from typing import Any, Callable, Mapping
from nptyping import Complex128, Float, Int, NDArray, Shape

from rich.progress import track

import numpy as np

from . import denoiser

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_average_spectrogram_matrix(
    data_n: NDArray[Shape["*,*"], Float], denoiser_kwargs: Mapping[str, Any],
    run_denoise: Callable[[NDArray[Shape["*,*"], Float], Mapping[str, Any]], denoiser.Result],
    stride: int, width: int
) -> NDArray[Shape["*,*"], Complex128]:
    # pylint: disable=too-many-arguments
    """
    Make an average spectrogram over the entire year of denoised data.

    This function is only used as a convenience function for plotting spectrograms.

    Parameters
    ----------
    data_n : numpy.ndarray of float, (n_timestep, n_meter or n_meter_channel)
        The noisy voltage magnitude data.
    denoiser_kwargs : dict of {str: any}
        The kwargs to pass to the denoiser.
    run_denoise : callable of (data: NDArray[(Any, Any), float],
                  **kwargs: Mapping[str, Any]) -> denoiser.Result
        The denoising function to use. See `denoiser.py` for valid denoisers.
    stride : int
        The stride of the windows in timesteps.
    width : int
        The width of the window in timesteps.

    Returns
    -------
    numpy.ndarray of numpy.complex128, (width, n_window)
        The time series as a matrix of stride-lagged windows.
    """
    window_count = math.ceil((data_n.shape[0] - width) / stride)
    start_indices = np.array([stride * i for i in range(window_count)])

    spectrogram = np.zeros(shape=(width, window_count), dtype=np.complex128)

    for (i, start) in track(
        enumerate(start_indices), "Making spectrogram...", total=len(start_indices)
    ):
        window_n = data_n[start:(start + width), :]
        window_nd = run_denoise(data=window_n, **denoiser_kwargs).data
        window_dft = np.fft.fft(window_nd, axis=0)

        average_dft = np.mean(window_dft, axis=1)

        spectrogram[:, i] = average_dft

    return spectrogram

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_ideal_phase_correlation_matrix(
    labels: NDArray[Shape["*"], Int]
) -> NDArray[Shape["*,*"], Float]:
    """
    Make the ideal correlation matrix for load phases. In the ideal case, loads connected to the
    same phase will be perfectly correlated (coefficient of 1).

    Parameters
    ----------
    labels : numpy.ndarray of int, (n_load,)
        The phase label of each load.

    Returns
    -------
    numpy.ndarray of float, (n_load, n_load)
        The ideal phase correlation matrix.
    """
    sort_indices = np.argsort(labels)
    labels = labels[sort_indices]
    phase_indices = [np.where(labels == i)[0] for i in np.unique(labels)]

    n_load = len(labels)
    result = np.zeros(shape=(n_load, n_load))

    for indices in phase_indices:
        for i in indices:
            for j in indices:
                result[i, j] = 1

    return result

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_phase_sorted_correlation_matrix(
    data: NDArray[Shape["*,*"], Float], labels: NDArray[Shape["*"], Int]
) -> NDArray[Shape["*,*"], Float]:
    """
    Make a correlation matrix that is sorted by load phase.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_load)
        The data to calculate the correlation matrix from.
    labels : numpy.ndarray of int, (n_load,)
        The phase label of each load.

    Returns
    -------
    numpy.ndarray of float, (n_load, n_load)
        The phase sorted correlation matrix.
    """
    sort_indices = np.argsort(labels)

    result = np.corrcoef(data[:, sort_indices], rowvar=False)

    return result

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_series_spectrogram(
    data_n: NDArray[Shape["*,*"], Float], denoiser_kwargs: Mapping[str, Any], index: int,
    run_denoise: Callable[[NDArray[Shape["*,*"], Float], Mapping[str, Any]], denoiser.Result],
    stride: int, width: int
) -> NDArray[Shape["*,*"], Complex128]:
    # pylint: disable=too-many-arguments
    """
    Make the spectrogram for a single load over the entire year of data.

    This function is only used as a convenience function for plotting spectrograms.

    Parameters
    ----------
    data_n : numpy.ndarray of float, (n_timestep, n_load)
        The noisy load voltage magnitude data.
    denoiser_kwargs : dict of {str: any}
        The kwargs to pass to the denoiser.
    index : int
        The index of the load's time series to use.
    run_denoise : callable of (data: NDArray[(Any, Any), float],
                  **kwargs: Mapping[str, Any]) -> denoiser.Result
        The denoising function to use. See `denoiser.py` for valid denoisers.
    stride : int
        The stride of the windows in timesteps.
    width : int
        The width of the window in timesteps.

    Returns
    -------
    numpy.ndarray of numpy.complex128, (width, n_window)
        The spectrogram of the load.
    """
    window_count = math.ceil((data_n.shape[0] - width) / stride)
    start_indices = np.array([stride * i for i in range(window_count)])

    spectrogram = np.zeros(shape=(width, window_count), dtype=np.complex128)

    for (i, start) in track(
        enumerate(start_indices), "Making spectrogram...", total=len(start_indices)
    ):
        window_n = data_n[start:(start + width), :]
        window_nd = run_denoise(data=window_n, **denoiser_kwargs).data
        series_dft = np.fft.fft(window_nd[:, index])

        spectrogram[:, i] = series_dft

    return spectrogram
