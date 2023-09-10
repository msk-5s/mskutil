# SPDX-License-Identifier: BSD-3-Clause

"""
This modules contains functions for making different types of data window generators.
"""

from typing import Generator, NamedTuple, Optional
from nptyping import Float, NDArray, Shape

import numpy as np

#===================================================================================================
#===================================================================================================
class Result(NamedTuple):
    """
    A result object returned by a make generator function.

    This class exists to enforce specific return value names for each factory function.

    Attributes
    ----------
    generator : generator of ((numpy.ndarray of float, (width, n_load)), None, None)
        A generator that generates a random data window.
    n_window : int
        The number of unique windows that the `generator` can produce. If a generator has no
        replacement, then the generator can only generate `n_window` data windows before running
        out.
    """
    generator: Generator[NDArray[Shape["*,*"], Float], None, None]
    n_window: Optional[int] = None

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_bootstrap_replacement_generator(
    data: NDArray[Shape["*,*"], Float], stride: int, width: int, random_state: int
) -> Result:
    """
    Make a generator function that will return windows of data, selected uniformly at random, of
    width `width`. The data window will be placed back in the pool of possible resamples.

    The `data` will be split into windows of width `width` with a stride of `stride` inbetween them.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_load)
        The data to use.
    stride : int
        The stride of each window in timesteps.
    width : int
        The width of each window in timesteps.
    random_state : int
        The random state to use for pseudorandomness repeatability.

    Returns
    -------
    Result
        The result of the factory function.
    """
    rng = np.random.default_rng(seed=random_state)

    start_indices = list(np.arange(
        start=0, stop=data.shape[0] - width, step=stride
    ))

    def generator():
        while True:
            index = rng.integers(low=0, high=len(start_indices))
            start = start_indices[index]

            window = data[start:(start + width), :]

            yield window

    return Result(generator=generator(), n_window=len(start_indices))

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_bootstrap_no_replacement_generator(
    data: NDArray[Shape["*,*"], Float], stride: int, width: int, random_state: int
) -> Result:
    """
    Make a generator function that will return windows of data, selected uniformly at random, of
    width `width`. The data windows will not be placed back into the pool of windows.

    The `data` will be split into windows of width `width` with a stride of `stride` inbetween them.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_load)
        The data to use.
    width : int
        The width of each window in timesteps.
    random_state : int
        The random state to use for pseudorandomness repeatability.
    stride : int
        The stride of each window in timesteps.
    with_replacement : bool, default=True
        Whether the sampled window will remain selectable.

    Returns
    -------
    Result
        The result of the factory function.
    """
    rng = np.random.default_rng(seed=random_state)

    start_indices = list(np.arange(
        start=0, stop=data.shape[0] - width, step=stride
    ))

    def generator():
        while True:
            index = rng.integers(low=0, high=len(start_indices))
            start = start_indices.pop(index)

            window = data[start:(start + width), :]

            if start_indices:
                yield window
            else:
                return window

    return Result(generator=generator(), n_window=len(start_indices))
