# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains functions for dimensionality reduction.

Functions in the module must adhere to the following signature:
def function(
    data: NDArray[Shape["*,*"], Float], random_state: int, **kwargs: Mapping[str, Any]
) -> reducer.Result
"""

from typing import Any, Dict, Callable, Mapping, NamedTuple
from nptyping import Float, NDArray, Shape

import sklearn.decomposition
import sklearn.manifold

#===================================================================================================
#===================================================================================================
class Result(NamedTuple):
    """
    A result object returned by a reducing function.

    This class exists to enforce specific return value names while allowing for each reducer to
    return additional information, if needed.

    Attributes
    ----------
    data : numpy.ndarray of float, (n_sample, n_component)
        The reduced data.
    other : dict of {str: any}
        Any other extra return values.
    other_to_metadata: callable, default=returns empty dict.
        This function makes a JSON compatible dictionary of specific fields of `other` that need to
        be saved to disk.
    """
    data: NDArray[Shape["*,*"], Float]
    other: Mapping[str, Any] = {}
    other_to_metadata: Callable[[Mapping[str, Any]], Mapping[str, Any]] = lambda other: {}

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_valid_reducers() -> Dict[
    str, Callable[[NDArray[Shape["*,*"], Float], int, Mapping[str, Any]], Result]
]:
    """
    Make a dictionary of valid reducers that are in this module.

    The number of valid reducers returned might not include all of the reducer functions defined
    in this module. This function is primarily to aid in running a suite of simulation cases.

    Returns
    -------
    dict of {str: callable}
        A new dictionary of valid reducers as defined in this module.
    """
    valid_reducers = {
        "none": run_none,
        "nmf": run_nmf,
        "pca": run_pca,
        "le": run_le,
        "tsne": run_tsne
    }

    return valid_reducers

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_none(
    data: NDArray[Shape["*,*"], Float], random_state: int, **kwargs: Mapping[str, Any]
) -> Result:
    # pylint: disable=unused-argument
    """
    Pass the data through with no dimensionality reduction.

    Parameters
    ----------
    data : numpy.ndarray, (n_series, n_timestep)
        The data to reduce.
    random_state : int
        The state to use for rng.
    kwargs : dict of {str: any}
        Not used.

    Returns
    -------
    Result
        The result of the reducer.
    """
    result = Result(data=data)

    return result

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_nmf(
    data: NDArray[Shape["*,*"], Float], random_state: int, **kwargs: Mapping[str, Any]
) -> Result:
    """
    Use Non-Negative Matrix Factorization (NMF) to reduce the dimensionality of the `data`.

    Parameters
    ----------
    data : numpy.ndarray, (n_series, n_timestep)
        The data to reduce.
    random_state : int
        The state to use for rng.
    kwargs : dict of {str: any}
        See scikit-learn's documentation for `NMF`.

    Returns
    -------
    Result
        The result of the reducer.
    """
    reducer = sklearn.decomposition.NMF(random_state=random_state, **kwargs).fit(data)

    result = Result(
        data=reducer.transform(data),
        other={"n_iter_": reducer.n_iter_, "reconstruction_err_": reducer.reconstruction_err_},
        other_to_metadata=lambda other: other
    )

    return result

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_pca(
    data: NDArray[Shape["*,*"], Float], random_state: int, **kwargs: Mapping[str, Any]
) -> Result:
    """
    Use Principal Component Analysis (PCA) to reduce the dimensionality of the `data`.

    Parameters
    ----------
    data : numpy.ndarray, (n_series, n_timestep)
        The data to reduce.
    random_state : int
        The state to use for rng.
    kwargs : dict of {str: any}
        See scikit-learn's documentation for `PCA`.

    Returns
    -------
    Result
        The result of the reducer.
    """
    reducer = sklearn.decomposition.PCA(random_state=random_state, **kwargs).fit(data)

    result = Result(data=reducer.transform(data))

    return result

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_le(
    data: NDArray[Shape["*,*"], Float], random_state: int, **kwargs: Mapping[str, Any]
) -> Result:
    """
    Use laplacian eigenmaps (spectral embedding) to reduce the dimensionality of the data.

    Parameters
    ----------
    data : numpy.ndarray, (n_series, n_timestep)
        The data to reduce.
    random_state : int
        The state to use for rng.
    kwargs : dict of {str: any}
        See scikit-learn's documentation for `SpectralEmbedding`.

    Returns
    -------
    Result
        The result of the reducer.
    """
    reducer = sklearn.manifold.SpectralEmbedding(random_state=random_state, **kwargs).fit(data)

    result = Result(data=reducer.embedding_)

    return result

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_tsne(
    data: NDArray[Shape["*,*"], Float], random_state: int, **kwargs: Mapping[str, Any]
) -> Result:
    """
    Use T-distributed Stochastic Neighbor Embedding (T-SNE) to reduce the dimensionality of the
    data.

    Parameters
    ----------
    data : numpy.ndarray, (n_series, n_timestep)
        The data to reduce.
    random_state : int
        The state to use for rng.
    kwargs : dict of {str: any}
        See scikit-learn's documentation for `TSNE`.

    Returns
    -------
    Result
        The result of the reducer.
    """
    reducer = sklearn.manifold.TSNE(random_state=random_state, **kwargs).fit(data)

    result = Result(data=reducer.embedding_)

    return result
