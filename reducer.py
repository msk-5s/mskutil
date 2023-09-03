# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains functions for dimensionality reduction.

Functions in the module must adhere to the following signature:
def function(
    data: NDArray[Shape["*,*"], Float], random_state: Optional[int], **kwargs: Mapping[str, Any]
) -> reducer.Result
"""

from typing import Any, Dict, Callable, Mapping, NamedTuple, Optional
from nptyping import Float, NDArray, Shape

import numpy as np
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
    str, Callable[[NDArray[Shape["*,*"], Float], Optional[int], Mapping[str, Any]], Result]
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
        "le": run_le,
        "nmf": run_nmf,
        "none": run_none,
        "pca": run_pca,
        "svd": run_svd,
        "tsne": run_tsne
    }

    return valid_reducers

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_none(
    data: NDArray[Shape["*,*"], Float], random_state: Optional[int] = None,
    **kwargs: Mapping[str, Any]
) -> Result:
    # pylint: disable=unused-argument
    """
    Pass the data through with no dimensionality reduction.

    Parameters
    ----------
    data : numpy.ndarray, (n_series, n_timestep)
        The data to reduce.
    random_state : optional of int, default=None
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
def run_le(
    data: NDArray[Shape["*,*"], Float], random_state: Optional[int] = None,
    **kwargs: Mapping[str, Any]
) -> Result:
    """
    Use laplacian eigenmaps (spectral embedding) to reduce the dimensionality of the data.

    Parameters
    ----------
    data : numpy.ndarray, (n_series, n_timestep)
        The data to reduce.
    random_state : optional of int, default=None
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
def run_nmf(
    data: NDArray[Shape["*,*"], Float], random_state: Optional[int] = None,
    **kwargs: Mapping[str, Any]
) -> Result:
    """
    Use Non-Negative Matrix Factorization (NMF) to reduce the dimensionality of the `data`.

    Parameters
    ----------
    data : numpy.ndarray, (n_series, n_timestep)
        The data to reduce.
    random_state : optional of int, default=None
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
    data: NDArray[Shape["*,*"], Float], random_state: Optional[int] = None,
    **kwargs: Mapping[str, Any]
) -> Result:
    """
    Use Principal Component Analysis (PCA) to reduce the dimensionality of the `data`.

    Parameters
    ----------
    data : numpy.ndarray, (n_series, n_timestep)
        The data to reduce.
    random_state : optional of int, default=None
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
def run_svd(
    data: NDArray[Shape["*,*"], Float], random_state: Optional[int] = None,
    **kwargs: Mapping[str, Any]
) -> Result:
    # pylint: disable=unused-argument
    """
    Use Singular Value Decomposition (SVD) to reduce the dimensionality of the data.

    Parameters
    ----------
    data : numpy.ndarray, (n_series, n_timestep)
        The data to reduce.
    random_state : optional of int, default=None
        The state to use for rng.
    kwargs : dict of {str: any}
        n_components : optional of int, default=None
            The number of dimensions in the output. If `None` is passed, then the Singular Value
            Hard Threshold (SVHT) is used to determine the number of singular values, and thus the
            number of dimensions.

    Returns
    -------
    Result
        The result of the reducer.
        other : dict of {str: any}
            axis : numpy.ndarray of float, (n_components, n_series)
                The right singular vectors.
            values : numpy.ndarray of float, (n_series,)
                The singular values of the data.
            value_count : int
                The number of singular values kept.
    """
    value_count = kwargs.get("n_components", None)

    # pylint: disable=invalid-name
    (u, s, vt) = np.linalg.svd(data, full_matrices=False)

    if value_count is None:
        # Calculate the Singular Value Hard Threshold (SVHT) as presented in "The Optimal Hard
        # Threshold for Singular Values is 4 / sqrt(3)" by Gavish, et al.
        beta = (data.shape[0] / data.shape[1]) if (data.shape[0] < data.shape[1]) else \
               (data.shape[1] / data.shape[0])

        omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
        threshold = omega * np.median(s)

        value_count = len(s[s >= threshold])
        value_count = value_count if value_count > 0 else 1

    data_dr = u[:, :value_count]

    result = Result(
        data=data_dr, other={
            "axis": vt[:value_count, :].T, "values": s[:value_count], "value_count": value_count
        },
        other_to_metadata=lambda other: {"values": other["values"],
        "value_count": other["value_count"]}
    )

    return result

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_tsne(
    data: NDArray[Shape["*,*"], Float], random_state: Optional[int] = None,
    **kwargs: Mapping[str, Any]
) -> Result:
    """
    Use T-distributed Stochastic Neighbor Embedding (T-SNE) to reduce the dimensionality of the
    data.

    Parameters
    ----------
    data : numpy.ndarray, (n_series, n_timestep)
        The data to reduce.
    random_state : optional of int, default=None
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
