# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains functions for clustering.

Functions in the module must adhere to the following signature:
def function(
    data: NDArray[Shape["*,*"], Float], random_state: int, **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*"], Int]
"""

from typing import Any, Callable, Dict, Mapping
from nptyping import Float, Int, NDArray, Shape

import numpy as np
import sklearn.cluster
import sklearn.metrics
import sklearn.mixture
import sklearn.neighbors
import sklearn_extra.cluster

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_valid_clusterers() -> Dict[
    str, Callable[[NDArray[Shape["*,*"], Float], int], NDArray[Shape["*,*"], Int]]
]:
    """
    Make a dictionary of valid clusterers that are in this module.

    Returns
    -------
    dict of {str: callable}
        A new dictionary of valid clusterers as defined in this module.
    """
    valid_clusterers = {
        "dbscan": run_dbscan,
        "gmm": run_gmm,
        "hierarchical": run_hierarchical,
        "kmeans": run_kmeans,
        "kmedoids": run_kmedoids,
    }

    return valid_clusterers

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_dbscan(
    data: NDArray[Shape["*,*"], Float], random_state: int, **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*"], Int]:
    # pylint: disable=unused-argument
    """
    Cluster data using a Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Parameters
    ----------
    data : numpy.ndarray of float, (n_sample, n_dim)
        The data.
    random_state : int
        The random state to use.
    kwargs : dict of {str: any}
        will_merge_noise : bool, default=False
            Whether to merge the noise into its nearest cluster. A sample labeled as noise
            will be reclassified to the cluster it is closest to using a KNN classifier.
        See the scikit-learn's documentation for `DBSCAN`.

    Returns
    -------
    numpy.ndarray of int, (n_sample,)
        The predicted clusters.
    """
    will_merge_noise = kwargs.get("will_merge_noise", False)
    kwargs_local = kwargs.copy()
    kwargs_local.pop("will_merge_noise")

    clusters = sklearn.cluster.DBSCAN(**kwargs_local).fit_predict(data)

    noise_indices = np.where(clusters == -1)[0]

    if will_merge_noise and len(noise_indices) > 0:
        normal_indices = np.where(clusters != -1)[0]

        noise_data = data[noise_indices, :]
        normal_data = data[normal_indices, :]

        normal_clusters = clusters[normal_indices]
        model = sklearn.neighbors.KNeighborsClassifier().fit(X=normal_data, y=normal_clusters)

        clusters[noise_indices] = model.predict(noise_data)

    return clusters

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_gmm(
    data: NDArray[Shape["*,*"], Float], random_state: int, **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*"], Int]:
    """
    Cluster the data using a Gaussian Mixture Model (GMM).

    Parameters
    ----------
    data : numpy.ndarray of float, (n_sample, n_dim)
        The data.
    random_state : int
        The random state to use.
    kwargs : dict of {str: any}
        See the scikit-learn's documentation for `GaussianMixtureModel`.

    Returns
    -------
    numpy.ndarray of int, (n_sample,)
        The predicted clusters.
    """
    clusters = sklearn.mixture.GaussianMixture(
        random_state=random_state, **kwargs
    ).fit_predict(data)

    return clusters

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_hierarchical(
    data: NDArray[Shape["*,*"], Float], random_state: int, **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*"], Int]:
    # pylint: disable=unused-argument
    """
    Cluster the data using hierarchical clustering.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_sample, n_dim)
        The data.
    random_state : int
        The random state to use.
    kwargs : dict of {str: any}
        See the scikit-learn's documentation for `AgglomerativeClustering`.

    Returns
    -------
    numpy.ndarray of int, (n_sample,)
        The predicted clusters.
    """
    clusters = sklearn.cluster.AgglomerativeClustering(**kwargs).fit_predict(data)

    return clusters

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_kmeans(
    data: NDArray[Shape["*,*"], Float], random_state: int, **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*"], Int]:
    """
    Cluster the data using k-means clustering.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_sample, n_dim)
        The data.
    random_state : int
        The random state to use.
    kwargs : dict of {str: any}
        See the scikit-learn's documentation for `KMeans`.

    Returns
    -------
    numpy.ndarray of int, (n_sample,)
        The predicted clusters.
    """
    clusters = sklearn.cluster.KMeans(random_state=random_state, **kwargs).fit_predict(data)

    return clusters

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_kmedoids(
    data: NDArray[Shape["*,*"], Float], random_state: int, **kwargs: Mapping[str, Any]
) -> NDArray[Shape["*"], Int]:
    """
    Cluster the data using k-medoids clustering.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_sample, n_dim)
        The data.
    random_state : int
        The random state to use.
    kwargs : dict of {str: any}
        n_init : int
            The number of times to run the k-medoids clustering. The clusters with the highest
            Calinski-Harabasz score will be used.
        See the scikit-learn-extra's documentation for `KMedoids`.

    Returns
    -------
    numpy.ndarray of int, (n_sample,)
        The predicted clusters.
    """
    n_init = kwargs["n_init"]
    kwargs_local = kwargs.copy()
    kwargs_local.pop("n_init")

    rng = np.random.default_rng(seed=random_state)

    best_clusters = None
    best_score = -np.inf

    for _ in range(n_init):
        random_state_cluster = rng.integers(np.iinfo(np.int32).max)

        clusters = sklearn_extra.cluster.KMedoids(
            random_state=random_state_cluster, **kwargs_local
        ).fit_predict(data)

        score = sklearn.metrics.calinski_harabasz_score(X=data, labels=clusters)

        if score > best_score:
            best_score = score
            best_clusters = clusters

    return best_clusters
