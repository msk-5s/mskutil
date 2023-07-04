# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains functions for label modification.

Functions in the module must adhere to the following signature:
def function(
    data: NDArray[Shape["*"], float], labels: NDArray[Shape["*"], int]
) -> NDArray[(Any,), int]
"""

from nptyping import Int, Shape, NDArray

import numpy as np

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_majority_vote_correction(
    clusters: NDArray[Shape["*"], Int], labels: NDArray[Shape["*"], Int]
)-> NDArray[Shape["*"], Int]:
    """
    Do label correction using predicted clusters and labels with a majority vote rule.

    Parameters
    ----------
    clusters : numpy.ndarray of int, (n_label,)
        The predicted clusters.
    labels : numpy.ndarray of int, (n_label,)
        The labels to use in the majority vote approach.

    Returns
    -------
    numpy.ndarray of int, (n_label,)
        The label predictions via majority vote.
    """
    unique_clusters = np.unique(clusters)

    indices_list = [np.where(clusters == i)[0] for i in unique_clusters]

    predictions = np.zeros(shape=len(clusters), dtype=int)

    for indices in indices_list:
        observed_labels = labels[indices]
        predicted_label = np.bincount(observed_labels).argmax()

        predictions[indices] = predicted_label

    return predictions

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_error_change(
    labels: NDArray[Shape["*"], Int], percent: float, random_state: int
)-> NDArray[Shape["*"], Int]:
    """
    Make the `labels` erroneous by changing a random number of labels (a percent of the total
    labels).

    Parameters
    ----------
    labels : numpy.ndarray of int, (n_label,)
        The labels to change.
    percent : float
        The percentage of phase labels to make incorrect.
    random_state: int
        The random state to use for rng.

    Returns
    -------
    numpy.ndarray of int, (n_load,)
        The erroneous labels.
    """
    rng = np.random.default_rng(random_state)

    label_count = len(labels)
    error_count = int(label_count * percent)
    indices = rng.permutation(label_count)[:error_count]

    unique_count = len(np.unique(labels))
    result = labels.copy()

    # Increment the original label by 1 and wrap around when appropriate.
    result[indices] = (result[indices] + 1) % unique_count

    return result
