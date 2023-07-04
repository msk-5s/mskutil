# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains functions for denoising.

Functions in the module must adhere to the following signature:
def function(data: NDArray[Shape["*,*"], Float], **kwargs: Mapping[str, Any]) -> denoiser.Result
"""

from typing import Any, Dict, Callable, Mapping, NamedTuple
from nptyping import Float, NDArray, Shape

import numpy as np
import sklearn.preprocessing

from . import filterer

#===================================================================================================
#===================================================================================================
class Result(NamedTuple):
    """
    A result object returned by a denoising function.

    This class exists to enforce specific return value names while allowing for each denoiser to
    return additional information, if needed.

    Attributes
    ----------
    data : numpy.ndarray of float, (n_timestep, n_series)
        The denoised data.
    other : dict of {str: any}
        Any other extra return values.
    other_to_metadata: callable, default=returns empty dict.
        This function makes a JSON compatible dictionary of specific fields of `other` that need to
        be saved to disk.
    """
    data: NDArray[Shape["*,*"], Float]
    other: Mapping[str, Any]
    other_to_metadata: Callable[[Mapping[str, Any]], Mapping[str, Any]] = lambda other: {}

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_valid_denoisers() -> Dict[
    str, Callable[[NDArray[Shape["*,*"], Float], Mapping[str, Any]], Result]
]:
    """
    Make a dictionary of valid denoisers that are in this module.

    Returns
    -------
    dict of {str: callable}
        A new dictionary of valid denoisers as defined in this module.
    """
    valid_denoisers = {
        "iqr": run_iqr,
        "iqr_svd": run_iqr_svd,
        "none": run_none,
        "rpca": run_rpca,
        "svd": run_svd
    }

    return valid_denoisers

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_none(data: NDArray[Shape["*,*"], Float], **kwargs: Mapping[str, Any]) -> Result:
    # pylint: disable=unused-argument
    """
    Pass the data through with no denoising.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The data to denoise.
    kwargs : dict of {str: any}
        Not used.

    Returns
    -------
    Result
        The result of the denoiser.
    """
    result = Result(data=data, other={}, other_to_metadata=lambda other: {})

    return result

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_iqr(data: NDArray[Shape["*,*"], Float], **kwargs: Mapping[str, Any]) -> Result:
    # pylint: disable=unused-argument,too-many-locals
    """
    An Interquartile Range (IQR) anomaly detector is used to classify timesteps that contain an
    impulse as an anomaly. The value at each anomalous timestep is then replaced with the mean of
    of the non-anomalous values at that timestep across series'.

    If a timestep is anomalous for all series', then the value of the timestep will be set to 0.
    Since time series data is usually made stationary via highpass filtering, this filtering will
    remove the DC component (i.e. the data will be centered at 0).

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The stationary data to denoise.
    kwargs : dict of {str: any}
        filterer_kwargs : dict of {str: any}, default={}
            The kwargs to pass to the filterer.
        run_filter : callable, default=filterer.run_unit_magnitude
            The filterer to high-pass filter the `data`. Each time series needs to be stationary
            (IQR detector performance will degrade with non-stationary data). See `filterer.py` for
            valid filterers. The default filter is the unit magnitude filter, which will only
            utilize the frequency-phase information.

    Returns
    -------
    Result
        The result of the denoiser.
        other : dict of {str: any}
            mask_anomaly : numpy.ndarray of bool, (n_timestep, n_load)
                The mask of anomalous (True) and normal (False) timesteps.
    """
    #***********************************************************************************************
    # Split the data into its low and high frequency components.
    #***********************************************************************************************
    filterer_kwargs = kwargs.get("filterer_kwargs", {})
    run_filter = kwargs.get("run_filter", filterer.run_unit_magnitude)

    temp = run_filter(data=data, **filterer_kwargs)

    # The filtered data may have less timesteps depending on the filtering method.
    data_high = np.zeros(data.shape, dtype=float)
    start = data.shape[0] - temp.shape[0]
    data_high[start:, :] = temp
    del temp

    data_low = data - data_high

    #***********************************************************************************************
    # Determine anomalous values.
    #***********************************************************************************************
    (qrt_1, qrt_3) = np.quantile(a=data_high, q=[0.25, 0.75], axis=0)
    iqr = qrt_3 - qrt_1

    threshold_upper = qrt_3 + (iqr * 1.5)
    threshold_lower = qrt_1 - (iqr * 1.5)

    # The thresholds will be broadcasted such that each meter is compared with its respective
    # threshold.
    mask_anomaly = np.logical_or(data_high > threshold_upper, data_high < threshold_lower)

    #***********************************************************************************************
    # Impute the anomalous values.
    #***********************************************************************************************
    # To prevent division by zero, we set the count of normal values to 1 if there are no normal
    # values. This fine since (0 normal values) / 1 is still 0.
    normal_count = np.sum(~mask_anomaly, axis=1)
    normal_count[normal_count == 0] = 1

    # Anomalous values are initially set to zero. This also accounts for the case where all values
    # at a specific timestep are anomalous.
    data_high[mask_anomaly] = 0

    # Average of normal values across the meters for each timestep.
    impute_values = np.sum(data_high, axis=1) / normal_count
    impute_values = np.repeat(impute_values.reshape(-1, 1), repeats=data_high.shape[1], axis=1)

    # Replace anomalous values with imputed values.
    data_high[mask_anomaly] = impute_values[mask_anomaly]

    # Add back the low-frequency content.
    data_final = data_high + data_low

    result = Result(data=data_final, other={"mask_anomaly": mask_anomaly})

    return result

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_iqr_svd(data: NDArray[Shape["*,*"], Float], **kwargs: Mapping[str, Any]) -> Result:
    """
    Remove impulse noise from the columns of `data` using a non-linear filter and then denoise using
    Singular Value Decomposition (SVD).

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The data to denoise.
    kwargs : dict of {str: any}
        filterer_kwargs : dict of {str: any}
            The kwargs to pass to the filterer.
        run_filter : callable, default=filterer.run_unit_magnitude
            The filterer to high-pass filter the `data`. Each time series needs to be stationary
            (IQR detector performance will degrade with non-stationary data). See `filterer.py` for
            valid filterers. The default filter is the unit magnitude filter, which will only
            utilize the frequency-phase information. However, it is recommended to choose a more
            appropriate filter with appropriate cutoff frequencies.
        value_count : optional of int, default=None
            The number of singular values to use for reconstruction. If `None` is passed, then the
            Singular Value Hard Threshold (SVHT) is used to determine the number of singular values.

    Returns
    -------
    Result
        The result of the denoiser.
        other : dict of {str: any}
            mask_anomaly : numpy.ndarray of bool, (n_timestep, n_load)
                The mask of anomalous (True) and normal (False) timesteps, as determined by the IQR
                anomaly detector.
            values : numpy.ndarray of float, (n_series,)
                The singular values of the denoised data.
            value_count : int
                The number of singular values kept.

    See Also
    --------
    run_iqr, run_svd
    """
    result_iqr = run_iqr(data=data, **kwargs)

    # The data needs to be normalized before applying SVD.
    scaler_obj = sklearn.preprocessing.MinMaxScaler([0, 1]).fit(result_iqr.data)
    data_iqr = scaler_obj.transform(result_iqr.data)

    result_svd = run_svd(data=data_iqr, **kwargs)

    # The normalization is reversed for the final result.
    result_svd = Result(
        data=scaler_obj.inverse_transform(result_svd.data),
        other=result_svd.other,
        other_to_metadata=result_svd.other_to_metadata
    )

    result = Result(
        data=result_svd.data, other={**result_svd.other, **result_iqr.other},
        other_to_metadata=lambda other: {"value_count": other["value_count"]}
    )

    return result

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_rpca(data: NDArray[Shape["*,*"], Float], **kwargs: Mapping[str, Any]) -> Result:
    # pylint: disable=invalid-name,too-many-locals
    """
    Separate the "normal" (low-rank matrix) and "extreme" (sparse matrix) values using Robust
    Principal Component Analysis (RPCA). See "Robust Principal Component Analysis?" by Candes et al.

    This function implements the Principal Component Pursuit (PCP) by Alternating Directions
    algorithm as used in the paper (Algorithm 1 - page 11:28).

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The data to denoise.
    kwargs : dict of {str: any}
        delta : optional of float, default=10e-7
            This coefficient is multiplied by the Frobenius norm of the `data` to give the minimum
            error for convergence. The default value may be too high depending on the `data`. The
            default value is based on the paper.
        lmbda : optional of float, default=1 / np.sqrt(np.max(data.shape))
            The tunable parameter in the rPCA optimization problem. The default value is
            recommended. The default value is based on the paper.
        max_iterations : optional of int, default=1000
            The maximum number of iterations to run PCP for.
        mu : optional of float, default=data.size / (4 * np.linalg.norm(data, ord=1))
            A tunable parameter in the PCP algorithm. The default value is recommended. The default
            value is based on the paper.

    Returns
    -------
    Result
        The result of the denoiser.
        other : dict of {str: any}
            data_sparse : numpy.ndarray of float, (n_timestep, n_series)
                The sparse data maxtrix containing extreme values.
            did_converge : boolean
                Whether PCP converged within the `max_iterations`.
            errors : list of float
                The Frobenius error at each iteration.
            iteration_count : int
                The number of iterations that PCP ran for.
    """
    #***********************************************************************************************
    # Get keyword arguments.
    #***********************************************************************************************
    delta = kwargs.get("delta", 10e-7)
    lmbda = kwargs.get("lmbda", 1 / np.sqrt(np.max(data.shape)))
    max_iterations = kwargs.get("max_iterations", 1000)
    mu = kwargs.get("mu", data.size / (4 * np.linalg.norm(data, ord=1)))
    mu_i = 1.0 / mu

    #***********************************************************************************************
    # Define operators.
    #***********************************************************************************************
    # A function that implements the shrinkage operator.
    def shrink(X, tau):
        return np.sign(X) * np.maximum(np.abs(X) - tau, np.zeros(shape=X.shape))

    # A function that implements the singular value thresholding operator.
    def threshold(X, tau):
        (u, s, vt) = np.linalg.svd(X, full_matrices=False)
        return u @ np.diag(shrink(X=s, tau=tau)) @ vt

    #***********************************************************************************************
    # Perform Principle Component Pursuit by Alternating Directions.
    #***********************************************************************************************
    data_low_rank = np.zeros(shape=data.shape)
    data_sparse = np.zeros(shape=data.shape)
    lagrange = np.zeros(shape=data.shape)

    min_error = delta * np.linalg.norm(data, ord="fro")
    did_converge = False
    errors = []

    for _ in range(max_iterations):
        data_low_rank = threshold(X=data - data_sparse + mu_i * lagrange, tau=mu_i)
        data_sparse = shrink(X=data - data_low_rank + mu_i * lagrange, tau=lmbda * mu_i)
        lagrange = lagrange + mu * (data - data_low_rank - data_sparse)

        error = np.linalg.norm(data - data_low_rank - data_sparse, ord="fro")
        errors.append(error)

        if error <= min_error:
            did_converge = True
            break

    #***********************************************************************************************
    # Return the results.
    #***********************************************************************************************
    result = Result(
        data=data_low_rank,
        other={
            "data_sparse": data_sparse, "did_converge": did_converge, "errors": errors,
            "iteration_count": len(errors)
        },
        other_to_metadata=lambda other: {
            "did_converge": other["did_converge"], "errors": other["errors"],
            "iteration_count": other["iteration_count"]
        }
    )

    return result

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_svd(data: NDArray[Shape["*,*"], Float], **kwargs: Mapping[str, Any]) -> Result:
    """
    Denoise the data using Singular Value Decomposition (SVD).

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The data to denoise.
    kwargs : dict of {str: any}
        value_count : optional of int, default=None
            The number of singular values to use for reconstruction. If `None` is passed, then the
            Singular Value Hard Threshold (SVHT) is used to determine the number of singular values.

    Returns
    -------
    Result
        The result of the denoiser.
        other : dict of {str: any}
            values : numpy.ndarray of float, (n_series,)
                The singular values of the denoised data.
            value_count : int
                The number of singular values kept.
    """
    value_count = kwargs.get("value_count", None)

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

    data_d = u[:, :value_count] @ np.diag(s[:value_count]) @ vt[:value_count, :]

    result = Result(
        data=data_d, other={"values": s, "value_count": value_count},
        other_to_metadata=lambda other: {"value_count": other["value_count"]}
    )

    return result
