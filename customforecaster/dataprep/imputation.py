import numpy as np
import pandas as pd


def impute_missing_values(time_series, method="linear", order=2):
    """
    Impute missing values in a time series using configurable methods.

    Parameters:
    - time_series (pandas Series): Input time series with missing values.
    - method (str, optional): Imputation method ('linear', 'quadratic', 'cubic', 'nearest', 'mean', or 'pandas').
                              Default is 'linear'.

    Returns:
    - imputed_series (pandas Series): Time series with missing values imputed.
    """

    if method in ["linear", "quadratic", "cubic", "nearest", "mean"]:
        # Interpolation methods
        non_nan_indices = time_series.index[~time_series.isnull()]
        non_nan_values = time_series[~time_series.isnull()]

        if method == "linear":
            interpolation_function = np.interp
        elif method == "quadratic":
            interpolation_function = np.interp
        elif method == "cubic":
            interpolation_function = np.interp
        elif method == "nearest":
            interpolation_function = np.interp
        elif method == "mean":
            mean_value = non_nan_values.mean()
            imputed_series = time_series.fillna(mean_value)

        imputed_values = interpolation_function(
            time_series.index, non_nan_indices, non_nan_values
        )
        imputed_series = pd.Series(imputed_values, index=time_series.index)

    elif method in ["backfill", "bfill", "pad", "ffill"]:
        # Use pandas fillna method for imputation
        imputed_series = time_series.fillna(
            method=method
        )  # Forward fill missing values

    elif method in ["spline", "polynomial"]:
        imputed_series = time_series.interpolate(method=method, order=order)

    else:
        raise ValueError(
            "Invalid imputation method. Choose from 'linear', 'quadratic', 'cubic', 'nearest', 'mean', 'backfill', 'bfill', 'pad', 'ffill', 'spline', 'polynomial'."
        )

    return imputed_series
