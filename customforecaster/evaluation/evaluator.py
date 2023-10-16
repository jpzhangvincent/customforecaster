from typing import Callable, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, mase, mse, smape
from darts.metrics.metrics import _get_values_or_raise


def _remove_nan_union(
    array_a: np.ndarray, array_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the two inputs arrays where all elements are deleted that have an index that corresponds to
    a NaN value in either of the two input arrays.
    """

    isnan_mask = np.logical_or(np.isnan(array_a), np.isnan(array_b))
    return np.delete(array_a, isnan_mask), np.delete(array_b, isnan_mask)


def forecast_bias(
    actual_series: Union[TimeSeries, Sequence[TimeSeries], np.ndarray],
    pred_series: Union[TimeSeries, Sequence[TimeSeries], np.ndarray],
    intersect: bool = True,
) -> Union[float, np.ndarray]:
    """Forecast Bias (FB).

    Given a time series of actual values :math:`y_t` and a time series of predicted values :math:`\\hat{y}_t`
    both of length :math:`T`, it is a percentage value computed as

    .. math:: 100 \\cdot \\frac{\\sum_{t=1}^{T}{y_t}
              - \\sum_{t=1}^{T}{\\hat{y}_t}}{\\sum_{t=1}^{T}{y_t}}.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

    Parameters
    ----------
    actual_series
        The `TimeSeries` or `Sequence[TimeSeries]` of actual values.
    pred_series
        The `TimeSeries` or `Sequence[TimeSeries]` of predicted values.
    intersect
        For time series that are overlapping in time without having the same time index, setting `intersect=True`
        will consider the values only over their common time interval (intersection in time).

    Raises
    ------
    ValueError
        If :math:`\\sum_{t=1}^{T}{y_t} = 0`.

    Returns
    -------
    float
        The Forecast Bias (OPE)
    """
    assert type(actual_series) is type(
        pred_series
    ), "actual_series and pred_series should be of same type."
    if isinstance(actual_series, np.ndarray):
        y_true, y_pred = actual_series, pred_series
    else:
        y_true, y_pred = _get_values_or_raise(actual_series, pred_series, intersect)
    y_true, y_pred = _remove_nan_union(y_true, y_pred)
    y_true_sum, y_pred_sum = np.sum(y_true), np.sum(y_pred)
    return ((y_true_sum - y_pred_sum) / y_true_sum) * 100.0


def cast_to_series(df):
    is_pd_dataframe = isinstance(df, pd.DataFrame)
    if is_pd_dataframe:
        if df.shape[1] == 1:
            df = df.squeeze()
        else:
            raise ValueError(
                "Dataframes with more than one columns cannot be converted to pd.Series"
            )
    return df


def calculate_metrics(
    y: Union[pd.Series, TimeSeries],
    y_pred: Union[pd.Series, TimeSeries],
    name: str,
    y_train: Union[pd.Series, TimeSeries] = None,
):
    """Method to calculate the metrics given the actual and predicted series

    Args:
        y (pd.Series or TimeSeries): Actual target with datetime index
        y_pred (pd.Series or TimeSeries): Predictions with datetime index
        name (str): Name or identification for the model
        y_train (pd.Series, optional): Actual train target to calculate MASE with datetime index. Defaults to None.

    Returns:
        Dict: Dictionary with MAE, MSE, MASE, sMAPE and Forecast Bias
    """
    y, y_pred = cast_to_series(y), cast_to_series(y_pred)
    if y_train is not None:
        y_train = cast_to_series(y_train)

    is_nd_array = isinstance(y, np.ndarray)
    is_pd_series = isinstance(y, pd.Series)

    if is_nd_array:
        y, y_pred = TimeSeries.from_values(y), TimeSeries.from_values(y_pred)
        if y_train is not None:
            y_train = TimeSeries.from_values(y_train)

    elif is_pd_series:
        y, y_pred = TimeSeries.from_series(y), TimeSeries.from_series(y_pred)
        if y_train is not None:
            y_train = TimeSeries.from_series(y_train)

    return {
        "Algorithm": name,
        "MAE": mae(actual_series=y, pred_series=y_pred),
        "MSE": mse(actual_series=y, pred_series=y_pred),
        "MASE": mase(actual_series=y, pred_series=y_pred, insample=y_train)
        if y_train is not None
        else None,
        "sMAPE": smape(actual_series=y, pred_series=y_pred),
        "Forecast Bias": forecast_bias(actual_series=y, pred_series=y_pred),
    }
