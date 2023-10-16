import numpy as np
import scipy.stats as stats


def detect_outlier_sd(ts, sd_multiple=2):
    mean = ts.mean()
    std = ts.std()
    higher_bound = mean + sd_multiple * std
    lower_bound = mean - sd_multiple * std
    outlier_mask = (ts > higher_bound) | (ts < lower_bound)
    return outlier_mask


def detect_outlier_iqr(ts, iqr_multiple=1.5):
    q1, q2, q3 = np.quantile(ts, 0.25), np.quantile(ts, 0.5), np.quantile(ts, 0.75)
    iqr = q3 - q1
    higher_bound = q3 + iqr_multiple * iqr
    lower_bound = q1 - iqr_multiple * iqr
    outlier_mask = (ts > higher_bound) | (ts < lower_bound)
    return outlier_mask
