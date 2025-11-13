"""
Project: BlueMath_tk
Sub-Module: deeplearning.metrics
Author: GeoOcean Research Group, Universidad de Cantabria
Repository: https://github.com/GeoOcean/BlueMath_tk.git
Status: Under development (Working)

Evaluation metrics for deep learning models.
"""

import numpy as np


def mse(A: np.ndarray, B: np.ndarray) -> float:
    """
    Calculate Mean Squared Error (MSE).

    Parameters
    ----------
    A : np.ndarray
        True values.
    B : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Mean squared error.
    """
    return float(np.mean((A - B) ** 2))


def mae(A: np.ndarray, B: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    Parameters
    ----------
    A : np.ndarray
        True values.
    B : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Mean absolute error.
    """
    return float(np.mean(np.abs(A - B)))


def rmse(A: np.ndarray, B: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).

    Parameters
    ----------
    A : np.ndarray
        True values.
    B : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Root mean squared error.
    """
    return float(np.sqrt(mse(A, B)))


def r2_overall(X: np.ndarray, Xhat: np.ndarray) -> float:
    """
    Calculate overall R² score.

    Parameters
    ----------
    X : np.ndarray
        True values.
    Xhat : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Overall R² score.
    """
    SST = np.sum((X - X.mean(axis=0, keepdims=True)) ** 2)
    SSE = np.sum((X - Xhat) ** 2)
    return float(1.0 - SSE / (SST + 1e-12))


def r2_per_feature(X: np.ndarray, Xhat: np.ndarray) -> np.ndarray:
    """
    Calculate R² score per feature.

    Parameters
    ----------
    X : np.ndarray
        True values.
    Xhat : np.ndarray
        Predicted values.

    Returns
    -------
    np.ndarray
        Array of R² scores per feature, shape (n_features,).
    """
    num = np.sum((X - Xhat) ** 2, axis=0)
    den = np.sum((X - X.mean(axis=0, keepdims=True)) ** 2, axis=0) + 1e-12
    return 1.0 - num / den


def r2_map_over_time(
    X_true: np.ndarray, X_pred: np.ndarray, axis_time: int = 0
) -> np.ndarray:
    """
    Calculate R² map over time for spatial data.

    Parameters
    ----------
    X_true : np.ndarray
        True values, shape (T, H, W, C) or similar.
    X_pred : np.ndarray
        Predicted values, same shape as X_true.
    axis_time : int, optional
        Time axis dimension, by default 0.

    Returns
    -------
    np.ndarray
        R² map over time, shape (H, W, C) or similar.
    """
    Xt = X_true - np.nanmean(X_true, axis=axis_time, keepdims=True)
    num = np.nanmean((X_true - X_pred) ** 2, axis=axis_time)
    den = np.nanmean(Xt**2, axis=axis_time) + 1e-12
    return 1.0 - num / den


def r2(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """
    Calculate R² score per feature (alternative implementation).

    Parameters
    ----------
    y : np.ndarray
        True values.
    yhat : np.ndarray
        Predicted values.

    Returns
    -------
    np.ndarray
        Array of R² scores per feature.
    """
    ss_res = np.sum((y - yhat) ** 2, axis=0)
    ss_tot = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def compute_all_metrics(
    X_true: np.ndarray, X_pred: np.ndarray, spatial: bool = False
) -> dict:
    """
    Compute all available metrics for model evaluation.

    Parameters
    ----------
    X_true : np.ndarray
        True values.
    X_pred : np.ndarray
        Predicted values.
    spatial : bool, optional
        Whether to compute spatial metrics, by default False.

    Returns
    -------
    dict
        Dictionary containing all computed metrics.
    """
    metrics = {
        "mse": mse(X_true, X_pred),
        "mae": mae(X_true, X_pred),
        "rmse": rmse(X_true, X_pred),
        "r2_overall": r2_overall(X_true, X_pred),
        "r2_per_feature": r2_per_feature(X_true, X_pred),
    }

    if spatial and X_true.ndim >= 3:
        metrics["r2_map_over_time"] = r2_map_over_time(X_true, X_pred)

    return metrics
