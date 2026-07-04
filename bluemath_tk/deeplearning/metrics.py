"""Metrics and losses for deeplearning models.

This module provides reconstruction metrics that can be used both for
post-training evaluation and, when using PyTorch tensors, as differentiable
training losses.
"""

from __future__ import annotations

from typing import Dict, Literal, Tuple, Union

import numpy as np
import torch
from torch import nn


ArrayLike = Union[np.ndarray, torch.Tensor]
MetricName = Literal["mse", "mae", "rmse"]
ReductionName = Literal["none", "sample", "mean", "sum"]


__all__ = [
    "reconstruction_error",
    "evaluate_reconstruction",
    "ReconstructionLoss",
]


def _normalise_options(metric: str, reduction: str) -> Tuple[str, str]:
    """Validate and normalise metric/reduction names."""
    metric = metric.lower()
    reduction = reduction.lower()

    valid_metrics = {"mse", "mae", "rmse"}
    valid_reductions = {"none", "sample", "mean", "sum"}

    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics}, got {metric!r}")
    if reduction not in valid_reductions:
        raise ValueError(
            f"reduction must be one of {valid_reductions}, got {reduction!r}"
        )

    return metric, reduction


def _uses_torch(y_true: ArrayLike, y_pred: ArrayLike) -> bool:
    """Return True when at least one input is a PyTorch tensor."""
    return torch.is_tensor(y_true) or torch.is_tensor(y_pred)


def _to_matching_tensor(
    array: ArrayLike,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert an array to a tensor matching a reference tensor if provided."""
    if torch.is_tensor(array):
        return array

    if reference is not None:
        return torch.as_tensor(array, dtype=reference.dtype, device=reference.device)

    return torch.as_tensor(array)


def _to_numpy(array: ArrayLike) -> np.ndarray:
    """Convert NumPy arrays or tensors to NumPy arrays for summary statistics."""
    if torch.is_tensor(array):
        return array.detach().cpu().numpy()

    return np.asarray(array)


def _check_same_shape(y_true: ArrayLike, y_pred: ArrayLike) -> None:
    """Raise a ValueError when target and prediction shapes differ."""
    if tuple(y_true.shape) != tuple(y_pred.shape):
        raise ValueError(
            "y_true and y_pred must have the same shape. "
            f"Got y_true.shape={tuple(y_true.shape)} and "
            f"y_pred.shape={tuple(y_pred.shape)}."
        )


def _elementwise_error_torch(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    """Return elementwise reconstruction error for tensors."""
    diff = y_pred - y_true

    if metric == "mae":
        return torch.abs(diff)

    return diff.pow(2)


def _elementwise_error_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Return elementwise reconstruction error for NumPy arrays."""
    diff = y_pred - y_true

    if metric == "mae":
        return np.abs(diff)

    return diff**2


def _reduce_torch(
    elementwise_error: torch.Tensor,
    metric: str,
    reduction: str,
    eps: float,
) -> torch.Tensor:
    """Reduce elementwise tensor errors."""
    if reduction == "none":
        if metric == "rmse":
            return torch.sqrt(elementwise_error + eps)
        return elementwise_error

    if elementwise_error.ndim <= 1:
        sample_errors = elementwise_error
    else:
        axes = tuple(range(1, elementwise_error.ndim))
        sample_errors = elementwise_error.mean(dim=axes)

    if metric == "rmse":
        sample_errors = torch.sqrt(sample_errors + eps)

    if reduction == "sample":
        return sample_errors
    if reduction == "mean":
        return sample_errors.mean()

    return sample_errors.sum()


def _reduce_numpy(
    elementwise_error: np.ndarray,
    metric: str,
    reduction: str,
    eps: float,
) -> Union[np.ndarray, float]:
    """Reduce elementwise NumPy errors."""
    if reduction == "none":
        if metric == "rmse":
            return np.sqrt(elementwise_error + eps)
        return elementwise_error

    if elementwise_error.ndim <= 1:
        sample_errors = elementwise_error
    else:
        axes = tuple(range(1, elementwise_error.ndim))
        sample_errors = np.mean(elementwise_error, axis=axes)

    if metric == "rmse":
        sample_errors = np.sqrt(sample_errors + eps)

    if reduction == "sample":
        return sample_errors
    if reduction == "mean":
        return float(np.mean(sample_errors))

    return float(np.sum(sample_errors))


def reconstruction_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metric: MetricName = "mse",
    reduction: ReductionName = "mean",
    eps: float = 0.0,
) -> Union[np.ndarray, torch.Tensor, float]:
    """Compute reconstruction error between target and prediction.

    Parameters
    ----------
    y_true : np.ndarray or torch.Tensor
        Target/reference data.
    y_pred : np.ndarray or torch.Tensor
        Reconstructed or predicted data.
    metric : {"mse", "mae", "rmse"}, optional
        Reconstruction metric. Default is "mse".
    reduction : {"none", "sample", "mean", "sum"}, optional
        Reduction mode.

        - "none": return elementwise errors.
        - "sample": return one value per sample, reducing over all axes except
          the first.
        - "mean": return the mean of sample errors.
        - "sum": return the sum of sample errors.

        Default is "mean".
    eps : float, optional
        Small value added inside the square root for RMSE. Default is 0.0.

    Returns
    -------
    np.ndarray, torch.Tensor or float
        Reconstruction error according to the selected metric and reduction.

    Notes
    -----
    When PyTorch tensors are provided, the returned value remains a tensor and
    can be used as a differentiable loss.
    """
    metric, reduction = _normalise_options(metric, reduction)

    if _uses_torch(y_true, y_pred):
        if torch.is_tensor(y_pred):
            y_pred_tensor = y_pred
            y_true_tensor = _to_matching_tensor(y_true, reference=y_pred_tensor)
        else:
            y_true_tensor = y_true
            y_pred_tensor = _to_matching_tensor(y_pred, reference=y_true_tensor)

        _check_same_shape(y_true_tensor, y_pred_tensor)

        elementwise_error = _elementwise_error_torch(
            y_true_tensor,
            y_pred_tensor,
            metric,
        )
        return _reduce_torch(elementwise_error, metric, reduction, eps)

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    _check_same_shape(y_true_array, y_pred_array)

    elementwise_error = _elementwise_error_numpy(
        y_true_array,
        y_pred_array,
        metric,
    )
    return _reduce_numpy(elementwise_error, metric, reduction, eps)


def evaluate_reconstruction(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metric: MetricName = "mse",
    eps: float = 0.0,
) -> Dict[str, float]:
    """Return summary statistics for per-sample reconstruction error.

    Parameters
    ----------
    y_true : np.ndarray or torch.Tensor
        Target/reference data.
    y_pred : np.ndarray or torch.Tensor
        Reconstructed or predicted data.
    metric : {"mse", "mae", "rmse"}, optional
        Reconstruction metric. Default is "mse".
    eps : float, optional
        Small value added inside the square root for RMSE. Default is 0.0.

    Returns
    -------
    dict
        Dictionary containing the metric name, number of samples, mean,
        standard deviation, median, minimum and maximum reconstruction error.
    """
    metric, _ = _normalise_options(metric, "sample")

    sample_errors = reconstruction_error(
        y_true,
        y_pred,
        metric=metric,
        reduction="sample",
        eps=eps,
    )

    values = np.ravel(_to_numpy(sample_errors)).astype(float)

    return {
        "metric": metric,
        "n_samples": int(values.shape[0]),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


class ReconstructionLoss(nn.Module):
    """PyTorch loss wrapper for reconstruction metrics.

    Parameters
    ----------
    metric : {"mse", "mae", "rmse"}, optional
        Reconstruction metric. Default is "mse".
    reduction : {"none", "sample", "mean", "sum"}, optional
        Reduction mode. Default is "mean".
    eps : float, optional
        Small value added inside the square root for RMSE. Default is 0.0.

    Examples
    --------
    >>> criterion = ReconstructionLoss(metric="mse", reduction="mean")
    >>> loss = criterion(y_pred, y_true)
    >>> loss.backward()
    """

    def __init__(
        self,
        metric: MetricName = "mse",
        reduction: ReductionName = "mean",
        eps: float = 0.0,
    ):
        super().__init__()
        self.metric, self.reduction = _normalise_options(metric, reduction)
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted/reconstructed tensor.
        y_true : torch.Tensor
            Target/reference tensor.

        Returns
        -------
        torch.Tensor
            Reconstruction loss.
        """
        loss = reconstruction_error(
            y_true,
            y_pred,
            metric=self.metric,
            reduction=self.reduction,
            eps=self.eps,
        )

        if not torch.is_tensor(loss):
            return torch.as_tensor(loss, dtype=y_pred.dtype, device=y_pred.device)

        return loss
