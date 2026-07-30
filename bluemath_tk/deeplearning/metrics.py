"""Metrics and losses for deeplearning models.

This module provides reconstruction metrics that can be used both for
post-training evaluation and, when using PyTorch tensors, as differentiable
training losses.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Literal

import numpy as np
import torch
from torch import nn

ArrayLike = np.ndarray | torch.Tensor
MetricName = Literal["mse", "mae", "rmse"]
ReductionName = Literal["none", "sample", "mean", "sum"]


__all__ = [
    "reconstruction_error",
    "evaluate_reconstruction",
    "ReconstructionLoss",
]


def _normalise_options(metric: str, reduction: str) -> tuple[str, str]:
    """Validate and normalise metric/reduction names."""
    if not isinstance(metric, str):
        raise TypeError("metric must be a string.")
    if not isinstance(reduction, str):
        raise TypeError("reduction must be a string.")
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


def _validate_eps(eps: float) -> float:
    """Return a finite, non-negative real scalar epsilon."""
    if (
        not isinstance(eps, Real)
        or isinstance(eps, (bool, np.bool_))
        or not math.isfinite(float(eps))
        or eps < 0
    ):
        raise ValueError("eps must be a finite, non-negative real scalar.")
    return float(eps)


def _uses_torch(y_true: ArrayLike, y_pred: ArrayLike) -> bool:
    """Return True when at least one input is a PyTorch tensor."""
    return torch.is_tensor(y_true) or torch.is_tensor(y_pred)


def _validate_array_like(array: ArrayLike, name: str) -> None:
    """Require a finite, non-empty real NumPy array or PyTorch tensor."""
    if isinstance(array, np.ndarray):
        if array.ndim < 1:
            raise ValueError(f"{name} must include a sample dimension.")
        if any(dimension < 1 for dimension in array.shape):
            raise ValueError(f"{name} dimensions must all be non-empty.")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} must contain numeric values.")
        if np.issubdtype(array.dtype, np.complexfloating):
            raise TypeError(f"{name} must contain real-valued data.")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values.")
        return

    if torch.is_tensor(array):
        if array.layout != torch.strided or array.is_quantized:
            raise TypeError(f"{name} must be a dense numeric tensor.")
        if array.ndim < 1:
            raise ValueError(f"{name} must include a sample dimension.")
        if any(dimension < 1 for dimension in array.shape):
            raise ValueError(f"{name} dimensions must all be non-empty.")
        if array.dtype == torch.bool:
            raise TypeError(f"{name} must contain numeric values.")
        if array.is_complex():
            raise TypeError(f"{name} must contain real-valued data.")
        if not torch.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values.")
        return

    raise TypeError(f"{name} must be a NumPy array or PyTorch tensor.")


def _as_metric_tensor(array: torch.Tensor) -> torch.Tensor:
    """Use floating arithmetic for integral tensor metrics."""
    if array.is_floating_point():
        return array
    return array.to(dtype=torch.float64)


def _as_metric_array(array: np.ndarray) -> np.ndarray:
    """Use floating arithmetic for integral NumPy metrics."""
    if np.issubdtype(array.dtype, np.floating):
        return array
    return array.astype(np.float64)


def _to_matching_tensor(
    array: ArrayLike,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert an array to a tensor matching a reference tensor if provided."""
    if torch.is_tensor(array):
        return array

    if reference is not None:
        converted = torch.as_tensor(
            array,
            dtype=reference.dtype,
            device=reference.device,
        )
        if not torch.isfinite(converted).all():
            raise FloatingPointError(
                "Metric input is not finite after conversion to the tensor "
                "dtype and device."
            )
        return converted

    return torch.as_tensor(array)


def _to_numpy(array: ArrayLike) -> np.ndarray:
    """Convert NumPy arrays or tensors to NumPy arrays for summary statistics."""
    if torch.is_tensor(array):
        return array.detach().to(device="cpu", dtype=torch.float64).numpy()

    return np.asarray(array)


def _check_same_shape(y_true: ArrayLike, y_pred: ArrayLike) -> None:
    """Raise a ValueError when target and prediction shapes differ."""
    if tuple(y_true.shape) != tuple(y_pred.shape):
        raise ValueError(
            "y_true and y_pred must have the same shape. "
            f"Got y_true.shape={tuple(y_true.shape)} and "
            f"y_pred.shape={tuple(y_pred.shape)}."
        )


def _require_finite_result(result, phase: str) -> None:
    """Reject non-finite intermediate or final metric arithmetic."""
    if torch.is_tensor(result):
        finite = torch.isfinite(result).all()
    else:
        finite = np.isfinite(result).all()
    if not finite:
        raise FloatingPointError(f"{phase} produced non-finite values.")


def _stable_mean_torch(
    values: torch.Tensor,
    dimensions: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Return a max-scaled mean without accumulating unscaled values."""
    magnitudes = torch.abs(values.detach())
    if dimensions is None:
        scale = torch.amax(magnitudes)
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return scale * torch.mean(values / safe_scale)

    scale = torch.amax(magnitudes, dim=dimensions, keepdim=True)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalised_mean = torch.mean(values / safe_scale, dim=dimensions)
    reduced_scale = scale
    for dimension in sorted(dimensions, reverse=True):
        reduced_scale = reduced_scale.squeeze(dimension)
    return reduced_scale * normalised_mean


def _stable_mean_numpy(
    values: np.ndarray,
    axes: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Return a max-scaled NumPy mean without unscaled accumulation."""
    scale = np.max(np.abs(values), axis=axes, keepdims=True)
    safe_scale = np.where(scale > 0, scale, np.ones_like(scale))
    normalised_mean = np.mean(values / safe_scale, axis=axes)
    if axes is None:
        reduced_scale = np.squeeze(scale)
    else:
        reduced_scale = np.squeeze(scale, axis=axes)
    return reduced_scale * normalised_mean


def _reduce_torch(
    elementwise_error: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """Reduce elementwise MSE or MAE tensor errors."""
    if reduction == "none":
        return elementwise_error

    if elementwise_error.ndim <= 1:
        sample_errors = elementwise_error
    else:
        axes = tuple(range(1, elementwise_error.ndim))
        sample_errors = _stable_mean_torch(elementwise_error, axes)

    if reduction == "sample":
        return sample_errors
    if reduction == "mean":
        return _stable_mean_torch(sample_errors)

    return sample_errors.sum()


def _reduce_numpy(
    elementwise_error: np.ndarray,
    reduction: str,
) -> np.ndarray | float:
    """Reduce elementwise MSE or MAE NumPy errors."""
    if reduction == "none":
        return elementwise_error

    if elementwise_error.ndim <= 1:
        sample_errors = elementwise_error
    else:
        axes = tuple(range(1, elementwise_error.ndim))
        sample_errors = _stable_mean_numpy(elementwise_error, axes)

    if reduction == "sample":
        return sample_errors
    if reduction == "mean":
        return float(_stable_mean_numpy(sample_errors))

    return float(np.sum(sample_errors))


def _stable_mse_torch(
    difference: torch.Tensor,
    dimensions: tuple[int, ...],
) -> torch.Tensor:
    """Return a max-scaled tensor MSE over the requested dimensions."""
    scale = torch.amax(
        torch.abs(difference.detach()),
        dim=dimensions,
        keepdim=True,
    )
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    element_count = math.prod(difference.shape[dimension] for dimension in dimensions)
    scaled_square_sum = torch.sum(
        ((difference / safe_scale) * difference) / element_count,
        dim=dimensions,
    )
    reduced_scale = scale
    for dimension in sorted(dimensions, reverse=True):
        reduced_scale = reduced_scale.squeeze(dimension)
    return reduced_scale * scaled_square_sum


def _mse_torch(difference: torch.Tensor, reduction: str) -> torch.Tensor:
    """Compute tensor MSE without squaring an unscaled reduction group."""
    if reduction == "none":
        return difference.square()
    if reduction == "mean":
        return _stable_mse_torch(difference, tuple(range(difference.ndim)))

    if difference.ndim <= 1:
        sample_errors = difference.square()
    else:
        axes = tuple(range(1, difference.ndim))
        sample_errors = _stable_mse_torch(difference, axes)

    if reduction == "sample":
        return sample_errors
    return sample_errors.sum()


def _stable_mse_numpy(
    difference: np.ndarray,
    axes: tuple[int, ...],
) -> np.ndarray:
    """Return a max-scaled NumPy MSE over the requested axes."""
    scale = np.max(np.abs(difference), axis=axes, keepdims=True)
    safe_scale = np.where(scale > 0, scale, np.ones_like(scale))
    element_count = math.prod(difference.shape[axis] for axis in axes)
    scaled_square_sum = np.sum(
        ((difference / safe_scale) * difference) / element_count,
        axis=axes,
    )
    reduced_scale = np.squeeze(scale, axis=axes)
    return reduced_scale * scaled_square_sum


def _mse_numpy(difference: np.ndarray, reduction: str) -> np.ndarray | float:
    """Compute NumPy MSE without squaring an unscaled reduction group."""
    if reduction == "none":
        return difference**2
    if reduction == "mean":
        result = _stable_mse_numpy(difference, tuple(range(difference.ndim)))
        return float(result)

    if difference.ndim <= 1:
        sample_errors = difference**2
    else:
        axes = tuple(range(1, difference.ndim))
        sample_errors = _stable_mse_numpy(difference, axes)

    if reduction == "sample":
        return sample_errors
    return float(np.sum(sample_errors))


def _rmse_torch(
    difference: torch.Tensor,
    reduction: str,
    eps: float,
) -> torch.Tensor:
    """Compute exact, autograd-safe tensor RMSE reductions."""
    if reduction == "none" or difference.ndim <= 1:
        errors = torch.abs(difference)
    else:
        flattened = difference.reshape(difference.shape[0], -1)
        scale = torch.amax(torch.abs(flattened.detach()), dim=1, keepdim=True)
        nonzero = scale > 0
        safe_scale = torch.where(nonzero, scale, torch.ones_like(scale))
        normalised = flattened / safe_scale
        normalised_mean_square = normalised.square().mean(dim=1)

        nonzero = nonzero.squeeze(1)
        safe_mean_square = torch.where(
            nonzero,
            normalised_mean_square,
            torch.ones_like(normalised_mean_square),
        )
        errors = scale.squeeze(1) * torch.sqrt(safe_mean_square)

    if eps > 0:
        root_eps = torch.full_like(errors, math.sqrt(eps))
        errors = torch.hypot(errors, root_eps)

    if reduction in {"none", "sample"}:
        return errors
    if reduction == "mean":
        return _stable_mean_torch(errors)
    return errors.sum()


def _rmse_numpy(
    difference: np.ndarray,
    reduction: str,
    eps: float,
) -> np.ndarray | float:
    """Compute numerically stable NumPy RMSE reductions."""
    if reduction == "none" or difference.ndim <= 1:
        errors = np.abs(difference)
    else:
        flattened = difference.reshape(difference.shape[0], -1)
        scaled = np.abs(flattened) / math.sqrt(flattened.shape[1])
        errors = np.hypot.reduce(scaled, axis=1)

    if eps > 0:
        errors = np.hypot(errors, math.sqrt(eps))

    if reduction in {"none", "sample"}:
        return errors
    if reduction == "mean":
        return float(_stable_mean_numpy(errors))
    return float(np.sum(errors))


def reconstruction_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metric: MetricName = "mse",
    reduction: ReductionName = "mean",
    eps: float = 0.0,
) -> np.ndarray | torch.Tensor | float:
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
    eps = _validate_eps(eps)
    _validate_array_like(y_true, "y_true")
    _validate_array_like(y_pred, "y_pred")
    _check_same_shape(y_true, y_pred)

    if _uses_torch(y_true, y_pred):
        if torch.is_tensor(y_pred):
            y_pred_tensor = _as_metric_tensor(y_pred)
            y_true_tensor = _to_matching_tensor(y_true, reference=y_pred_tensor)
        else:
            y_true_tensor = _as_metric_tensor(y_true)
            y_pred_tensor = _to_matching_tensor(y_pred, reference=y_true_tensor)

        difference = y_pred_tensor - y_true_tensor
        _require_finite_result(difference, "Metric subtraction")
        if metric == "rmse":
            result = _rmse_torch(difference, reduction, eps)
        elif metric == "mse":
            result = _mse_torch(difference, reduction)
        else:
            elementwise_error = torch.abs(difference)
            _require_finite_result(elementwise_error, "Metric arithmetic")
            result = _reduce_torch(elementwise_error, reduction)
        _require_finite_result(result, "Metric reduction")
        return result

    y_true_array = _as_metric_array(y_true)
    y_pred_array = _as_metric_array(y_pred)
    with np.errstate(over="ignore", invalid="ignore"):
        difference = y_pred_array - y_true_array
        _require_finite_result(difference, "Metric subtraction")
        if metric == "rmse":
            result = _rmse_numpy(difference, reduction, eps)
        elif metric == "mse":
            result = _mse_numpy(difference, reduction)
        else:
            elementwise_error = np.abs(difference)
            _require_finite_result(elementwise_error, "Metric arithmetic")
            result = _reduce_numpy(elementwise_error, reduction)
    _require_finite_result(result, "Metric reduction")
    return result


def evaluate_reconstruction(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metric: MetricName = "mse",
    eps: float = 0.0,
) -> dict[str, float]:
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
    eps = _validate_eps(eps)

    sample_errors = reconstruction_error(
        y_true,
        y_pred,
        metric=metric,
        reduction="sample",
        eps=eps,
    )

    values = np.ravel(_to_numpy(sample_errors)).astype(float)

    with np.errstate(over="ignore", invalid="ignore"):
        summary = {
            "metric": metric,
            "n_samples": int(values.shape[0]),
            "mean": float(_stable_mean_numpy(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    if not all(
        math.isfinite(summary[name]) for name in ("mean", "std", "median", "min", "max")
    ):
        raise FloatingPointError("Reconstruction summary produced non-finite values.")
    return summary


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
        self.eps = _validate_eps(eps)

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
