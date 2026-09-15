"""Reusable latent-structure regularization for BlueMath_tk autoencoders.

The feature is deliberately architecture-agnostic. Existing encoder projection
layers remain unchanged; this module operates on the resulting latent scores
and, optionally, on the projection weight tensor.

Modes
-----
none
    Exact pass-through. No regularization and no ordered masking.
orthogonal
    Penalize non-orthogonality of the final latent projection and latent
    cross-correlation.
pca_like
    Same penalties as ``orthogonal`` plus ordered nested latent dropout during
    training. Earlier latent coordinates are therefore required to be useful
    more often than later coordinates.

This does NOT make a nonlinear autoencoder equivalent to PCA. The purpose is
to impose PCA-like geometric structure while preserving the nonlinear encoder
and decoder.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

LATENT_STRUCTURE_MODES = ("none", "orthogonal", "pca_like")


def _validate_nonnegative_finite(name: str, value: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative number.")
    return float(value)


def _validate_probability(name: str, value: float) -> float:
    value = _validate_nonnegative_finite(name, value)
    if value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return value


def validate_latent_structure_mode(mode: str) -> str:
    """Validate and normalize a latent-structure mode."""
    if not isinstance(mode, str):
        raise TypeError("latent_structure must be a string.")
    normalized = mode.strip().lower()
    if normalized not in LATENT_STRUCTURE_MODES:
        allowed = ", ".join(repr(item) for item in LATENT_STRUCTURE_MODES)
        raise ValueError(f"latent_structure must be one of: {allowed}.")
    return normalized


class LatentStructureRegularizer(nn.Module):
    """Pass-through latent regularizer with optional ordered masking.

    Parameters
    ----------
    k : int
        Latent dimension.
    mode : {"none", "orthogonal", "pca_like"}
        Structural mode.
    orthogonality_weight : float
        Weight applied to the normalized projection orthogonality penalty.
    decorrelation_weight : float
        Weight applied to the off-diagonal latent correlation penalty.
    ordering_probability : float
        In ``pca_like`` mode and training mode, probability that each sample is
        reconstructed from a random prefix of its latent vector. A value of
        ``0.5`` means roughly half of samples use a shortened prefix while the
        remainder retain all ``k`` coordinates.
    eps : float
        Dimensionless tolerance used to identify collapsed coordinates after rescaling.

    Notes
    -----
    This module owns no trainable parameters or persistent buffers. Therefore,
    adding it does not change the default model state_dict when mode="none",
    and it can be introduced without invalidating legacy parameter tensors.

    The projection orthogonality penalty is

        mean((W W^T - I)^2)

    and is only defined when the latent projection weight is supplied.

    The latent decorrelation penalty is the mean squared off-diagonal correlation
    of the *unmasked* latent scores. Ordered dropout is applied only after the
    regularization terms have been computed.
    """

    _bluemath_latent_regularizer = True

    def __init__(
        self,
        k: int,
        mode: str = "none",
        orthogonality_weight: float = 1e-2,
        decorrelation_weight: float = 1e-2,
        ordering_probability: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__()
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer.")
        self.k = k
        self.mode = validate_latent_structure_mode(mode)
        self.orthogonality_weight = _validate_nonnegative_finite(
            "orthogonality_weight", orthogonality_weight
        )
        self.decorrelation_weight = _validate_nonnegative_finite(
            "decorrelation_weight", decorrelation_weight
        )
        self.ordering_probability = _validate_probability(
            "ordering_probability", ordering_probability
        )
        self.eps = _validate_nonnegative_finite("eps", eps)
        if self.eps == 0.0:
            raise ValueError("eps must be greater than zero.")

        self._last_losses: dict[str, torch.Tensor] = {}

    @property
    def enabled(self) -> bool:
        """Return whether latent-structure regularization is enabled."""
        return self.mode != "none"

    def _validate_latent(self, z: torch.Tensor) -> None:
        if not isinstance(z, torch.Tensor):
            raise TypeError("Latent scores must be a PyTorch tensor.")
        if z.ndim != 2:
            raise ValueError(
                "Latent scores must have shape "
                f"(batch, {self.k}); got {tuple(z.shape)}."
            )
        if z.shape[1] != self.k:
            raise ValueError(
                f"Latent scores must have width {self.k}; got {z.shape[1]}."
            )
        if not torch.isfinite(z).all():
            raise FloatingPointError("Latent scores are not finite.")

    def _orthogonality_loss(
        self,
        projection_weight: torch.Tensor | None,
        z: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.orthogonality_weight == 0.0:
            return None
        if projection_weight is None:
            raise ValueError(
                "projection_weight is required when latent orthogonality is enabled."
            )
        if not isinstance(projection_weight, torch.Tensor):
            raise TypeError("projection_weight must be a PyTorch tensor.")
        if projection_weight.ndim != 2:
            raise ValueError("projection_weight must be a 2D tensor.")
        if projection_weight.shape[0] != self.k:
            raise ValueError(
                "projection_weight must have shape "
                f"({self.k}, in_features); got {tuple(projection_weight.shape)}."
            )
        if projection_weight.shape[1] < self.k:
            raise ValueError(
                "Exact row orthogonality requires latent input dimension >= k; "
                f"got in_features={projection_weight.shape[1]} and k={self.k}."
            )
        if not torch.isfinite(projection_weight).all():
            raise FloatingPointError("projection_weight must be finite.")

        gram = projection_weight @ projection_weight.transpose(0, 1)
        identity = torch.eye(
            self.k,
            dtype=gram.dtype,
            device=gram.device,
        )
        raw = torch.mean((gram - identity) ** 2)
        return raw.to(dtype=z.dtype) * self.orthogonality_weight

    def _decorrelation_loss(self, z: torch.Tensor) -> torch.Tensor | None:
        """Return a scale-invariant latent-correlation penalty.

        Noncollapsed coordinates are centered, independently rescaled, and
        normalized to unit Euclidean norm. Their dot products are Pearson
        correlations, so the loss is invariant to any representable nonzero
        per-coordinate scaling.

        Pearson correlation is undefined for an exactly collapsed coordinate.
        To prevent collapse from reducing the objective, every off-diagonal
        pair involving such a coordinate receives the maximal squared-
        correlation cost of 1.0.
        """
        if (
            self.decorrelation_weight == 0.0
            or z.shape[0] < 2
            or self.k < 2
        ):
            return None

        centered = z - z.mean(dim=0, keepdim=True)
        coordinate_scale = torch.amax(torch.abs(centered), dim=0)
        noncollapsed = coordinate_scale > 0
        safe_scale = torch.where(
            noncollapsed,
            coordinate_scale,
            torch.ones_like(coordinate_scale),
        )
        scaled = centered / safe_scale

        coordinate_norm = torch.linalg.vector_norm(scaled, dim=0)
        active = noncollapsed & (coordinate_norm > self.eps)
        safe_norm = torch.where(
            active,
            coordinate_norm,
            torch.ones_like(coordinate_norm),
        )
        normalized = scaled / safe_norm

        correlation = normalized.transpose(0, 1) @ normalized
        squared_correlation = torch.clamp(correlation**2, max=1.0)
        pair_active = active.unsqueeze(1) & active.unsqueeze(0)
        guarded_squared_correlation = torch.where(
            pair_active,
            squared_correlation,
            torch.ones_like(squared_correlation),
        )

        off_diagonal_mask = ~torch.eye(
            self.k,
            dtype=torch.bool,
            device=z.device,
        )
        raw = torch.mean(
            guarded_squared_correlation[off_diagonal_mask]
        )
        return raw * self.decorrelation_weight

    def _compute_losses(
        self,
        z: torch.Tensor,
        projection_weight: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if not self.enabled:
            return {}

        losses: dict[str, torch.Tensor] = {}
        orthogonality = self._orthogonality_loss(projection_weight, z)
        if orthogonality is not None:
            losses["latent_orthogonality"] = orthogonality

        decorrelation = self._decorrelation_loss(z)
        if decorrelation is not None:
            losses["latent_decorrelation"] = decorrelation

        return losses

    def apply_ordering(self, z: torch.Tensor) -> torch.Tensor:
        """Apply per-sample nested latent dropout in training mode only."""
        self._validate_latent(z)
        if (
            self.mode != "pca_like"
            or not self.training
            or self.ordering_probability == 0.0
            or self.k == 1
        ):
            return z

        batch_size = z.shape[0]
        shorten = (
            torch.rand(batch_size, device=z.device) < self.ordering_probability
        )
        # Prefix lengths 1..k-1 for shortened samples. Non-shortened samples
        # keep all k coordinates.
        short_prefixes = torch.randint(
            low=1,
            high=self.k,
            size=(batch_size,),
            device=z.device,
        )
        full_prefixes = torch.full(
            (batch_size,),
            self.k,
            dtype=torch.long,
            device=z.device,
        )
        prefix_lengths = torch.where(shorten, short_prefixes, full_prefixes)
        coordinates = torch.arange(self.k, device=z.device).unsqueeze(0)
        mask = coordinates < prefix_lengths.unsqueeze(1)
        return z * mask.to(dtype=z.dtype)

    def forward(
        self,
        z: torch.Tensor,
        projection_weight: torch.Tensor | None = None,
        *,
        apply_ordering: bool = True,
    ) -> torch.Tensor:
        """Register current regularization losses and return structured scores."""
        self._validate_latent(z)
        if not self.enabled:
            self._last_losses = {}
            return z

        self._last_losses = self._compute_losses(z, projection_weight)
        if apply_ordering:
            return self.apply_ordering(z)
        return z

    def regularization_losses(self) -> dict[str, torch.Tensor]:
        """Return losses from the most recent forward pass."""
        return dict(self._last_losses)

    def extra_repr(self) -> str:
        """Return a concise module configuration representation."""
        return (
            f"k={self.k}, mode={self.mode!r}, "
            f"orthogonality_weight={self.orthogonality_weight}, "
            f"decorrelation_weight={self.decorrelation_weight}, "
            f"ordering_probability={self.ordering_probability}"
        )



class StructuredLatentLinear(nn.Linear):
    """Linear projection that keeps legacy weight/bias state-dict names."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        mode: str = "none",
        orthogonality_weight: float = 1e-2,
        decorrelation_weight: float = 1e-2,
        ordering_probability: float = 0.5,
        apply_ordering_in_forward: bool = True,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.latent_regularizer = LatentStructureRegularizer(
            k=out_features,
            mode=mode,
            orthogonality_weight=orthogonality_weight,
            decorrelation_weight=decorrelation_weight,
            ordering_probability=ordering_probability,
        )
        self.apply_ordering_in_forward = bool(apply_ordering_in_forward)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project inputs and apply the configured latent structure."""
        z = super().forward(x)
        return self.latent_regularizer(
            z,
            projection_weight=self.weight,
            apply_ordering=self.apply_ordering_in_forward,
        )

    def apply_ordering(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the configured training-only ordered prefix mask."""
        return self.latent_regularizer.apply_ordering(z)


def projection_orthogonality_error(weight: np.ndarray) -> float:
    """Return normalized Frobenius error of row orthogonality."""
    array = np.asarray(weight, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("weight must be a 2D array.")
    k, in_features = array.shape
    if k < 1 or in_features < 1:
        raise ValueError("weight dimensions must be positive.")
    gram = array @ array.T
    identity = np.eye(k, dtype=np.float64)
    return float(np.sqrt(np.mean((gram - identity) ** 2)))


def compute_latent_diagnostics(
    z: np.ndarray,
    *,
    variance_order_tolerance: float = 1e-12,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Compute interpretation diagnostics for a fitted latent representation.

    The function is intentionally independent of PCA. It reports whether latent
    coordinates are decorrelated and whether variance is monotonically ordered.
    Correlation summaries use only pairs for which both coordinates have
    nonzero centered norm; exactly collapsed coordinates are reported
    separately.

    Returns JSON-compatible values.
    """
    array = np.asarray(z, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("z must have shape (n_samples, k).")
    if array.shape[0] < 2:
        raise ValueError("At least two samples are required.")
    if array.shape[1] < 1:
        raise ValueError("At least one latent dimension is required.")
    if not np.isfinite(array).all():
        raise ValueError("z must contain only finite values.")
    if variance_order_tolerance < 0 or not np.isfinite(variance_order_tolerance):
        raise ValueError("variance_order_tolerance must be finite and non-negative.")
    if eps <= 0 or not np.isfinite(eps):
        raise ValueError("eps must be finite and positive.")

    centered = array - array.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / float(array.shape[0] - 1)
    variance = np.diag(covariance).copy()

    coordinate_scale = np.max(np.abs(centered), axis=0)
    noncollapsed = coordinate_scale > 0.0
    safe_scale = np.where(noncollapsed, coordinate_scale, 1.0)
    scaled = centered / safe_scale

    coordinate_norm = np.linalg.norm(scaled, axis=0)
    active = noncollapsed & (coordinate_norm > eps)
    safe_norm = np.where(active, coordinate_norm, 1.0)
    normalized = scaled / safe_norm
    correlation = np.clip(normalized.T @ normalized, -1.0, 1.0)

    if array.shape[1] == 1:
        mean_abs_offdiag = 0.0
        max_abs_offdiag = 0.0
    else:
        off_diagonal = ~np.eye(array.shape[1], dtype=bool)
        defined_pairs = off_diagonal & np.outer(active, active)
        if np.any(defined_pairs):
            values = np.abs(correlation[defined_pairs])
            mean_abs_offdiag = float(values.mean())
            max_abs_offdiag = float(values.max())
        else:
            mean_abs_offdiag = 0.0
            max_abs_offdiag = 0.0

    variance_total = float(variance.sum())
    if variance_total > 0.0:
        variance_fraction = variance / variance_total
    else:
        variance_fraction = np.zeros_like(variance)

    ordering_violations = int(
        np.sum(
            variance[1:]
            > variance[:-1] + float(variance_order_tolerance)
        )
    )

    return {
        "n_samples": int(array.shape[0]),
        "k": int(array.shape[1]),
        "mean": array.mean(axis=0).tolist(),
        "variance": variance.tolist(),
        "variance_fraction": variance_fraction.tolist(),
        "cumulative_variance_fraction": np.cumsum(variance_fraction).tolist(),
        "mean_abs_offdiag_correlation": mean_abs_offdiag,
        "max_abs_offdiag_correlation": max_abs_offdiag,
        "n_collapsed_coordinates": int(np.sum(~active)),
        "collapsed_coordinates": np.flatnonzero(~active).astype(int).tolist(),
        "variance_ordering_violations": ordering_violations,
        "variance_monotonic_nonincreasing": ordering_violations == 0,
    }


def prefix_latent(z: np.ndarray, n_components: int) -> np.ndarray:
    """Zero all coordinates after ``n_components`` for prefix-reconstruction tests."""
    array = np.asarray(z)
    if array.ndim != 2:
        raise ValueError("z must have shape (n_samples, k).")
    if (
        not isinstance(n_components, int)
        or isinstance(n_components, bool)
        or n_components < 1
        or n_components > array.shape[1]
    ):
        raise ValueError(
            f"n_components must be an integer between 1 and {array.shape[1]}."
        )
    result = array.copy()
    result[:, n_components:] = 0
    return result
