"""Common reconstruction benchmarking for PCA and autoencoder models.

This module provides the infrastructure required to compare dimensionality
reduction methods on exactly the same held-out samples. Membership of the
train, validation, and test partitions always comes from a
:class:`~bluemath_tk.validation.chronological.ChronologicalSplit`, so the
benchmark never creates a random split of its own and the test partition never
reaches any fitting step.

The framework measures *reconstruction* performance only. A low reconstruction
error does not establish that a method is scientifically better for a
downstream task, and the reported latent dimensionality is not a storage
compression ratio.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

import numpy as np
import xarray as xr

from ..datamining.pca import PCA
from ..validation.chronological import (
    ChronologicalSplit,
    JsonValue,
    _canonical_json,
    _freeze_json,
    _thaw_json,
    _validate_json_value,
)

__all__ = [
    "AutoencoderReconstruction",
    "BenchmarkMethod",
    "MethodBenchmarkResult",
    "PCAReconstruction",
    "ReconstructionBenchmarkReport",
    "ReconstructionMethod",
    "autoencoder_benchmark_method",
    "pca_benchmark_method",
    "run_reconstruction_benchmark",
]

_SCHEMA_VERSION = 1
_SUPPORTED_METRICS = ("mae", "mse", "rmse")
_DEFAULT_METRICS = ("mse", "mae", "rmse")
_METRIC_REDUCTION = "mean"
_PCA_VARIABLE = "value"
_PCA_SAMPLE_DIM = "sample"
_PARTITION_NAMES = ("train", "validation", "test", "excluded")


def _load_reconstruction_error() -> Callable[..., Any]:
    """Import the accepted BlueMath reconstruction metric implementation.

    The import is deferred because ``bluemath_tk.deeplearning.metrics``
    requires PyTorch, which is an optional dependency. Importing it lazily
    keeps ``import bluemath_tk`` usable without the deeplearning extra.
    """
    try:
        from ..deeplearning.metrics import reconstruction_error
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ImportError(
            "Reconstruction benchmarking reuses bluemath_tk.deeplearning.metrics, "
            "which requires PyTorch. Install the deeplearning extra with "
            "pip install 'bluemath-tk[deeplearning]'."
        ) from exc
    return reconstruction_error


def _validate_positive_integer(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an exact non-Boolean integer.")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value}.")
    return int(value)


def _validate_non_empty_string(name: str, value: Any) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact built-in string.")
    if not value.strip():
        raise ValueError(f"{name} must not be empty or blank.")
    return value


def _validate_boolean(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact built-in boolean.")
    return value


def _validate_finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _validate_configuration(
    configuration: Any,
    *,
    name: str,
) -> dict[str, JsonValue]:
    """Validate a user-declared, JSON-serializable method configuration."""
    if configuration is None:
        return {}
    if not isinstance(configuration, Mapping):
        raise TypeError(f"{name} must be a mapping of JSON-compatible values.")
    payload = {key: value for key, value in configuration.items()}
    validated = _validate_json_value(payload, path=name)
    if not isinstance(validated, dict):
        raise TypeError(f"{name} must be a JSON object.")
    return validated


def _validate_sample_data(X: Any, *, name: str = "X") -> np.ndarray:
    """Reject datasets the benchmark cannot compare fairly."""
    if not isinstance(X, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")
    if X.ndim < 2:
        raise ValueError(
            f"{name} must include a leading sample dimension and at least one "
            "feature dimension. For tabular data use shape "
            "(n_samples, n_features)."
        )
    if X.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one sample.")
    if any(dimension < 1 for dimension in X.shape[1:]):
        raise ValueError(f"Every per-sample dimension of {name} must be positive.")
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values.")
    if np.issubdtype(X.dtype, np.complexfloating):
        raise TypeError(
            f"{name} must contain real-valued data; the shared reconstruction "
            "metrics do not support complex arrays."
        )
    if not np.isfinite(X).all():
        raise ValueError(f"{name} must not contain NaN or infinite values.")
    return X


def _validate_metrics(metrics: Any) -> tuple[str, ...]:
    """Validate the requested metric names, preserving the requested order."""
    if isinstance(metrics, str):
        raise TypeError("metrics must be a sequence of metric names, not a string.")
    if not isinstance(metrics, Sequence):
        raise TypeError("metrics must be a sequence of metric names.")
    if not metrics:
        raise ValueError("metrics must request at least one metric.")
    names: list[str] = []
    for metric in metrics:
        name = _validate_non_empty_string("metric name", metric)
        if name not in _SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric {name!r}; supported metrics are "
                f"{list(_SUPPORTED_METRICS)}."
            )
        if name in names:
            raise ValueError(f"Duplicate metric requested: {name!r}.")
        names.append(name)
    return tuple(names)


class ReconstructionMethod(Protocol):
    """Structural interface every benchmarked reconstruction model provides.

    A benchmark method is fitted on the training partition, optionally given
    the validation partition, and then asked to reconstruct arbitrary samples
    that share the training per-sample shape.

    Attributes
    ----------
    latent_dimension : int
        Number of latent scalars retained for one sample.
    is_fitted : bool
        Whether this instance has already been fitted. The runner rejects
        instances that are already fitted so that state cannot leak between
        benchmark runs.
    """

    latent_dimension: int
    is_fitted: bool

    def fit(self, X_train: np.ndarray, X_validation: np.ndarray | None) -> None:
        """Fit the method on the training partition only."""
        ...

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Return reconstructions with exactly the shape of ``X``."""
        ...


def _feature_coordinate_names(sample_shape: tuple[int, ...]) -> list[str]:
    return [f"feature_{index}" for index in range(len(sample_shape))]


def _to_pca_dataset(X: np.ndarray) -> xr.Dataset:
    """Wrap ``(n_samples, ...)`` data in the Dataset layout the PCA class needs."""
    sample_shape = tuple(X.shape[1:])
    coordinate_names = _feature_coordinate_names(sample_shape)
    dimensions = [_PCA_SAMPLE_DIM, *coordinate_names]
    coordinates: dict[str, np.ndarray] = {
        _PCA_SAMPLE_DIM: np.arange(X.shape[0]),
    }
    for name, size in zip(coordinate_names, sample_shape):
        coordinates[name] = np.arange(size)
    return xr.Dataset({_PCA_VARIABLE: (dimensions, X)}, coords=coordinates)


class PCAReconstruction:
    """Benchmark adapter around :class:`bluemath_tk.datamining.pca.PCA`.

    Samples of shape ``(n_samples, d1, ..., dm)`` are presented to the existing
    BlueMath PCA implementation as a single stacked variable. Stacking and the
    inverse reshape both use C order, so sample order and the per-sample shape
    survive the round trip unchanged.

    Parameters
    ----------
    n_components : int
        Number of principal components to retain. This is the latent
        dimensionality used for comparison against autoencoders.
    scale_data : bool, optional
        When True, the BlueMath PCA standardizes features with a
        ``StandardScaler`` fitted on the training partition only. Default is
        False, which leaves only the centering that scikit-learn's PCA performs
        intrinsically. The conservative default keeps the comparison against
        autoencoders free of preprocessing that they do not receive.

    Notes
    -----
    PCA has no early stopping and no validation-driven model selection, so this
    adapter ignores the validation partition entirely and fits on the training
    partition alone.
    """

    def __init__(self, n_components: int, *, scale_data: bool = False):
        self.n_components = _validate_positive_integer("n_components", n_components)
        self.scale_data = _validate_boolean("scale_data", scale_data)
        self._pca: PCA | None = None
        self._sample_shape: tuple[int, ...] | None = None

    @property
    def latent_dimension(self) -> int:
        """Return the number of retained principal components."""
        return self.n_components

    @property
    def is_fitted(self) -> bool:
        """Return whether the underlying PCA model has been fitted."""
        return self._pca is not None and bool(self._pca.is_fitted)

    @property
    def pca(self) -> PCA:
        """Return the fitted BlueMath PCA instance."""
        if self._pca is None:
            raise ValueError("The PCA benchmark method has not been fitted yet.")
        return self._pca

    def fit(self, X_train: np.ndarray, X_validation: np.ndarray | None = None) -> None:
        """Fit PCA on the training partition only.

        Parameters
        ----------
        X_train : np.ndarray
            Training samples with shape ``(n_train, ...)``.
        X_validation : np.ndarray, optional
            Ignored. PCA performs no validation-driven model selection.
        """
        _validate_sample_data(X_train, name="X_train")
        sample_shape = tuple(X_train.shape[1:])
        n_features = int(np.prod(sample_shape))
        available = min(int(X_train.shape[0]), n_features)
        if self.n_components > available:
            raise ValueError(
                f"n_components={self.n_components} exceeds the {available} "
                "components available from a training partition of "
                f"{X_train.shape[0]} samples with {n_features} scalars per "
                "sample."
            )

        self._sample_shape = sample_shape
        self._pca = PCA(n_components=self.n_components)
        self._pca.fit(
            data=_to_pca_dataset(X_train),
            vars_to_stack=[_PCA_VARIABLE],
            coords_to_stack=_feature_coordinate_names(sample_shape),
            pca_dim_for_rows=_PCA_SAMPLE_DIM,
            scale_data=self.scale_data,
        )

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Project ``X`` onto the fitted components and invert the projection."""
        if self._pca is None or self._sample_shape is None:
            raise ValueError(
                "The PCA benchmark method must be fitted before reconstructing."
            )
        _validate_sample_data(X, name="X")
        if tuple(X.shape[1:]) != self._sample_shape:
            raise ValueError(
                f"Expected per-sample shape {self._sample_shape}, got "
                f"{tuple(X.shape[1:])}."
            )
        principal_components = self._pca.transform(data=_to_pca_dataset(X))
        reconstructed = self._pca.inverse_transform(PCs=principal_components)
        return np.asarray(reconstructed[_PCA_VARIABLE].values, dtype=np.float64)


class AutoencoderReconstruction:
    """Benchmark adapter around a BlueMath autoencoder.

    The adapter uses the accepted public workflow only: ``model.fit(...)`` with
    explicit chronological validation data, and ``model.predict(...)`` for
    deterministic reconstruction. Model architectures are never modified.

    Parameters
    ----------
    model : object
        An unfitted BlueMath autoencoder exposing ``fit`` and ``predict``.
    latent_dimension : int
        The latent width declared for this model. When the model exposes ``k``
        the two values must agree.
    fit_kwargs : dict, optional
        Extra keyword arguments forwarded to ``model.fit``. ``validation_data``
        and ``validation_split`` are rejected because the benchmark controls
        partition membership. ``verbose`` defaults to 0.
    predict_kwargs : dict, optional
        Extra keyword arguments forwarded to ``model.predict``. ``verbose``
        defaults to 0.

    Notes
    -----
    For variational autoencoders, ``predict`` defaults to the deterministic
    posterior-mean reconstruction. ``stochastic=True`` is rejected here because
    comparing a single stochastic draw against deterministic PCA and
    autoencoder reconstructions is not a like-for-like measurement.
    """

    _FORBIDDEN_FIT_KWARGS = ("X", "y", "validation_data", "validation_split")

    def __init__(
        self,
        model: Any,
        latent_dimension: int,
        *,
        fit_kwargs: Mapping[str, Any] | None = None,
        predict_kwargs: Mapping[str, Any] | None = None,
    ):
        for attribute in ("fit", "predict"):
            if not callable(getattr(model, attribute, None)):
                raise TypeError(
                    f"model must expose a callable {attribute}() method to be "
                    "benchmarked as an autoencoder."
                )
        self.model = model
        self._latent_dimension = _validate_positive_integer(
            "latent_dimension",
            latent_dimension,
        )
        declared_k = getattr(model, "k", None)
        if declared_k is not None and int(declared_k) != self._latent_dimension:
            raise ValueError(
                f"latent_dimension={self._latent_dimension} contradicts the "
                f"model latent width k={int(declared_k)}."
            )

        self._fit_kwargs = dict(fit_kwargs or {})
        forbidden = sorted(
            set(self._FORBIDDEN_FIT_KWARGS).intersection(self._fit_kwargs)
        )
        if forbidden:
            raise ValueError(
                "The benchmark controls partition membership; remove these "
                f"fit_kwargs: {forbidden}."
            )
        self._fit_kwargs.setdefault("verbose", 0)

        self._predict_kwargs = dict(predict_kwargs or {})
        if self._predict_kwargs.get("stochastic"):
            raise ValueError(
                "Stochastic reconstruction is not comparable with the "
                "deterministic PCA and autoencoder reconstructions used by this "
                "benchmark. Remove stochastic=True from predict_kwargs."
            )
        self._predict_kwargs.setdefault("verbose", 0)
        self._history: dict[str, list] | None = None

    @property
    def latent_dimension(self) -> int:
        """Return the declared latent width of the wrapped model."""
        return self._latent_dimension

    @property
    def is_fitted(self) -> bool:
        """Return whether the wrapped model reports itself as fitted."""
        return bool(getattr(self.model, "is_fitted", False))

    @property
    def history(self) -> dict[str, list] | None:
        """Return the training history returned by the last fit call."""
        return self._history

    def fit(self, X_train: np.ndarray, X_validation: np.ndarray | None = None) -> None:
        """Fit on the training partition, validating on the given samples only.

        Parameters
        ----------
        X_train : np.ndarray
            Training samples. Every one of them is used for optimisation.
        X_validation : np.ndarray
            Validation samples. Exactly these samples drive the validation loss
            and early stopping.
        """
        _validate_sample_data(X_train, name="X_train")
        if X_validation is None:
            raise ValueError(
                "Autoencoder benchmarking requires the validation partition for "
                "early stopping. Declare uses_validation_partition=True."
            )
        _validate_sample_data(X_validation, name="X_validation")
        self._history = self.model.fit(
            X_train,
            validation_data=(X_validation, None),
            **self._fit_kwargs,
        )

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Return the model's deterministic reconstruction of ``X``."""
        _validate_sample_data(X, name="X")
        return np.asarray(self.model.predict(X, **self._predict_kwargs))


@dataclass(frozen=True)
class BenchmarkMethod:
    """Reproducible specification of one benchmarked reconstruction method.

    Attributes
    ----------
    name : str
        Unique, human-readable identifier used in the benchmark report.
    method_type : str
        Explicit, user-supplied family label such as ``"pca"`` or
        ``"autoencoder"``. The benchmark never infers this by introspecting the
        factory, because callables cannot be serialized reproducibly.
    latent_dimension : int
        Latent scalars retained per sample. The runner cross-checks this
        against the value reported by the constructed method.
    factory : callable
        Zero-argument callable returning a fresh, unfitted
        :class:`ReconstructionMethod`. A new instance is built for every run so
        that fitted state cannot leak between runs.
    configuration : mapping, optional
        JSON-compatible description of the method configuration, recorded
        verbatim in the report.
    uses_validation_partition : bool, optional
        Whether the method consumes the validation partition, for example for
        early stopping. Default is True. PCA declares False because it performs
        no validation-driven model selection.
    """

    name: str
    method_type: str
    latent_dimension: int
    factory: Callable[[], ReconstructionMethod]
    configuration: Mapping[str, JsonValue] = field(default_factory=dict)
    uses_validation_partition: bool = True

    def __post_init__(self) -> None:
        """Validate and freeze the specification after construction."""
        object.__setattr__(self, "name", _validate_non_empty_string("name", self.name))
        object.__setattr__(
            self,
            "method_type",
            _validate_non_empty_string("method_type", self.method_type),
        )
        object.__setattr__(
            self,
            "latent_dimension",
            _validate_positive_integer("latent_dimension", self.latent_dimension),
        )
        if not callable(self.factory):
            raise TypeError("factory must be a zero-argument callable.")
        object.__setattr__(
            self,
            "uses_validation_partition",
            _validate_boolean(
                "uses_validation_partition",
                self.uses_validation_partition,
            ),
        )
        object.__setattr__(
            self,
            "configuration",
            _freeze_json(
                _validate_configuration(self.configuration, name="configuration")
            ),
        )


def pca_benchmark_method(
    name: str,
    *,
    n_components: int,
    scale_data: bool = False,
) -> BenchmarkMethod:
    """Build a PCA benchmark specification using the existing BlueMath PCA.

    Parameters
    ----------
    name : str
        Unique identifier for this method in the report.
    n_components : int
        Number of principal components, used as the latent dimensionality.
    scale_data : bool, optional
        Standardize features using a scaler fitted on the training partition
        only. Default is False.

    Returns
    -------
    BenchmarkMethod
        A specification whose factory builds a fresh
        :class:`PCAReconstruction`.
    """
    components = _validate_positive_integer("n_components", n_components)
    scale = _validate_boolean("scale_data", scale_data)

    def factory() -> ReconstructionMethod:
        return PCAReconstruction(n_components=components, scale_data=scale)

    return BenchmarkMethod(
        name=name,
        method_type="pca",
        latent_dimension=components,
        factory=factory,
        configuration={
            "implementation": "bluemath_tk.datamining.pca.PCA",
            "n_components": components,
            "scale_data": scale,
        },
        uses_validation_partition=False,
    )


def autoencoder_benchmark_method(
    name: str,
    *,
    model_factory: Callable[[], Any],
    latent_dimension: int,
    configuration: Mapping[str, JsonValue] | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
    predict_kwargs: Mapping[str, Any] | None = None,
) -> BenchmarkMethod:
    """Build an autoencoder benchmark specification.

    Parameters
    ----------
    name : str
        Unique identifier for this method in the report.
    model_factory : callable
        Zero-argument callable returning a fresh, unfitted BlueMath
        autoencoder. Supplying a factory rather than an instance guarantees
        that no fitted state is shared between runs.
    latent_dimension : int
        Latent width of the model, cross-checked against ``model.k`` when
        available.
    configuration : mapping, optional
        JSON-compatible description of the architecture and hyperparameters.
        This is recorded verbatim; the factory itself is never introspected.
    fit_kwargs : mapping, optional
        Extra keyword arguments for ``model.fit``.
    predict_kwargs : mapping, optional
        Extra keyword arguments for ``model.predict``.

    Returns
    -------
    BenchmarkMethod
        A specification whose factory builds a fresh
        :class:`AutoencoderReconstruction`.
    """
    if not callable(model_factory):
        raise TypeError("model_factory must be a zero-argument callable.")
    width = _validate_positive_integer("latent_dimension", latent_dimension)
    frozen_fit_kwargs = dict(fit_kwargs or {})
    frozen_predict_kwargs = dict(predict_kwargs or {})

    def factory() -> ReconstructionMethod:
        return AutoencoderReconstruction(
            model_factory(),
            latent_dimension=width,
            fit_kwargs=frozen_fit_kwargs,
            predict_kwargs=frozen_predict_kwargs,
        )

    return BenchmarkMethod(
        name=name,
        method_type="autoencoder",
        latent_dimension=width,
        factory=factory,
        configuration=configuration,
        uses_validation_partition=True,
    )


@dataclass(frozen=True)
class MethodBenchmarkResult:
    """Reconstruction result for one method on the test partition.

    Attributes
    ----------
    name : str
        Method identifier.
    method_type : str
        Declared method family.
    latent_dimension : int
        Latent scalars retained per sample.
    uses_validation_partition : bool
        Whether the method consumed the validation partition.
    original_scalars_per_sample : int
        Number of scalars in one input sample.
    latent_scalars_per_sample : int
        Number of scalars in one latent representation.
    latent_dimensionality_ratio : float
        ``latent_scalars_per_sample / original_scalars_per_sample``. This is a
        dimensionality ratio only. It is not a bitrate, a storage compression
        ratio, or a compressed file size, because it ignores latent precision,
        quantisation, entropy coding, and model parameter storage.
    test_metrics : mapping
        Reconstruction metrics computed on the test partition only.
    fit_seconds : float
        Observed wall-clock fitting time from a monotonic clock.
    reconstruction_seconds : float
        Observed wall-clock reconstruction time from a monotonic clock.
    configuration : mapping
        The specification configuration recorded verbatim.
    """

    name: str
    method_type: str
    latent_dimension: int
    uses_validation_partition: bool
    original_scalars_per_sample: int
    latent_scalars_per_sample: int
    latent_dimensionality_ratio: float
    test_metrics: Mapping[str, float]
    fit_seconds: float
    reconstruction_seconds: float
    configuration: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze the result after construction."""
        object.__setattr__(self, "name", _validate_non_empty_string("name", self.name))
        object.__setattr__(
            self,
            "method_type",
            _validate_non_empty_string("method_type", self.method_type),
        )
        for attribute in (
            "latent_dimension",
            "original_scalars_per_sample",
            "latent_scalars_per_sample",
        ):
            object.__setattr__(
                self,
                attribute,
                _validate_positive_integer(attribute, getattr(self, attribute)),
            )
        object.__setattr__(
            self,
            "uses_validation_partition",
            _validate_boolean(
                "uses_validation_partition",
                self.uses_validation_partition,
            ),
        )
        ratio = _validate_finite_float(
            "latent_dimensionality_ratio",
            self.latent_dimensionality_ratio,
        )
        expected_ratio = (
            self.latent_scalars_per_sample / self.original_scalars_per_sample
        )
        if ratio != expected_ratio:
            raise ValueError(
                "latent_dimensionality_ratio must equal "
                "latent_scalars_per_sample / original_scalars_per_sample "
                f"({expected_ratio!r}); got {ratio!r}."
            )
        object.__setattr__(self, "latent_dimensionality_ratio", ratio)
        for attribute in ("fit_seconds", "reconstruction_seconds"):
            seconds = _validate_finite_float(attribute, getattr(self, attribute))
            if seconds < 0:
                raise ValueError(f"{attribute} must not be negative.")
            object.__setattr__(self, attribute, seconds)

        if not isinstance(self.test_metrics, Mapping) or not self.test_metrics:
            raise TypeError("test_metrics must be a non-empty mapping.")
        metrics: dict[str, float] = {}
        for key, value in self.test_metrics.items():
            metric = _validate_non_empty_string("test_metrics key", key)
            if metric not in _SUPPORTED_METRICS:
                raise ValueError(f"Unsupported metric in test_metrics: {metric!r}.")
            metrics[metric] = _validate_finite_float(f"test_metrics[{metric!r}]", value)
        object.__setattr__(self, "test_metrics", _freeze_json(metrics))
        object.__setattr__(
            self,
            "configuration",
            _freeze_json(
                _validate_configuration(self.configuration, name="configuration")
            ),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible dictionary describing this result."""
        return {
            "name": self.name,
            "method_type": self.method_type,
            "latent_dimension": self.latent_dimension,
            "uses_validation_partition": self.uses_validation_partition,
            "original_scalars_per_sample": self.original_scalars_per_sample,
            "latent_scalars_per_sample": self.latent_scalars_per_sample,
            "latent_dimensionality_ratio": self.latent_dimensionality_ratio,
            "test_metrics": _thaw_json(self.test_metrics),
            "timing": {
                "fit_seconds": self.fit_seconds,
                "reconstruction_seconds": self.reconstruction_seconds,
            },
            "configuration": _thaw_json(self.configuration),
        }

    def identity(self) -> dict[str, JsonValue]:
        """Return the configuration identity, excluding measured outcomes."""
        return {
            "name": self.name,
            "method_type": self.method_type,
            "latent_dimension": self.latent_dimension,
            "uses_validation_partition": self.uses_validation_partition,
            "original_scalars_per_sample": self.original_scalars_per_sample,
            "latent_scalars_per_sample": self.latent_scalars_per_sample,
            "configuration": _thaw_json(self.configuration),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MethodBenchmarkResult:
        """Construct a validated result from a dictionary."""
        if not isinstance(payload, Mapping):
            raise TypeError("Result payload must be a mapping.")
        required = {
            "name",
            "method_type",
            "latent_dimension",
            "uses_validation_partition",
            "original_scalars_per_sample",
            "latent_scalars_per_sample",
            "latent_dimensionality_ratio",
            "test_metrics",
            "timing",
            "configuration",
        }
        _require_exact_fields(payload, required=required, label="Result")
        timing = payload["timing"]
        if type(timing) is not dict:
            raise TypeError("Result timing must be an exact JSON object.")
        _require_exact_fields(
            timing,
            required={"fit_seconds", "reconstruction_seconds"},
            label="Result timing",
        )
        if type(payload["test_metrics"]) is not dict:
            raise TypeError("Result test_metrics must be an exact JSON object.")
        if type(payload["configuration"]) is not dict:
            raise TypeError("Result configuration must be an exact JSON object.")
        return cls(
            name=payload["name"],
            method_type=payload["method_type"],
            latent_dimension=payload["latent_dimension"],
            uses_validation_partition=payload["uses_validation_partition"],
            original_scalars_per_sample=payload["original_scalars_per_sample"],
            latent_scalars_per_sample=payload["latent_scalars_per_sample"],
            latent_dimensionality_ratio=payload["latent_dimensionality_ratio"],
            test_metrics=payload["test_metrics"],
            fit_seconds=timing["fit_seconds"],
            reconstruction_seconds=timing["reconstruction_seconds"],
            configuration=payload["configuration"],
        )


def _require_exact_fields(
    payload: Mapping[str, Any],
    *,
    required: set[str],
    label: str,
) -> None:
    if any(type(key) is not str for key in payload):
        raise TypeError(f"{label} field names must be exact built-in strings.")
    missing = sorted(required.difference(payload))
    extra = sorted(set(payload).difference(required))
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}.")
    if extra:
        raise ValueError(f"{label} contains unsupported fields: {extra}.")


@dataclass(frozen=True)
class ReconstructionBenchmarkReport:
    """Complete record of one reconstruction benchmark run.

    Attributes
    ----------
    schema_version : int
        Report schema version.
    n_samples : int
        Total samples in the benchmarked dataset.
    sample_shape : tuple of int
        Per-sample shape, excluding the leading sample dimension.
    partition_sizes : mapping
        Sample counts for the train, validation, test, and excluded partitions.
    metrics : tuple of str
        Metric names in the order they were requested.
    seed : int or None
        Seed applied inside an isolated random state, if any.
    split_identity : mapping
        Stable identity of the chronological split that produced the
        partitions.
    results : tuple of MethodBenchmarkResult
        One result per benchmarked method, in specification order.

    Notes
    -----
    The report deliberately provides no ranking or "best method" field. It
    measures reconstruction error on held-out samples, which is not the same as
    downstream scientific skill.
    """

    schema_version: int
    n_samples: int
    sample_shape: tuple[int, ...]
    partition_sizes: Mapping[str, int]
    metrics: tuple[str, ...]
    seed: int | None
    split_identity: Mapping[str, JsonValue]
    results: tuple[MethodBenchmarkResult, ...]

    def __post_init__(self) -> None:
        """Validate and freeze the report after construction."""
        if type(self.schema_version) is not int:
            raise TypeError("schema_version must be an exact non-Boolean integer.")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported benchmark report schema version "
                f"{self.schema_version}; expected {_SCHEMA_VERSION}."
            )
        object.__setattr__(
            self,
            "n_samples",
            _validate_positive_integer("n_samples", self.n_samples),
        )
        if isinstance(self.sample_shape, (str, bytes)) or not isinstance(
            self.sample_shape, Sequence
        ):
            raise TypeError("sample_shape must be a sequence of positive integers.")
        object.__setattr__(
            self,
            "sample_shape",
            tuple(
                _validate_positive_integer("sample_shape entry", dimension)
                for dimension in self.sample_shape
            ),
        )
        if not self.sample_shape:
            raise ValueError("sample_shape must contain at least one dimension.")

        if not isinstance(self.partition_sizes, Mapping):
            raise TypeError("partition_sizes must be a mapping.")
        if set(self.partition_sizes) != set(_PARTITION_NAMES):
            raise ValueError(
                f"partition_sizes must define exactly {list(_PARTITION_NAMES)}."
            )
        sizes: dict[str, int] = {}
        for partition in _PARTITION_NAMES:
            value = self.partition_sizes[partition]
            if type(value) is not int or isinstance(value, bool):
                raise TypeError(
                    f"partition_sizes[{partition!r}] must be an exact integer."
                )
            if value < 0:
                raise ValueError(
                    f"partition_sizes[{partition!r}] must not be negative."
                )
            sizes[partition] = value
        object.__setattr__(self, "partition_sizes", _freeze_json(sizes))

        object.__setattr__(self, "metrics", _validate_metrics(self.metrics))

        if self.seed is not None:
            if type(self.seed) is not int or isinstance(self.seed, bool):
                raise TypeError("seed must be an exact non-Boolean integer or None.")
            if self.seed < 0:
                raise ValueError("seed must be non-negative.")

        object.__setattr__(
            self,
            "split_identity",
            _freeze_json(
                _validate_configuration(self.split_identity, name="split_identity")
            ),
        )

        if isinstance(self.results, (str, bytes)) or not isinstance(
            self.results, Sequence
        ):
            raise TypeError("results must be a sequence of MethodBenchmarkResult.")
        results = tuple(self.results)
        if not results:
            raise ValueError("results must contain at least one method result.")
        if any(not isinstance(result, MethodBenchmarkResult) for result in results):
            raise TypeError("Every result must be a MethodBenchmarkResult.")
        names = [result.name for result in results]
        if len(set(names)) != len(names):
            raise ValueError("Benchmark method names must be unique within a report.")
        for result in results:
            if set(result.test_metrics) != set(self.metrics):
                raise ValueError(
                    f"Result {result.name!r} does not report exactly the "
                    f"requested metrics {list(self.metrics)}."
                )
        object.__setattr__(self, "results", results)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible dictionary describing the complete run."""
        return {
            "schema_version": self.schema_version,
            "n_samples": self.n_samples,
            "sample_shape": list(self.sample_shape),
            "partition_sizes": _thaw_json(self.partition_sizes),
            "metrics": list(self.metrics),
            "seed": self.seed,
            "split_identity": _thaw_json(self.split_identity),
            "results": [result.to_dict() for result in self.results],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the complete run deterministically as strict JSON."""
        return _canonical_json(self.to_dict(), indent=indent) + "\n"

    def identity(self) -> dict[str, JsonValue]:
        """Return the deterministic identity of the benchmark configuration.

        The identity answers "what was compared, on which samples". It
        deliberately excludes measured outcomes: metric values and wall-clock
        timings are observational and are not reproducible bit for bit across
        machines, library versions, or devices.
        """
        return {
            "schema_version": self.schema_version,
            "n_samples": self.n_samples,
            "sample_shape": list(self.sample_shape),
            "partition_sizes": _thaw_json(self.partition_sizes),
            "metrics": list(self.metrics),
            "seed": self.seed,
            "split_identity": _thaw_json(self.split_identity),
            "methods": [result.identity() for result in self.results],
        }

    def identity_digest(self) -> str:
        """Return a SHA-256 digest of the deterministic benchmark identity."""
        payload = _canonical_json(self.identity()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReconstructionBenchmarkReport:
        """Construct a validated report from a dictionary."""
        if not isinstance(payload, Mapping):
            raise TypeError("Report payload must be a mapping.")
        required = {
            "schema_version",
            "n_samples",
            "sample_shape",
            "partition_sizes",
            "metrics",
            "seed",
            "split_identity",
            "results",
        }
        _require_exact_fields(payload, required=required, label="Report")
        for name in ("sample_shape", "metrics", "results"):
            if type(payload[name]) is not list:
                raise TypeError(f"Report {name} must be an exact JSON list.")
        for name in ("partition_sizes", "split_identity"):
            if type(payload[name]) is not dict:
                raise TypeError(f"Report {name} must be an exact JSON object.")
        return cls(
            schema_version=payload["schema_version"],
            n_samples=payload["n_samples"],
            sample_shape=tuple(payload["sample_shape"]),
            partition_sizes=payload["partition_sizes"],
            metrics=tuple(payload["metrics"]),
            seed=payload["seed"],
            split_identity=payload["split_identity"],
            results=tuple(
                MethodBenchmarkResult.from_dict(result) for result in payload["results"]
            ),
        )


def _optional_torch() -> Any | None:
    try:
        import torch
    except ImportError:  # pragma: no cover - depends on installation
        return None
    return torch


@contextmanager
def _isolated_random_state(seed: int | None) -> Iterator[None]:
    """Run a block with an isolated, optionally seeded random state.

    The caller's global NumPy random state and PyTorch generator states are
    restored on exit, so benchmarking never perturbs surrounding code.

    Seeding makes a run repeatable on the same machine, device, and library
    versions. It does not guarantee bitwise-identical PyTorch results across
    devices, because algorithm selection and reduction order may differ.
    """
    numpy_state = np.random.get_state()
    with ExitStack() as stack:
        torch = _optional_torch()
        if torch is not None:
            devices: list[int] = []
            if torch.cuda.is_available():  # pragma: no cover - needs CUDA
                current = torch.cuda.current_device()
                devices = [current]
            stack.enter_context(torch.random.fork_rng(devices=devices))
        try:
            if seed is not None:
                np.random.seed(seed)
                if torch is not None:
                    torch.manual_seed(seed)
            yield
        finally:
            np.random.set_state(numpy_state)


def _split_identity(
    split: ChronologicalSplit,
    *,
    time_axis_verified: bool,
) -> dict[str, JsonValue]:
    """Return a stable identity for the partitions produced by ``split``."""
    manifest = split.manifest
    partitions = {
        "train": list(manifest.train_indices),
        "validation": list(manifest.validation_indices),
        "test": list(manifest.test_indices),
        "excluded": list(manifest.excluded_indices),
    }
    digest = hashlib.sha256(_canonical_json(partitions).encode("utf-8")).hexdigest()
    return {
        "manifest_schema_version": manifest.schema_version,
        "method": manifest.method,
        "n_samples": manifest.n_samples,
        "dataset_fingerprint": manifest.dataset_fingerprint,
        "time_kind": manifest.time_kind,
        "axis_mode": manifest.axis_mode,
        "partition_digest": digest,
        "time_axis_verified": time_axis_verified,
    }


def _validate_methods(methods: Any) -> tuple[BenchmarkMethod, ...]:
    if isinstance(methods, (str, bytes)) or not isinstance(methods, Sequence):
        raise TypeError("methods must be a sequence of BenchmarkMethod objects.")
    specifications = tuple(methods)
    if not specifications:
        raise ValueError("methods must contain at least one BenchmarkMethod.")
    if any(
        not isinstance(specification, BenchmarkMethod)
        for specification in specifications
    ):
        raise TypeError("Every entry in methods must be a BenchmarkMethod.")
    names = [specification.name for specification in specifications]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"Benchmark method names must be unique; duplicates: {duplicates}."
        )
    return specifications


def _build_method(specification: BenchmarkMethod) -> ReconstructionMethod:
    """Instantiate one method and verify it satisfies the benchmark contract."""
    instance = specification.factory()
    if instance is None:
        raise TypeError(f"The factory for method {specification.name!r} returned None.")
    for attribute in ("fit", "reconstruct"):
        if not callable(getattr(instance, attribute, None)):
            raise TypeError(
                f"Method {specification.name!r} must expose a callable "
                f"{attribute}() method."
            )
    latent = getattr(instance, "latent_dimension", None)
    if latent is None:
        raise TypeError(
            f"Method {specification.name!r} must expose a latent_dimension."
        )
    if _validate_positive_integer("latent_dimension", latent) != (
        specification.latent_dimension
    ):
        raise ValueError(
            f"Method {specification.name!r} reports latent dimension "
            f"{int(latent)}, but the specification declares "
            f"{specification.latent_dimension}."
        )
    if getattr(instance, "is_fitted", False):
        raise ValueError(
            f"The factory for method {specification.name!r} returned an already "
            "fitted instance. Factories must return a fresh, unfitted model so "
            "that state cannot leak between benchmark runs."
        )
    return instance


def _validate_reconstruction(
    reconstruction: Any,
    reference: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Reject reconstructions NumPy would otherwise broadcast into shape."""
    if not isinstance(reconstruction, np.ndarray):
        raise TypeError(
            f"Method {name!r} must return a NumPy array from reconstruct()."
        )
    if tuple(reconstruction.shape) != tuple(reference.shape):
        raise ValueError(
            f"Method {name!r} returned reconstruction shape "
            f"{tuple(reconstruction.shape)}, but the test partition has shape "
            f"{tuple(reference.shape)}. Broadcasting is never applied."
        )
    if not np.issubdtype(reconstruction.dtype, np.number):
        raise TypeError(f"Method {name!r} must return numeric reconstructions.")
    if np.issubdtype(reconstruction.dtype, np.complexfloating):
        raise TypeError(f"Method {name!r} must return real-valued reconstructions.")
    if not np.isfinite(reconstruction).all():
        raise ValueError(f"Method {name!r} returned non-finite reconstruction values.")
    return reconstruction


def _test_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: tuple[str, ...],
) -> dict[str, float]:
    reconstruction_error = _load_reconstruction_error()
    values: dict[str, float] = {}
    for metric in metrics:
        values[metric] = float(
            reconstruction_error(
                y_true,
                y_pred,
                metric=metric,
                reduction=_METRIC_REDUCTION,
            )
        )
    return values


def run_reconstruction_benchmark(
    X: np.ndarray,
    *,
    split: ChronologicalSplit,
    methods: Sequence[BenchmarkMethod],
    metrics: Sequence[str] = _DEFAULT_METRICS,
    seed: int | None = None,
    sample_times: Any | None = None,
    sample_start_times: Any | None = None,
    sample_end_times: Any | None = None,
) -> ReconstructionBenchmarkReport:
    """Compare reconstruction methods on identical chronological partitions.

    Every method is fitted on ``X[split.train_indices]``. Methods that declare
    ``uses_validation_partition`` additionally receive
    ``X[split.validation_indices]`` for early stopping. Metrics are computed
    only on ``X[split.test_indices]``, in the original sample space, using the
    shared implementation in :mod:`bluemath_tk.deeplearning.metrics`.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape ``(n_samples, ...)``. It is never modified, and every
        partition handed to a method is an independent copy.
    split : ChronologicalSplit
        Manifest-backed chronological partitions. The benchmark never creates a
        split of its own.
    methods : sequence of BenchmarkMethod
        Method specifications with unique names.
    metrics : sequence of str, optional
        Metric names from ``("mse", "mae", "rmse")``. Default is
        ``("mse", "mae", "rmse")``.
    seed : int, optional
        Seed applied inside an isolated random state before each method is
        built and fitted. The caller's random state is restored afterwards.
    sample_times, sample_start_times, sample_end_times : array-like, optional
        The time coordinates that order ``X``. When supplied, the split
        manifest is validated against them, which proves the manifest is being
        replayed against the dataset it was created from rather than a
        reordered or different dataset.

    Returns
    -------
    ReconstructionBenchmarkReport
        The complete record of the run.

    Raises
    ------
    ValueError
        If the data, split, methods, metrics, or any reconstruction violates
        the benchmark contract.

    Notes
    -----
    Domain-specific normalisation is the caller's responsibility in this first
    framework release. Any preprocessing must be applied identically to every
    compared method and fitted on the training partition alone.

    The reported ``latent_dimensionality_ratio`` compares latent scalars with
    input scalars. It is not a bitrate and not a storage compression ratio.
    """
    _validate_sample_data(X, name="X")
    if not isinstance(split, ChronologicalSplit):
        raise TypeError(
            "split must be a bluemath_tk.validation.ChronologicalSplit, so that "
            "partition membership is always backed by a validated manifest."
        )
    manifest = split.manifest
    if manifest.n_samples != int(X.shape[0]):
        raise ValueError(
            f"The split describes {manifest.n_samples} samples, but X contains "
            f"{int(X.shape[0])}."
        )

    time_axis_verified = any(
        coordinates is not None
        for coordinates in (sample_times, sample_start_times, sample_end_times)
    )
    if time_axis_verified:
        manifest.validate_against(
            sample_times=sample_times,
            sample_start_times=sample_start_times,
            sample_end_times=sample_end_times,
        )

    requested_metrics = _validate_metrics(metrics)
    specifications = _validate_methods(methods)
    if seed is not None:
        if type(seed) is not int or isinstance(seed, bool):
            raise TypeError("seed must be an exact non-Boolean integer or None.")
        if seed < 0:
            raise ValueError("seed must be non-negative.")

    train_indices = np.asarray(split.train_indices)
    validation_indices = np.asarray(split.validation_indices)
    test_indices = np.asarray(split.test_indices)
    for name, indices in (
        ("train_indices", train_indices),
        ("validation_indices", validation_indices),
        ("test_indices", test_indices),
    ):
        if indices.size == 0:
            raise ValueError(f"split.{name} must not be empty.")
        if int(indices.max()) >= int(X.shape[0]) or int(indices.min()) < 0:
            raise ValueError(f"split.{name} contains an index outside X.")

    # Fancy indexing copies, so no method can reach or mutate the caller's X.
    X_train = X[train_indices]
    X_validation = X[validation_indices]
    X_test = X[test_indices]

    sample_shape = tuple(int(dimension) for dimension in X.shape[1:])
    original_scalars = int(np.prod(sample_shape))

    built: list[ReconstructionMethod] = []
    results: list[MethodBenchmarkResult] = []
    for specification in specifications:
        with _isolated_random_state(seed):
            instance = _build_method(specification)
            if any(instance is other for other in built):
                raise ValueError(
                    f"The factory for method {specification.name!r} returned an "
                    "instance already used by another method. Every method must "
                    "get its own model."
                )
            built.append(instance)

            fit_start = perf_counter()
            instance.fit(
                X_train,
                X_validation if specification.uses_validation_partition else None,
            )
            fit_seconds = perf_counter() - fit_start

            reconstruction_start = perf_counter()
            reconstruction = instance.reconstruct(X_test)
            reconstruction_seconds = perf_counter() - reconstruction_start

        reconstruction = _validate_reconstruction(
            reconstruction,
            X_test,
            name=specification.name,
        )
        results.append(
            MethodBenchmarkResult(
                name=specification.name,
                method_type=specification.method_type,
                latent_dimension=specification.latent_dimension,
                uses_validation_partition=specification.uses_validation_partition,
                original_scalars_per_sample=original_scalars,
                latent_scalars_per_sample=specification.latent_dimension,
                latent_dimensionality_ratio=(
                    specification.latent_dimension / original_scalars
                ),
                test_metrics=_test_metrics(
                    X_test,
                    reconstruction,
                    requested_metrics,
                ),
                fit_seconds=max(fit_seconds, 0.0),
                reconstruction_seconds=max(reconstruction_seconds, 0.0),
                configuration=_thaw_json(specification.configuration),
            )
        )

    counts = split.counts
    return ReconstructionBenchmarkReport(
        schema_version=_SCHEMA_VERSION,
        n_samples=int(X.shape[0]),
        sample_shape=sample_shape,
        partition_sizes={name: int(counts[name]) for name in _PARTITION_NAMES},
        metrics=requested_metrics,
        seed=seed,
        split_identity=_split_identity(split, time_axis_verified=time_axis_verified),
        results=tuple(results),
    )
