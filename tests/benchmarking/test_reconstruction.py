"""Regression tests for the common reconstruction benchmark framework."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bluemath_tk.benchmarking import (  # noqa: E402
    AutoencoderReconstruction,
    BenchmarkMethod,
    MethodBenchmarkResult,
    PCAReconstruction,
    ReconstructionBenchmarkReport,
    autoencoder_benchmark_method,
    pca_benchmark_method,
    run_reconstruction_benchmark,
)
from bluemath_tk.deeplearning.autoencoders import StandardAutoencoder  # noqa: E402
from bluemath_tk.deeplearning.metrics import reconstruction_error  # noqa: E402
from bluemath_tk.deeplearning.variational_autoencoders import (  # noqa: E402
    VariationalAutoencoder,
)
from bluemath_tk.validation import split_chronologically  # noqa: E402

METRICS = ("mse", "mae", "rmse")


def _low_rank_dataset(
    n_samples: int = 60,
    sample_shape: tuple[int, ...] = (3, 4),
    rank: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """Build deterministic synthetic data lying exactly on a low-rank subspace."""
    rng = np.random.default_rng(seed)
    n_features = int(np.prod(sample_shape))
    latent = rng.normal(size=(n_samples, rank))
    mixing = rng.normal(size=(rank, n_features))
    return (latent @ mixing).reshape(n_samples, *sample_shape)


def _split_for(n_samples: int, fractions=(0.6, 0.2, 0.2)):
    return split_chronologically(
        sample_times=np.arange(n_samples),
        fractions=fractions,
    )


def _digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


class _RecordingMethod:
    """Controlled benchmark method that records every array it is handed."""

    def __init__(self, latent_dimension=2, transform=None, draw_random=False):
        self.latent_dimension = latent_dimension
        self.is_fitted = False
        self._transform = transform
        self._draw_random = draw_random
        self.fit_train = None
        self.fit_train_object = None
        self.fit_validation = None
        self.reconstruct_inputs = []
        self.reconstruct_input_objects = []
        self.random_draws = []

    def fit(self, X_train, X_validation):
        self.fit_train_object = X_train
        self.fit_train = np.array(X_train, copy=True)
        self.fit_validation = (
            None if X_validation is None else np.array(X_validation, copy=True)
        )
        if self._draw_random:
            self.random_draws.append(float(np.random.rand()))
            self.random_draws.append(float(torch.rand(1).item()))
        self.is_fitted = True

    def reconstruct(self, X):
        self.reconstruct_input_objects.append(X)
        self.reconstruct_inputs.append(np.array(X, copy=True))
        if self._transform is None:
            return np.array(X, copy=True)
        return self._transform(X)


def _recording_spec(
    name: str = "recorder",
    *,
    latent_dimension: int = 2,
    transform=None,
    uses_validation_partition: bool = True,
    draw_random: bool = False,
    declared_latent_dimension: int | None = None,
):
    """Return a specification plus the list receiving every built instance."""
    created: list[_RecordingMethod] = []

    def factory():
        instance = _RecordingMethod(
            latent_dimension=latent_dimension,
            transform=transform,
            draw_random=draw_random,
        )
        created.append(instance)
        return instance

    specification = BenchmarkMethod(
        name=name,
        method_type="controlled-fake",
        latent_dimension=(
            latent_dimension
            if declared_latent_dimension is None
            else declared_latent_dimension
        ),
        factory=factory,
        configuration={"kind": "controlled-fake"},
        uses_validation_partition=uses_validation_partition,
    )
    return specification, created


def _pca_spec(name: str = "pca", *, n_components: int = 3, scale_data: bool = False):
    """Return a PCA specification plus the list receiving every built adapter."""
    created: list[PCAReconstruction] = []

    def factory():
        instance = PCAReconstruction(
            n_components=n_components,
            scale_data=scale_data,
        )
        created.append(instance)
        return instance

    specification = BenchmarkMethod(
        name=name,
        method_type="pca",
        latent_dimension=n_components,
        factory=factory,
        configuration={"n_components": n_components, "scale_data": scale_data},
        uses_validation_partition=False,
    )
    return specification, created


# ---------------------------------------------------------------------------
# A. PCA correctness baseline
# ---------------------------------------------------------------------------


def test_pca_reconstructs_low_rank_data_with_sufficient_components():
    X = _low_rank_dataset(rank=3)
    split = _split_for(len(X))

    report = run_reconstruction_benchmark(
        X,
        split=split,
        methods=[pca_benchmark_method("pca-k3", n_components=3)],
    )

    result = report.results[0]
    assert result.test_metrics["mse"] < 1e-20
    assert result.test_metrics["mae"] < 1e-10
    assert result.test_metrics["rmse"] < 1e-10


def test_pca_adapter_matches_the_existing_pca_api_semantics():
    X = _low_rank_dataset(rank=3)
    split = _split_for(len(X))
    X_train = X[split.train_indices]
    X_test = X[split.test_indices]

    adapter = PCAReconstruction(n_components=3)
    adapter.fit(X_train)
    reconstruction = adapter.reconstruct(X_test)

    # The adapter must not invent a second PCA: it delegates to the fitted
    # scikit-learn estimator owned by bluemath_tk.datamining.pca.PCA.
    estimator = adapter.pca.pca
    expected = estimator.inverse_transform(
        estimator.transform(X_test.reshape(len(X_test), -1))
    ).reshape(X_test.shape)
    assert np.array_equal(reconstruction, expected)
    assert adapter.pca.is_fitted is True
    assert adapter.latent_dimension == 3


def test_fewer_components_than_rank_degrades_but_stays_finite():
    X = _low_rank_dataset(rank=3)
    split = _split_for(len(X))

    report = run_reconstruction_benchmark(
        X,
        split=split,
        methods=[
            pca_benchmark_method("pca-k1", n_components=1),
            pca_benchmark_method("pca-k3", n_components=3),
        ],
    )

    poor, good = report.results
    assert poor.test_metrics["mse"] > good.test_metrics["mse"]
    assert np.isfinite(poor.test_metrics["mse"])


# ---------------------------------------------------------------------------
# B. Shape round trip and sample ordering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sample_shape", [(7,), (3, 4), (2, 3, 4), (2, 1, 3, 2)])
def test_pca_preserves_sample_shape_for_n_dimensional_samples(sample_shape):
    X = _low_rank_dataset(n_samples=40, sample_shape=sample_shape, rank=2)
    split = _split_for(len(X))
    X_train = X[split.train_indices]
    X_test = X[split.test_indices]

    adapter = PCAReconstruction(n_components=2)
    adapter.fit(X_train)
    reconstruction = adapter.reconstruct(X_test)

    assert reconstruction.shape == X_test.shape
    assert np.allclose(reconstruction, X_test, atol=1e-8)


def test_pca_preserves_sample_order():
    X = _low_rank_dataset(n_samples=40, sample_shape=(3, 4), rank=3)
    split = _split_for(len(X))
    X_train = X[split.train_indices]
    X_test = X[split.test_indices]

    adapter = PCAReconstruction(n_components=3)
    adapter.fit(X_train)
    reconstruction = adapter.reconstruct(X_test)

    aligned = np.max(np.abs(reconstruction - X_test), axis=tuple(range(1, X.ndim)))
    rolled = np.max(
        np.abs(reconstruction - np.roll(X_test, 1, axis=0)),
        axis=tuple(range(1, X.ndim)),
    )
    assert np.all(aligned < 1e-8)
    assert np.all(rolled > 1e-6)


def test_results_are_independent_of_the_input_memory_layout():
    X = _low_rank_dataset(n_samples=40, sample_shape=(3, 4), rank=3)
    fortran = np.asfortranarray(X)
    assert fortran.flags.f_contiguous
    assert np.array_equal(fortran, X)
    split = _split_for(len(X))
    method = pca_benchmark_method("pca-k3", n_components=3)

    c_order = run_reconstruction_benchmark(X, split=split, methods=[method])
    f_order = run_reconstruction_benchmark(fortran, split=split, methods=[method])

    assert dict(c_order.results[0].test_metrics) == dict(
        f_order.results[0].test_metrics
    )
    assert c_order.identity_digest() == f_order.identity_digest()


def test_pca_reconstruction_is_identical_for_fortran_ordered_samples():
    X = _low_rank_dataset(n_samples=30, sample_shape=(3, 5), rank=4)
    train, test = X[:20], X[20:]

    c_adapter = PCAReconstruction(n_components=4)
    c_adapter.fit(train)
    f_adapter = PCAReconstruction(n_components=4)
    f_adapter.fit(np.asfortranarray(train))

    assert np.array_equal(
        c_adapter.reconstruct(test),
        f_adapter.reconstruct(np.asfortranarray(test)),
    )


def test_pca_reconstruction_uses_c_order_flattening():
    X = _low_rank_dataset(n_samples=30, sample_shape=(3, 5), rank=4)
    adapter = PCAReconstruction(n_components=4)
    adapter.fit(X[:20])

    stacked = adapter.pca.stacked_data_matrix
    assert np.array_equal(stacked, X[:20].reshape(20, -1))


# ---------------------------------------------------------------------------
# C. Shared metric agreement
# ---------------------------------------------------------------------------


def test_benchmark_metrics_agree_with_shared_metric_implementation():
    X = _low_rank_dataset()
    split = _split_for(len(X))
    specification, _ = _recording_spec(transform=lambda values: values + 0.25)

    report = run_reconstruction_benchmark(
        X,
        split=split,
        methods=[specification],
        metrics=METRICS,
    )

    X_test = X[split.test_indices]
    result = report.results[0]
    for metric in METRICS:
        expected = float(
            reconstruction_error(
                X_test,
                X_test + 0.25,
                metric=metric,
                reduction="mean",
            )
        )
        assert result.test_metrics[metric] == expected


def test_reported_metrics_preserve_the_requested_order():
    X = _low_rank_dataset()
    split = _split_for(len(X))
    specification, _ = _recording_spec()

    report = run_reconstruction_benchmark(
        X,
        split=split,
        methods=[specification],
        metrics=("rmse", "mse"),
    )
    assert report.metrics == ("rmse", "mse")
    assert set(report.results[0].test_metrics) == {"rmse", "mse"}


def test_metrics_are_computed_only_on_the_test_partition():
    X = _low_rank_dataset()
    split = _split_for(len(X))
    specification, created = _recording_spec()

    run_reconstruction_benchmark(X, split=split, methods=[specification])

    reconstruct_inputs = created[0].reconstruct_inputs
    assert len(reconstruct_inputs) == 1
    assert np.array_equal(reconstruct_inputs[0], X[split.test_indices])


# ---------------------------------------------------------------------------
# D. Chronological split membership
# ---------------------------------------------------------------------------


def test_runner_uses_exactly_the_supplied_chronological_partitions():
    X = _low_rank_dataset(n_samples=53)
    split = _split_for(len(X), fractions=(0.5, 0.25, 0.25))
    specification, created = _recording_spec()

    report = run_reconstruction_benchmark(X, split=split, methods=[specification])

    instance = created[0]
    assert np.array_equal(instance.fit_train, X[split.train_indices])
    assert np.array_equal(instance.fit_validation, X[split.validation_indices])
    assert np.array_equal(instance.reconstruct_inputs[0], X[split.test_indices])
    assert report.partition_sizes["train"] == int(split.train_indices.size)
    assert report.partition_sizes["validation"] == int(split.validation_indices.size)
    assert report.partition_sizes["test"] == int(split.test_indices.size)


def test_partitions_have_no_off_by_one_boundaries():
    X = _low_rank_dataset(n_samples=41)
    split = _split_for(len(X), fractions=(0.6, 0.2, 0.2))
    specification, created = _recording_spec()

    run_reconstruction_benchmark(X, split=split, methods=[specification])

    instance = created[0]
    assert len(instance.fit_train) == int(split.train_indices.size)
    assert len(instance.fit_validation) == int(split.validation_indices.size)
    # The first validation sample must not appear in training and the last
    # training sample must not appear in validation.
    assert not np.array_equal(instance.fit_train[-1], X[split.validation_indices[0]])
    assert np.array_equal(instance.fit_train[-1], X[split.train_indices[-1]])
    assert np.array_equal(instance.fit_validation[0], X[split.validation_indices[0]])


def test_runner_rejects_a_split_describing_a_different_sample_count():
    X = _low_rank_dataset(n_samples=40)
    split = _split_for(30)
    specification, _ = _recording_spec()

    with pytest.raises(ValueError, match="describes 30 samples"):
        run_reconstruction_benchmark(X, split=split, methods=[specification])


def test_runner_requires_a_manifest_backed_chronological_split():
    X = _low_rank_dataset()
    specification, _ = _recording_spec()

    with pytest.raises(TypeError, match="ChronologicalSplit"):
        run_reconstruction_benchmark(
            X,
            split={"train": [0], "validation": [1], "test": [2]},
            methods=[specification],
        )


def test_optional_time_axis_verification_detects_a_replayed_wrong_dataset():
    X = _low_rank_dataset(n_samples=40)
    times = np.arange(40)
    split = split_chronologically(sample_times=times, fractions=(0.6, 0.2, 0.2))
    specification, _ = _recording_spec()

    report = run_reconstruction_benchmark(
        X,
        split=split,
        methods=[specification],
        sample_times=times,
    )
    assert report.split_identity["time_axis_verified"] is True

    specification, _ = _recording_spec()
    with pytest.raises(ValueError, match="fingerprint"):
        run_reconstruction_benchmark(
            X,
            split=split,
            methods=[specification],
            sample_times=times + 1,
        )


def test_time_axis_verification_is_recorded_as_false_when_not_requested():
    X = _low_rank_dataset(n_samples=40)
    split = _split_for(40)
    specification, _ = _recording_spec()

    report = run_reconstruction_benchmark(X, split=split, methods=[specification])
    assert report.split_identity["time_axis_verified"] is False


# ---------------------------------------------------------------------------
# E. Leakage adversary
# ---------------------------------------------------------------------------


def test_altering_test_values_does_not_change_fitted_pca_state():
    X = _low_rank_dataset()
    split = _split_for(len(X))

    first_spec, first_created = _pca_spec()
    run_reconstruction_benchmark(X, split=split, methods=[first_spec])

    attacked = X.copy()
    attacked[split.test_indices] = attacked[split.test_indices] * 1e6 + 12345.0
    second_spec, second_created = _pca_spec()
    run_reconstruction_benchmark(attacked, split=split, methods=[second_spec])

    original = first_created[0].pca.pca
    adversarial = second_created[0].pca.pca
    assert np.array_equal(original.components_, adversarial.components_)
    assert np.array_equal(original.mean_, adversarial.mean_)
    assert np.array_equal(original.explained_variance_, adversarial.explained_variance_)
    assert np.array_equal(
        first_created[0].pca.stacked_data_matrix,
        second_created[0].pca.stacked_data_matrix,
    )


def test_altering_test_values_does_not_change_the_data_reaching_fitting():
    X = _low_rank_dataset()
    split = _split_for(len(X))

    first_spec, first_created = _recording_spec()
    run_reconstruction_benchmark(X, split=split, methods=[first_spec])

    attacked = X.copy()
    attacked[split.test_indices] = 9.9e5
    second_spec, second_created = _recording_spec()
    run_reconstruction_benchmark(attacked, split=split, methods=[second_spec])

    assert _digest(first_created[0].fit_train) == _digest(second_created[0].fit_train)
    assert _digest(first_created[0].fit_validation) == _digest(
        second_created[0].fit_validation
    )


def test_pca_scaler_statistics_come_from_the_training_partition_only():
    X = _low_rank_dataset()
    split = _split_for(len(X))
    specification, created = _pca_spec(n_components=3, scale_data=True)

    run_reconstruction_benchmark(X, split=split, methods=[specification])

    scaler = created[0].pca.scaler
    train_flat = X[split.train_indices].reshape(int(split.train_indices.size), -1)
    assert np.allclose(scaler.mean_, train_flat.mean(axis=0))
    assert not np.allclose(scaler.mean_, X.reshape(len(X), -1).mean(axis=0))


def test_pca_is_not_given_the_validation_partition():
    X = _low_rank_dataset()
    split = _split_for(len(X))
    specification, created = _recording_spec(uses_validation_partition=False)

    run_reconstruction_benchmark(X, split=split, methods=[specification])

    assert created[0].fit_validation is None
    assert pca_benchmark_method("p", n_components=2).uses_validation_partition is False


# ---------------------------------------------------------------------------
# F. Validation-set fidelity
# ---------------------------------------------------------------------------


def test_autoencoder_adapter_forwards_the_exact_validation_partition():
    X = _low_rank_dataset(n_samples=40, sample_shape=(6,), rank=3)
    split = _split_for(len(X))
    captured = {}

    class _CapturingModel:
        k = 2
        is_fitted = False

        def fit(self, X_train, validation_data=None, **kwargs):
            captured["train"] = np.array(X_train, copy=True)
            captured["validation"] = np.array(validation_data[0], copy=True)
            captured["validation_target"] = validation_data[1]
            captured["kwargs"] = dict(kwargs)
            self.is_fitted = True
            return {"train_loss": [0.0], "val_loss": [0.0]}

        def predict(self, X, **kwargs):
            return np.array(X, copy=True)

    specification = autoencoder_benchmark_method(
        "capturing",
        model_factory=_CapturingModel,
        latent_dimension=2,
        configuration={"architecture": "capturing"},
    )
    run_reconstruction_benchmark(X, split=split, methods=[specification])

    assert np.array_equal(captured["train"], X[split.train_indices])
    assert np.array_equal(captured["validation"], X[split.validation_indices])
    assert captured["validation_target"] is None
    assert captured["kwargs"]["verbose"] == 0


def test_test_samples_never_reach_autoencoder_fitting():
    X = _low_rank_dataset(n_samples=40, sample_shape=(6,), rank=3)
    split = _split_for(len(X))
    seen = []

    class _WatchingModel:
        k = 2
        is_fitted = False

        def fit(self, X_train, validation_data=None, **kwargs):
            seen.append(np.array(X_train, copy=True))
            seen.append(np.array(validation_data[0], copy=True))
            self.is_fitted = True
            return {}

        def predict(self, X, **kwargs):
            return np.array(X, copy=True)

    specification = autoencoder_benchmark_method(
        "watching",
        model_factory=_WatchingModel,
        latent_dimension=2,
    )
    run_reconstruction_benchmark(X, split=split, methods=[specification])

    test_rows = {row.tobytes() for row in X[split.test_indices]}
    for array in seen:
        for row in array:
            assert row.tobytes() not in test_rows


def test_autoencoder_adapter_requires_a_validation_partition():
    X = _low_rank_dataset(n_samples=40, sample_shape=(6,), rank=3)
    split = _split_for(len(X))

    class _Model:
        k = 2
        is_fitted = False

        def fit(self, X_train, validation_data=None, **kwargs):
            return {}

        def predict(self, X, **kwargs):
            return np.array(X, copy=True)

    base = autoencoder_benchmark_method(
        "no-validation",
        model_factory=_Model,
        latent_dimension=2,
    )
    specification = BenchmarkMethod(
        name=base.name,
        method_type=base.method_type,
        latent_dimension=base.latent_dimension,
        factory=base.factory,
        uses_validation_partition=False,
    )

    with pytest.raises(ValueError, match="requires the validation partition"):
        run_reconstruction_benchmark(X, split=split, methods=[specification])


# ---------------------------------------------------------------------------
# G. No mutation of caller state
# ---------------------------------------------------------------------------


def test_runner_does_not_mutate_data_split_or_configuration():
    X = _low_rank_dataset()
    split = _split_for(len(X))
    X_before = X.copy()
    train_before = split.train_indices.copy()
    validation_before = split.validation_indices.copy()
    test_before = split.test_indices.copy()
    configuration = {"nested": {"values": [1, 2, 3]}}
    configuration_before = copy.deepcopy(configuration)

    specification = BenchmarkMethod(
        name="pca",
        method_type="pca",
        latent_dimension=3,
        factory=lambda: PCAReconstruction(n_components=3),
        configuration=configuration,
        uses_validation_partition=False,
    )
    run_reconstruction_benchmark(X, split=split, methods=[specification])

    assert np.array_equal(X, X_before)
    assert np.array_equal(split.train_indices, train_before)
    assert np.array_equal(split.validation_indices, validation_before)
    assert np.array_equal(split.test_indices, test_before)
    assert configuration == configuration_before
    assert split.train_indices.flags.writeable is False
    assert split.test_indices.flags.writeable is False


def test_methods_receive_copies_that_do_not_share_memory_with_the_input():
    X = _low_rank_dataset()
    split = _split_for(len(X))
    specification, created = _recording_spec()

    run_reconstruction_benchmark(X, split=split, methods=[specification])

    instance = created[0]
    assert not np.shares_memory(instance.fit_train_object, X)
    assert not np.shares_memory(instance.reconstruct_input_objects[0], X)


def test_a_method_that_mutates_its_partition_cannot_corrupt_the_input():
    X = _low_rank_dataset()
    X_before = X.copy()
    split = _split_for(len(X))

    def _destructive(values):
        values[...] = 0.0
        return np.array(values, copy=True)

    specification, _ = _recording_spec(transform=_destructive)
    run_reconstruction_benchmark(X, split=split, methods=[specification])

    assert np.array_equal(X, X_before)


def test_each_run_builds_a_fresh_method_instance():
    X = _low_rank_dataset()
    split = _split_for(len(X))
    specification, created = _recording_spec()

    run_reconstruction_benchmark(X, split=split, methods=[specification])
    run_reconstruction_benchmark(X, split=split, methods=[specification])

    assert len(created) == 2
    assert created[0] is not created[1]


def test_runner_rejects_an_already_fitted_instance():
    X = _low_rank_dataset()
    split = _split_for(len(X))
    shared = PCAReconstruction(n_components=3)
    shared.fit(X[split.train_indices])

    specification = BenchmarkMethod(
        name="reused",
        method_type="pca",
        latent_dimension=3,
        factory=lambda: shared,
        uses_validation_partition=False,
    )
    with pytest.raises(ValueError, match="already fitted instance"):
        run_reconstruction_benchmark(X, split=split, methods=[specification])


def test_runner_rejects_one_instance_shared_by_two_methods():
    X = _low_rank_dataset()
    split = _split_for(len(X))

    class _StatelessMethod:
        """A method that never reports being fitted, defeating the fit guard."""

        latent_dimension = 2
        is_fitted = False

        def fit(self, X_train, X_validation):
            return None

        def reconstruct(self, X):
            return np.array(X, copy=True)

    shared = _StatelessMethod()
    methods = [
        BenchmarkMethod(
            name=name,
            method_type="controlled-fake",
            latent_dimension=2,
            factory=lambda: shared,
            uses_validation_partition=False,
        )
        for name in ("first", "second")
    ]
    with pytest.raises(ValueError, match="already used by another method"):
        run_reconstruction_benchmark(X, split=split, methods=methods)


# ---------------------------------------------------------------------------
# H. Invalid latent dimensions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_components", [0, -1, -10])
def test_non_positive_latent_dimension_is_rejected(n_components):
    with pytest.raises(ValueError, match="positive integer"):
        pca_benchmark_method("bad", n_components=n_components)


@pytest.mark.parametrize("n_components", [1.5, True, "3", None])
def test_non_integer_latent_dimension_is_rejected(n_components):
    with pytest.raises(TypeError, match="exact non-Boolean integer"):
        pca_benchmark_method("bad", n_components=n_components)


def test_pca_rejects_more_components_than_available():
    X = _low_rank_dataset(n_samples=40, sample_shape=(5,), rank=3)
    split = _split_for(len(X))

    with pytest.raises(ValueError, match="components available"):
        run_reconstruction_benchmark(
            X,
            split=split,
            methods=[pca_benchmark_method("too-many", n_components=6)],
        )


def test_pca_rejects_more_components_than_training_samples():
    X = _low_rank_dataset(n_samples=12, sample_shape=(20,), rank=3)
    split = _split_for(len(X), fractions=(0.5, 0.25, 0.25))

    with pytest.raises(ValueError, match="components available"):
        run_reconstruction_benchmark(
            X,
            split=split,
            methods=[pca_benchmark_method("too-many", n_components=10)],
        )


def test_autoencoder_latent_width_must_match_the_specification():
    class _Model:
        k = 3
        is_fitted = False

        def fit(self, X, **kwargs):
            return {}

        def predict(self, X, **kwargs):
            return X

    with pytest.raises(ValueError, match="contradicts the model latent width"):
        AutoencoderReconstruction(_Model(), latent_dimension=5)


def test_method_reporting_a_different_latent_dimension_is_rejected():
    X = _low_rank_dataset()
    split = _split_for(len(X))
    specification, _ = _recording_spec(
        latent_dimension=2,
        declared_latent_dimension=4,
    )

    with pytest.raises(ValueError, match="reports latent dimension"):
        run_reconstruction_benchmark(X, split=split, methods=[specification])


# ---------------------------------------------------------------------------
# I. Reconstruction shape mismatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "transform",
    [
        lambda values: values[:, :1, :],
        lambda values: values.mean(axis=1, keepdims=True),
        lambda values: values[:1],
        lambda values: values.reshape(len(values), -1),
        lambda values: np.zeros((len(values), 1, 1)),
    ],
)
def test_broadcastable_but_wrong_reconstruction_shape_is_rejected(transform):
    X = _low_rank_dataset()
    split = _split_for(len(X))
    specification, _ = _recording_spec(transform=transform)

    with pytest.raises(ValueError, match="Broadcasting is never applied"):
        run_reconstruction_benchmark(X, split=split, methods=[specification])


def test_non_array_reconstruction_is_rejected():
    X = _low_rank_dataset()
    split = _split_for(len(X))
    specification, _ = _recording_spec(transform=lambda values: values.tolist())

    with pytest.raises(TypeError, match="NumPy array"):
        run_reconstruction_benchmark(X, split=split, methods=[specification])


def test_complex_reconstruction_is_rejected():
    X = _low_rank_dataset()
    split = _split_for(len(X))
    specification, _ = _recording_spec(
        transform=lambda values: values.astype(np.complex128),
    )

    with pytest.raises(TypeError, match="real-valued"):
        run_reconstruction_benchmark(X, split=split, methods=[specification])


# ---------------------------------------------------------------------------
# J. Non-finite and invalid input data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_datasets_are_rejected(bad_value):
    X = _low_rank_dataset()
    X = X.copy()
    X[3, 1, 2] = bad_value
    split = _split_for(len(X))
    specification, _ = _recording_spec()

    with pytest.raises(ValueError, match="NaN or infinite"):
        run_reconstruction_benchmark(X, split=split, methods=[specification])


def test_non_finite_reconstruction_is_rejected():
    X = _low_rank_dataset()
    split = _split_for(len(X))
    specification, _ = _recording_spec(
        transform=lambda values: np.full_like(values, np.nan),
    )

    with pytest.raises(ValueError, match="non-finite reconstruction"):
        run_reconstruction_benchmark(X, split=split, methods=[specification])


def test_complex_and_non_numeric_datasets_are_rejected():
    split = _split_for(30)
    specification, _ = _recording_spec()

    complex_data = np.ones((30, 4), dtype=np.complex128)
    with pytest.raises(TypeError, match="real-valued"):
        run_reconstruction_benchmark(
            complex_data,
            split=split,
            methods=[specification],
        )

    text_data = np.full((30, 4), "a", dtype="<U1")
    with pytest.raises(TypeError, match="numeric values"):
        run_reconstruction_benchmark(text_data, split=split, methods=[specification])


def test_empty_and_one_dimensional_datasets_are_rejected():
    split = _split_for(30)
    specification, _ = _recording_spec()

    with pytest.raises(ValueError, match="leading sample dimension"):
        run_reconstruction_benchmark(
            np.arange(30.0),
            split=split,
            methods=[specification],
        )

    with pytest.raises(ValueError, match="at least one sample"):
        run_reconstruction_benchmark(
            np.empty((0, 4)),
            split=split,
            methods=[specification],
        )

    with pytest.raises(ValueError, match="per-sample dimension"):
        run_reconstruction_benchmark(
            np.empty((30, 0)),
            split=split,
            methods=[specification],
        )


# ---------------------------------------------------------------------------
# K. Duplicate method names and invalid specifications
# ---------------------------------------------------------------------------


def test_duplicate_method_names_are_rejected():
    X = _low_rank_dataset()
    split = _split_for(len(X))
    first, _ = _recording_spec(name="same")
    second, _ = _recording_spec(name="same")

    with pytest.raises(ValueError, match="unique"):
        run_reconstruction_benchmark(X, split=split, methods=[first, second])


def test_empty_or_invalid_method_sequences_are_rejected():
    X = _low_rank_dataset()
    split = _split_for(len(X))

    with pytest.raises(ValueError, match="at least one BenchmarkMethod"):
        run_reconstruction_benchmark(X, split=split, methods=[])

    with pytest.raises(TypeError, match="must be a BenchmarkMethod"):
        run_reconstruction_benchmark(X, split=split, methods=[object()])


def test_invalid_metric_requests_are_rejected():
    X = _low_rank_dataset()
    split = _split_for(len(X))
    specification, _ = _recording_spec()

    with pytest.raises(ValueError, match="Unsupported metric"):
        run_reconstruction_benchmark(
            X,
            split=split,
            methods=[specification],
            metrics=("r2",),
        )

    with pytest.raises(ValueError, match="Duplicate metric"):
        run_reconstruction_benchmark(
            X,
            split=split,
            methods=[specification],
            metrics=("mse", "mse"),
        )

    with pytest.raises(TypeError, match="not a string"):
        run_reconstruction_benchmark(
            X,
            split=split,
            methods=[specification],
            metrics="mse",
        )


def test_non_serializable_configuration_is_rejected():
    with pytest.raises(TypeError, match="non-JSON value"):
        BenchmarkMethod(
            name="bad",
            method_type="fake",
            latent_dimension=1,
            factory=lambda: None,
            configuration={"callable": len},
        )


def test_factory_must_be_callable():
    with pytest.raises(TypeError, match="zero-argument callable"):
        BenchmarkMethod(
            name="bad",
            method_type="fake",
            latent_dimension=1,
            factory="not-callable",
        )


# ---------------------------------------------------------------------------
# L. Timing
# ---------------------------------------------------------------------------


def test_timings_are_finite_and_non_negative():
    X = _low_rank_dataset()
    split = _split_for(len(X))
    specification, _ = _recording_spec()

    report = run_reconstruction_benchmark(X, split=split, methods=[specification])
    result = report.results[0]

    assert np.isfinite(result.fit_seconds)
    assert np.isfinite(result.reconstruction_seconds)
    assert result.fit_seconds >= 0.0
    assert result.reconstruction_seconds >= 0.0


# ---------------------------------------------------------------------------
# M. Reproducible metadata
# ---------------------------------------------------------------------------


def _example_report() -> ReconstructionBenchmarkReport:
    X = _low_rank_dataset()
    split = _split_for(len(X))
    return run_reconstruction_benchmark(
        X,
        split=split,
        methods=[
            pca_benchmark_method("pca-k2", n_components=2),
            pca_benchmark_method("pca-k3", n_components=3),
        ],
        seed=5,
    )


def test_report_dictionary_and_json_round_trip():
    report = _example_report()

    assert ReconstructionBenchmarkReport.from_dict(report.to_dict()).to_dict() == (
        report.to_dict()
    )
    parsed = json.loads(report.to_json())
    assert ReconstructionBenchmarkReport.from_dict(parsed).to_dict() == report.to_dict()


def test_report_json_is_deterministic_and_sorted():
    report = _example_report()

    first = report.to_json()
    assert first == report.to_json()
    assert first.endswith("\n")

    payload = json.loads(first)
    assert list(payload) == sorted(payload)
    assert list(payload["results"][0]) == sorted(payload["results"][0])


def test_identity_excludes_timing_and_measured_metrics():
    report = _example_report()
    identity = report.identity()

    serialized = json.dumps(identity, sort_keys=True)
    assert "fit_seconds" not in serialized
    assert "reconstruction_seconds" not in serialized
    assert "test_metrics" not in serialized
    assert "timing" not in serialized


def test_identity_digest_ignores_timing_and_metric_values():
    report = _example_report()
    payload = report.to_dict()

    tampered = copy.deepcopy(payload)
    tampered["results"][0]["timing"]["fit_seconds"] = 999.0
    tampered["results"][0]["timing"]["reconstruction_seconds"] = 888.0
    tampered["results"][0]["test_metrics"]["mse"] = 42.0

    rebuilt = ReconstructionBenchmarkReport.from_dict(tampered)
    assert rebuilt.identity_digest() == report.identity_digest()
    assert rebuilt.to_dict() != payload


def test_identity_digest_changes_when_the_split_changes():
    X = _low_rank_dataset()
    method = pca_benchmark_method("pca-k2", n_components=2)

    first = run_reconstruction_benchmark(
        X,
        split=_split_for(len(X), fractions=(0.6, 0.2, 0.2)),
        methods=[method],
    )
    second = run_reconstruction_benchmark(
        X,
        split=_split_for(len(X), fractions=(0.5, 0.25, 0.25)),
        methods=[method],
    )
    assert first.identity_digest() != second.identity_digest()


def test_split_identity_is_stable_for_the_same_split():
    X = _low_rank_dataset()
    method = pca_benchmark_method("pca-k2", n_components=2)

    first = run_reconstruction_benchmark(X, split=_split_for(len(X)), methods=[method])
    second = run_reconstruction_benchmark(X, split=_split_for(len(X)), methods=[method])
    assert dict(first.split_identity) == dict(second.split_identity)


def test_report_rejects_malformed_payloads():
    report = _example_report()
    payload = report.to_dict()

    missing = {key: value for key, value in payload.items() if key != "metrics"}
    with pytest.raises(ValueError, match="missing required fields"):
        ReconstructionBenchmarkReport.from_dict(missing)

    extra = dict(payload)
    extra["unexpected"] = 1
    with pytest.raises(ValueError, match="unsupported fields"):
        ReconstructionBenchmarkReport.from_dict(extra)

    wrong_version = dict(payload)
    wrong_version["schema_version"] = 99
    with pytest.raises(ValueError, match="Unsupported benchmark report schema"):
        ReconstructionBenchmarkReport.from_dict(wrong_version)

    duplicated = copy.deepcopy(payload)
    duplicated["results"].append(copy.deepcopy(duplicated["results"][0]))
    with pytest.raises(ValueError, match="unique"):
        ReconstructionBenchmarkReport.from_dict(duplicated)


def test_result_rejects_non_finite_and_negative_values():
    with pytest.raises(ValueError, match="must be finite"):
        MethodBenchmarkResult(
            name="bad",
            method_type="pca",
            latent_dimension=1,
            uses_validation_partition=False,
            original_scalars_per_sample=4,
            latent_scalars_per_sample=1,
            latent_dimensionality_ratio=0.25,
            test_metrics={"mse": float("nan")},
            fit_seconds=0.1,
            reconstruction_seconds=0.1,
        )

    with pytest.raises(ValueError, match="must not be negative"):
        MethodBenchmarkResult(
            name="bad",
            method_type="pca",
            latent_dimension=1,
            uses_validation_partition=False,
            original_scalars_per_sample=4,
            latent_scalars_per_sample=1,
            latent_dimensionality_ratio=0.25,
            test_metrics={"mse": 1.0},
            fit_seconds=-1.0,
            reconstruction_seconds=0.1,
        )


def test_latent_dimensionality_ratio_is_a_dimension_ratio():
    X = _low_rank_dataset(sample_shape=(3, 4))
    split = _split_for(len(X))

    report = run_reconstruction_benchmark(
        X,
        split=split,
        methods=[pca_benchmark_method("pca-k3", n_components=3)],
    )
    result = report.results[0]
    assert result.original_scalars_per_sample == 12
    assert result.latent_scalars_per_sample == 3
    assert result.latent_dimensionality_ratio == pytest.approx(0.25)


def test_result_rejects_an_inconsistent_latent_dimensionality_ratio():
    with pytest.raises(ValueError, match="must equal"):
        MethodBenchmarkResult(
            name="misleading",
            method_type="pca",
            latent_dimension=3,
            uses_validation_partition=False,
            original_scalars_per_sample=12,
            latent_scalars_per_sample=3,
            latent_dimensionality_ratio=0.01,
            test_metrics={"mse": 1.0},
            fit_seconds=0.0,
            reconstruction_seconds=0.0,
        )


# ---------------------------------------------------------------------------
# N. Random-state isolation
# ---------------------------------------------------------------------------


def test_benchmark_does_not_leak_random_state_to_the_caller():
    X = _low_rank_dataset()
    split = _split_for(len(X))
    specification, _ = _recording_spec(draw_random=True)

    np.random.seed(1234)
    torch.manual_seed(4321)
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()

    run_reconstruction_benchmark(X, split=split, methods=[specification], seed=99)

    numpy_after = np.random.get_state()
    torch_after = torch.random.get_rng_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch_after)


def test_seeding_makes_the_random_draws_reproducible():
    X = _low_rank_dataset()
    split = _split_for(len(X))

    first_spec, first_created = _recording_spec(draw_random=True)
    run_reconstruction_benchmark(X, split=split, methods=[first_spec], seed=7)
    second_spec, second_created = _recording_spec(draw_random=True)
    run_reconstruction_benchmark(X, split=split, methods=[second_spec], seed=7)

    assert first_created[0].random_draws == second_created[0].random_draws


def test_every_method_starts_from_the_same_seeded_state():
    X = _low_rank_dataset()
    split = _split_for(len(X))
    first, first_created = _recording_spec(name="first", draw_random=True)
    second, second_created = _recording_spec(name="second", draw_random=True)

    run_reconstruction_benchmark(X, split=split, methods=[first, second], seed=3)

    assert first_created[0].random_draws == second_created[0].random_draws


@pytest.mark.parametrize("seed", [-1, 1.5, True, "3"])
def test_invalid_seeds_are_rejected(seed):
    X = _low_rank_dataset()
    split = _split_for(len(X))
    specification, _ = _recording_spec()

    with pytest.raises((TypeError, ValueError)):
        run_reconstruction_benchmark(
            X,
            split=split,
            methods=[specification],
            seed=seed,
        )


# ---------------------------------------------------------------------------
# O. Tiny real autoencoder integration and VAE determinism
# ---------------------------------------------------------------------------


def test_benchmark_runs_against_a_real_small_autoencoder():
    X = _low_rank_dataset(n_samples=40, sample_shape=(6,), rank=2)
    split = _split_for(len(X))

    report = run_reconstruction_benchmark(
        X,
        split=split,
        methods=[
            pca_benchmark_method("pca-k2", n_components=2),
            autoencoder_benchmark_method(
                "standard-ae-k2",
                model_factory=lambda: StandardAutoencoder(k=2, hidden_dims=[8]),
                latent_dimension=2,
                configuration={
                    "architecture": "StandardAutoencoder",
                    "hidden_dims": [8],
                },
                fit_kwargs={"epochs": 3, "batch_size": 8, "patience": 2},
            ),
        ],
        seed=0,
    )

    assert [result.name for result in report.results] == ["pca-k2", "standard-ae-k2"]
    for result in report.results:
        assert set(result.test_metrics) == set(METRICS)
        assert all(np.isfinite(value) for value in result.test_metrics.values())
        assert result.latent_dimension == 2
    assert report.results[0].uses_validation_partition is False
    assert report.results[1].uses_validation_partition is True


def test_variational_autoencoder_reconstruction_is_deterministic_by_default():
    X = _low_rank_dataset(n_samples=40, sample_shape=(6,), rank=2)
    split = _split_for(len(X))
    model = VariationalAutoencoder(k=2, hidden_dims=[8], beta=0.0)
    adapter = AutoencoderReconstruction(model, latent_dimension=2)

    adapter.fit(X[split.train_indices], X[split.validation_indices])
    X_test = X[split.test_indices]
    first = adapter.reconstruct(X_test)
    second = adapter.reconstruct(X_test)

    assert np.array_equal(first, second)


def test_stochastic_variational_reconstruction_is_rejected():
    model = VariationalAutoencoder(k=2, hidden_dims=[8])

    with pytest.raises(ValueError, match="Stochastic reconstruction"):
        AutoencoderReconstruction(
            model,
            latent_dimension=2,
            predict_kwargs={"stochastic": True},
        )


def test_benchmark_controlled_fit_kwargs_are_rejected():
    class _Model:
        k = 2
        is_fitted = False

        def fit(self, X, **kwargs):
            return {}

        def predict(self, X, **kwargs):
            return X

    with pytest.raises(ValueError, match="remove these"):
        AutoencoderReconstruction(
            _Model(),
            latent_dimension=2,
            fit_kwargs={"validation_split": 0.3},
        )


def test_autoencoder_adapter_requires_fit_and_predict():
    with pytest.raises(TypeError, match="callable fit"):
        AutoencoderReconstruction(object(), latent_dimension=2)
