"""Regression tests for explicit autoencoder validation data.

``validation_data`` was added so that chronological validation membership can
be supplied exactly. These tests pin both the new semantics and the unchanged
behaviour of the historical ``validation_split`` path.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bluemath_tk.deeplearning.autoencoders import (  # noqa: E402
    ConvLSTMAutoencoder,
    OrthogonalAutoencoder,
    StandardAutoencoder,
)
from bluemath_tk.deeplearning.spatiotemporal_autoencoders import (  # noqa: E402
    SpatialTokenConvLSTMTransformerAutoencoder,
)
from bluemath_tk.deeplearning.variational_autoencoders import (  # noqa: E402
    VariationalAutoencoder,
)


def _dataset(n_samples: int = 40, n_features: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_samples, 2))
    mixing = rng.normal(size=(2, n_features))
    return latent @ mixing


def _sequence_dataset(n_samples: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_samples, 2, 1, 4, 4))


class _PartitionRecorder:
    """Mixin capturing exactly what the resolved fit partitions contain."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolved = None

    def _resolve_fit_partitions(self, X, y, validation_split, validation_data):
        resolved = super()._resolve_fit_partitions(
            X,
            y,
            validation_split,
            validation_data,
        )
        self.resolved = tuple(np.array(part, copy=True) for part in resolved)
        return resolved


class _RecordingStandardAutoencoder(_PartitionRecorder, StandardAutoencoder):
    pass


class _RecordingOrthogonalAutoencoder(_PartitionRecorder, OrthogonalAutoencoder):
    pass


class _RecordingVariationalAutoencoder(_PartitionRecorder, VariationalAutoencoder):
    pass


RECORDING_MODELS = [
    _RecordingStandardAutoencoder,
    _RecordingOrthogonalAutoencoder,
    _RecordingVariationalAutoencoder,
]


@pytest.mark.parametrize("model_class", RECORDING_MODELS)
def test_explicit_validation_data_uses_exactly_the_supplied_samples(model_class):
    X = _dataset()
    X_train = X[:30]
    X_validation = X[30:36]
    X_test = X[36:]

    model = model_class(k=2, hidden_dims=[8])
    model.fit(
        X_train,
        validation_data=(X_validation, None),
        epochs=2,
        batch_size=8,
        patience=2,
        verbose=0,
    )

    resolved_train, resolved_train_y, resolved_val, resolved_val_y = model.resolved
    assert np.array_equal(resolved_train, X_train)
    assert np.array_equal(resolved_train_y, X_train)
    assert np.array_equal(resolved_val, X_validation)
    assert np.array_equal(resolved_val_y, X_validation)

    # No test sample reaches optimisation or validation.
    test_rows = {row.tobytes() for row in X_test}
    for array in (resolved_train, resolved_val):
        for row in array:
            assert row.tobytes() not in test_rows


@pytest.mark.parametrize("model_class", RECORDING_MODELS)
def test_explicit_validation_data_preserves_training_order(model_class):
    X = _dataset()
    model = model_class(k=2, hidden_dims=[8])
    model.fit(
        X[:30],
        validation_data=(X[30:36], None),
        epochs=2,
        batch_size=8,
        patience=2,
        verbose=0,
    )
    assert np.array_equal(model.resolved[0], X[:30])


@pytest.mark.parametrize("model_class", RECORDING_MODELS)
def test_explicit_validation_data_leaves_the_global_numpy_state_untouched(model_class):
    X = _dataset()
    model = model_class(k=2, hidden_dims=[8])

    np.random.seed(17)
    state_before = np.random.get_state()
    model.fit(
        X[:30],
        validation_data=(X[30:36], None),
        epochs=2,
        batch_size=8,
        patience=2,
        verbose=0,
    )
    state_after = np.random.get_state()

    assert state_before[0] == state_after[0]
    assert np.array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


@pytest.mark.parametrize("model_class", RECORDING_MODELS)
def test_validation_split_path_is_unchanged_and_still_shuffles(model_class):
    X = _dataset()
    model = model_class(k=2, hidden_dims=[8])

    np.random.seed(3)
    state_before = np.random.get_state()
    model.fit(X, validation_split=0.25, epochs=2, batch_size=8, patience=2, verbose=0)
    state_after = np.random.get_state()

    resolved_train, _, resolved_val, _ = model.resolved
    assert len(resolved_train) == int(0.75 * len(X))
    assert len(resolved_val) == len(X) - int(0.75 * len(X))
    # The historical path consumes the global random state.
    assert not np.array_equal(state_before[1], state_after[1])
    # The random split is not the chronological tail.
    assert not np.array_equal(resolved_val, X[-len(resolved_val) :])


def test_validation_split_selection_is_reproducible_for_a_fixed_seed():
    X = _dataset()

    def _resolved_validation():
        model = _RecordingStandardAutoencoder(k=2, hidden_dims=[8])
        np.random.seed(101)
        model.fit(
            X,
            validation_split=0.2,
            epochs=1,
            batch_size=8,
            patience=1,
            verbose=0,
        )
        return model.resolved[2]

    assert np.array_equal(_resolved_validation(), _resolved_validation())


def test_validation_data_takes_precedence_over_validation_split():
    X = _dataset()
    model = _RecordingStandardAutoencoder(k=2, hidden_dims=[8])
    model.fit(
        X[:30],
        validation_split=0.9,
        validation_data=(X[30:34], None),
        epochs=1,
        batch_size=8,
        patience=1,
        verbose=0,
    )
    assert np.array_equal(model.resolved[0], X[:30])
    assert np.array_equal(model.resolved[2], X[30:34])


def test_explicit_validation_targets_are_honoured():
    X = _dataset()
    targets = X[30:34] * 2.0
    model = _RecordingStandardAutoencoder(k=2, hidden_dims=[8])
    model.fit(
        X[:30],
        validation_data=(X[30:34], targets),
        epochs=1,
        batch_size=8,
        patience=1,
        verbose=0,
    )
    assert np.array_equal(model.resolved[3], targets)


@pytest.mark.parametrize(
    ("validation_data", "error", "message"),
    [
        ([1, 2], TypeError, "must be an"),
        ((1, 2, 3), TypeError, "must be an"),
        (("not-an-array", None), TypeError, "must be a NumPy array"),
        ((np.empty((0, 6)), None), ValueError, "at least one sample"),
        ((np.zeros((4, 5)), None), ValueError, "per-sample shape"),
        ((np.zeros((4, 6, 1)), None), ValueError, "dimensions to match X"),
        ((np.full((4, 6), np.nan), None), ValueError, "only finite values"),
        ((np.full((4, 6), np.inf), None), ValueError, "only finite values"),
        ((np.zeros((4, 6)), np.zeros((3, 6))), ValueError, "same number of samples"),
        ((np.zeros((4, 6)), np.zeros((4, 5))), ValueError, "incompatible"),
        ((np.zeros((4, 6)), "not-an-array"), TypeError, "NumPy array or None"),
    ],
)
def test_invalid_validation_data_is_rejected(validation_data, error, message):
    X = _dataset()
    model = StandardAutoencoder(k=2, hidden_dims=[8])

    with pytest.raises(error, match=message):
        model.fit(
            X[:30],
            validation_data=validation_data,
            epochs=1,
            batch_size=8,
            patience=1,
            verbose=0,
        )


def test_explicit_validation_data_requires_at_least_two_training_samples():
    X = _dataset()
    model = StandardAutoencoder(k=2, hidden_dims=[8])

    with pytest.raises(ValueError, match="at least two training"):
        model.fit(
            X[:1],
            validation_data=(X[30:34], None),
            epochs=1,
            batch_size=8,
            patience=1,
            verbose=0,
        )


def test_validation_split_is_still_validated_when_no_validation_data_is_given():
    X = _dataset()
    model = StandardAutoencoder(k=2, hidden_dims=[8])

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        model.fit(X, validation_split=1.5, epochs=1, batch_size=8, verbose=0)


def test_validation_split_bounds_are_ignored_with_explicit_validation_data():
    X = _dataset()
    model = _RecordingStandardAutoencoder(k=2, hidden_dims=[8])

    model.fit(
        X[:30],
        validation_split=0.0,
        validation_data=(X[30:34], None),
        epochs=1,
        batch_size=8,
        patience=1,
        verbose=0,
    )
    assert np.array_equal(model.resolved[2], X[30:34])


def test_sequence_models_forward_validation_data_through_their_wrappers():
    X = _sequence_dataset()

    builders = (
        lambda: ConvLSTMAutoencoder(k=2),
        lambda: SpatialTokenConvLSTMTransformerAutoencoder(
            k=2,
            spatial_pool_size=(1, 1),
            d_model=8,
            n_heads=2,
            n_layers=1,
        ),
    )
    for builder in builders:
        captured = {}
        model = builder()
        original = model._resolve_fit_partitions

        def _spy(X_fit, y, validation_split, validation_data, _original=original):
            resolved = _original(X_fit, y, validation_split, validation_data)
            captured["resolved"] = tuple(np.array(part, copy=True) for part in resolved)
            return resolved

        model._resolve_fit_partitions = _spy
        model.fit(
            X[:8],
            validation_data=(X[8:10], None),
            epochs=1,
            batch_size=4,
            patience=1,
            verbose=0,
        )
        assert np.array_equal(captured["resolved"][0], X[:8])
        assert np.array_equal(captured["resolved"][2], X[8:10])


def test_fitting_twice_with_explicit_validation_data_stays_consistent():
    X = _dataset()
    model = StandardAutoencoder(k=2, hidden_dims=[8])
    for _ in range(2):
        history = model.fit(
            X[:30],
            validation_data=(X[30:36], None),
            epochs=2,
            batch_size=8,
            patience=2,
            verbose=0,
        )
        assert len(history["val_loss"]) >= 1
        assert all(np.isfinite(value) for value in history["val_loss"])
    assert model.is_fitted is True
