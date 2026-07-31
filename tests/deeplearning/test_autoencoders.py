"""
Smoke tests for BlueMath_tk autoencoders.

These tests are intentionally small so they can run quickly in CI and in a local
Anaconda environment.

Run from the repository root with:

    pytest -q tests/deeplearning/test_autoencoders.py

"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bluemath_tk.deeplearning.autoencoders import (
    CNNAutoencoder,
    ConvLSTMAutoencoder,
    HybridConvLSTMTransformerAutoencoder,
    LSTMAutoencoder,
    OrthogonalAutoencoder,
    StandardAutoencoder,
    VisionTransformerAutoencoder,
)


@pytest.fixture(autouse=True)
def _set_reproducible_seed():
    """Keep tests deterministic and avoid excessive CPU thread use."""
    np.random.seed(123)
    torch.manual_seed(123)
    torch.set_num_threads(1)


def _fit_kwargs():
    """Common tiny training configuration."""
    return dict(
        epochs=1,
        batch_size=4,
        validation_split=0.25,
        patience=2,
        verbose=0,
        learning_rate=1e-3,
    )


def test_standard_autoencoder_fit_predict_encode_shapes():
    """StandardAutoencoder should reconstruct 2D tabular input."""
    X = np.random.randn(16, 10).astype("float32")

    ae = StandardAutoencoder(
        k=3,
        hidden_dims=[8],
        device="cpu",
    )

    history = ae.fit(X, **_fit_kwargs())
    X_hat = ae.predict(X, batch_size=4, verbose=0)
    Z = ae.encode(X, batch_size=4, verbose=0)

    assert set(history) == {"train_loss", "val_loss"}
    assert len(history["train_loss"]) >= 1
    assert len(history["val_loss"]) >= 1
    assert X_hat.shape == X.shape
    assert Z.shape == (16, 3)
    assert np.isfinite(X_hat).all()
    assert np.isfinite(Z).all()


def test_orthogonal_autoencoder_fit_predict_encode_shapes():
    """OrthogonalAutoencoder should reconstruct 2D tabular input."""
    X = np.random.randn(16, 10).astype("float32")

    ae = OrthogonalAutoencoder(
        k=3,
        hidden_dims=[8],
        lambda_W=1e-4,
        lambda_Z=1e-4,
        device="cpu",
    )

    history = ae.fit(X, **_fit_kwargs())
    X_hat = ae.predict(X, batch_size=4, verbose=0)
    Z = ae.encode(X, batch_size=4, verbose=0)

    assert set(history) == {"train_loss", "val_loss"}
    assert X_hat.shape == X.shape
    assert Z.shape == (16, 3)
    assert np.isfinite(X_hat).all()
    assert np.isfinite(Z).all()


def test_lstm_autoencoder_fit_predict_encode_shapes():
    """LSTMAutoencoder should reconstruct sequence input."""
    X = np.random.randn(16, 5, 3).astype("float32")

    ae = LSTMAutoencoder(
        k=4,
        hidden=(8, 6),
        device="cpu",
    )

    history = ae.fit(X, **_fit_kwargs())
    X_hat = ae.predict(X, batch_size=4, verbose=0)
    Z = ae.encode(X, batch_size=4, verbose=0)

    assert set(history) == {"train_loss", "val_loss"}
    assert X_hat.shape == X.shape
    assert Z.shape == (16, 4)
    assert np.isfinite(X_hat).all()
    assert np.isfinite(Z).all()


def test_cnn_autoencoder_fit_predict_encode_shapes():
    """CNNAutoencoder should reconstruct channels-first image/grid input."""
    X = np.random.randn(16, 1, 8, 8).astype("float32")

    ae = CNNAutoencoder(
        k=4,
        device="cpu",
    )

    history = ae.fit(X, **_fit_kwargs())
    X_hat = ae.predict(X, batch_size=4, verbose=0)
    Z = ae.encode(X, batch_size=4, verbose=0)

    assert set(history) == {"train_loss", "val_loss"}
    assert X_hat.shape == X.shape
    assert Z.shape == (16, 4)
    assert np.isfinite(X_hat).all()
    assert np.isfinite(Z).all()


def test_vit_autoencoder_d_model_can_differ_from_patch_dimension():
    """
    VisionTransformerAutoencoder should allow d_model != patch_size*patch_size*C.

    For C=1 and patch_size=4, patch dimension is 16.
    This test uses d_model=8 to catch missing decoder projection bugs.
    """
    X = np.random.randn(16, 1, 8, 8).astype("float32")

    ae = VisionTransformerAutoencoder(
        k=4,
        patch_size=4,
        d_model=8,
        depth_enc=1,
        depth_dec=1,
        heads=2,
        device="cpu",
    )

    history = ae.fit(X, **_fit_kwargs())
    X_hat = ae.predict(X, batch_size=4, verbose=0)
    Z = ae.encode(X, batch_size=4, verbose=0)

    assert set(history) == {"train_loss", "val_loss"}
    assert X_hat.shape == X.shape
    assert Z.shape == (16, 4)
    assert np.isfinite(X_hat).all()
    assert np.isfinite(Z).all()


def test_convlstm_autoencoder_reconstructs_complete_sequence():
    """ConvLSTMAutoencoder should reconstruct its complete input sequence."""
    X = np.random.randn(16, 3, 1, 8, 8).astype("float32")

    ae = ConvLSTMAutoencoder(
        k=4,
        device="cpu",
    )

    history = ae.fit(X, **_fit_kwargs())
    X_hat = ae.predict(X, batch_size=4, verbose=0)
    Z = ae.encode(X, batch_size=4, verbose=0)

    assert set(history) == {"train_loss", "val_loss"}
    assert X_hat.shape == X.shape
    assert Z.shape == (16, 4)
    assert np.isfinite(X_hat).all()
    assert np.isfinite(Z).all()


def test_hybrid_autoencoder_reconstructs_complete_sequence():
    """The hybrid autoencoder should reconstruct its complete input sequence."""
    X = np.random.randn(16, 3, 1, 8, 8).astype("float32")

    ae = HybridConvLSTMTransformerAutoencoder(
        k=4,
        d_model=8,
        n_heads=2,
        n_layers=1,
        efficient_attention="linear",
        device="cpu",
    )

    history = ae.fit(X, **_fit_kwargs())
    X_hat = ae.predict(X, batch_size=4, verbose=0)
    Z = ae.encode(X, batch_size=4, verbose=0)

    assert set(history) == {"train_loss", "val_loss"}
    assert X_hat.shape == X.shape
    assert Z.shape == (16, 4)
    assert np.isfinite(X_hat).all()
    assert np.isfinite(Z).all()


def test_standard_autoencoder_multidimensional_flatten_contract():
    """
    Document the current ambiguity in StandardAutoencoder.

    The docstring says multidimensional inputs are automatically flattened.
    If that is intended, fit/predict should work for (B, C, H) input.
    """
    X = np.random.randn(16, 2, 5).astype("float32")

    ae = StandardAutoencoder(
        k=3,
        hidden_dims=[8],
        device="cpu",
    )

    history = ae.fit(X, **_fit_kwargs())
    X_hat = ae.predict(X, batch_size=4, verbose=0)
    Z = ae.encode(X, batch_size=4, verbose=0)

    assert set(history) == {"train_loss", "val_loss"}
    assert X_hat.shape == X.shape
    assert Z.shape == (16, 3)
    assert np.isfinite(X_hat).all()
    assert np.isfinite(Z).all()