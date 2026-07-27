"""Tests for the public autoencoder API and checkpoint behaviour."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bluemath_tk.deeplearning.autoencoders import (  # noqa: E402
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
    np.random.seed(321)
    torch.manual_seed(321)
    torch.set_num_threads(1)


def _fit_kwargs():
    return {
        "epochs": 1,
        "batch_size": 4,
        "validation_split": 0.25,
        "patience": 2,
        "verbose": 0,
        "learning_rate": 1e-3,
    }


@pytest.mark.parametrize(
    ("factory", "shape", "decoded_shape"),
    [
        (
            lambda: StandardAutoencoder(k=3, hidden_dims=[8], device="cpu"),
            (16, 10),
            (16, 10),
        ),
        (
            lambda: OrthogonalAutoencoder(
                k=3,
                hidden_dims=[8],
                lambda_W=1e-4,
                lambda_Z=1e-4,
                device="cpu",
            ),
            (16, 10),
            (16, 10),
        ),
        (
            lambda: LSTMAutoencoder(k=4, hidden=(8, 6), device="cpu"),
            (16, 5, 3),
            (16, 5, 3),
        ),
        (
            lambda: CNNAutoencoder(k=4, device="cpu"),
            (16, 1, 8, 8),
            (16, 1, 8, 8),
        ),
        (
            lambda: VisionTransformerAutoencoder(
                k=4,
                patch_size=4,
                d_model=8,
                depth_enc=1,
                depth_dec=1,
                heads=2,
                device="cpu",
            ),
            (16, 1, 8, 8),
            (16, 1, 8, 8),
        ),
        (
            lambda: ConvLSTMAutoencoder(k=4, device="cpu"),
            (16, 3, 1, 8, 8),
            (16, 1, 8, 8),
        ),
        (
            lambda: HybridConvLSTMTransformerAutoencoder(
                k=4,
                d_model=8,
                n_heads=2,
                n_layers=1,
                efficient_attention="linear",
                device="cpu",
            ),
            (16, 3, 1, 8, 8),
            (16, 1, 8, 8),
        ),
    ],
)
def test_encode_decode_round_trip_shapes(factory, shape, decoded_shape):
    """All autoencoders should decode latent vectors to the documented shape."""
    X = np.random.randn(*shape).astype("float32")
    model = factory()
    model.fit(X, **_fit_kwargs())

    prediction = model.predict(X, batch_size=4, verbose=0)
    Z = model.encode(X, batch_size=4, verbose=0)
    decoded = model.decode(Z, batch_size=4, verbose=0)

    assert decoded.shape == decoded_shape
    assert np.isfinite(decoded).all()
    assert np.allclose(decoded, prediction, rtol=1e-5, atol=1e-6)


def test_standard_autoencoder_handles_singleton_remainder_batch():
    """StandardAutoencoder should avoid a singleton BatchNorm training batch."""
    X = np.random.randn(18, 10).astype("float32")
    model = StandardAutoencoder(k=3, hidden_dims=[8], device="cpu")
    history = model.fit(X, **_fit_kwargs())
    assert np.isfinite(history["train_loss"]).all()


def test_orthogonal_autoencoder_handles_singleton_remainder_batch():
    """OrthogonalAutoencoder should avoid singleton training batches."""
    X = np.random.randn(18, 10).astype("float32")
    model = OrthogonalAutoencoder(
        k=3,
        hidden_dims=[8],
        lambda_W=1e-4,
        lambda_Z=1e-4,
        device="cpu",
    )
    history = model.fit(X, **_fit_kwargs())
    assert np.isfinite(history["train_loss"]).all()


def test_batch_size_one_is_rejected_for_batchnorm1d_autoencoders():
    """BatchNorm1d autoencoders should reject singleton training batches."""
    X = np.random.randn(8, 10).astype("float32")
    model = StandardAutoencoder(k=3, hidden_dims=[8], device="cpu")

    with pytest.raises(ValueError, match="batch_size=1"):
        model.fit(
            X,
            validation_split=0.25,
            epochs=1,
            batch_size=1,
            patience=2,
            verbose=0,
        )


def test_batch_size_one_remains_available_without_batchnorm1d():
    """Models without BatchNorm1d should retain explicit batch_size=1."""
    X = np.random.randn(6, 3, 2).astype("float32")
    model = LSTMAutoencoder(k=2, hidden=(4, 3), device="cpu")

    history = model.fit(
        X,
        validation_split=0.33,
        epochs=1,
        batch_size=1,
        patience=2,
        verbose=0,
    )

    assert np.isfinite(history["train_loss"]).all()


def test_orthogonal_autoencoder_handles_singleton_validation_batch():
    """Latent decorrelation should be neutral for a one-sample validation batch."""
    X = np.random.randn(5, 6).astype("float32")
    model = OrthogonalAutoencoder(
        k=2,
        hidden_dims=[4],
        lambda_W=1e-4,
        lambda_Z=1e-4,
        device="cpu",
    )

    history = model.fit(
        X,
        validation_split=0.2,
        epochs=1,
        batch_size=2,
        patience=2,
        verbose=0,
    )

    assert np.isfinite(history["val_loss"]).all()


def test_repeated_fit_rejects_incompatible_sample_shape():
    """A built model should not silently accept a different sample shape."""
    X = np.random.randn(16, 10).astype("float32")
    model = StandardAutoencoder(k=3, hidden_dims=[8], device="cpu")
    model.fit(X, **_fit_kwargs())

    X_incompatible = np.random.randn(16, 12).astype("float32")
    with pytest.raises(ValueError, match="incompatible"):
        model.fit(X_incompatible, **_fit_kwargs())


def test_inference_methods_validate_batch_size():
    """Prediction, encoding, and decoding should reject invalid batch sizes."""
    X = np.random.randn(16, 10).astype("float32")
    model = StandardAutoencoder(k=3, hidden_dims=[8], device="cpu")
    model.fit(X, **_fit_kwargs())
    Z = model.encode(X, batch_size=4, verbose=0)

    with pytest.raises(ValueError, match="batch_size"):
        model.predict(X, batch_size=0, verbose=0)
    with pytest.raises(ValueError, match="batch_size"):
        model.encode(X, batch_size=0, verbose=0)
    with pytest.raises(ValueError, match="batch_size"):
        model.decode(Z, batch_size=0, verbose=0)


def test_fit_validates_sample_dimension_and_validation_split():
    """Fit should reject ambiguous inputs and invalid validation splits."""
    model = StandardAutoencoder(k=2, hidden_dims=[4], device="cpu")

    with pytest.raises(ValueError, match="leading sample dimension"):
        model.fit(np.ones(10, dtype="float32"), **_fit_kwargs())

    X = np.ones((8, 3), dtype="float32")
    with pytest.raises(ValueError, match="validation_split"):
        model.fit(
            X,
            validation_split=0.0,
            epochs=1,
            batch_size=4,
            patience=2,
            verbose=0,
        )


def test_decode_accepts_one_latent_vector():
    """A one-dimensional latent vector should decode as one sample."""
    X = np.random.randn(16, 10).astype("float32")
    model = StandardAutoencoder(k=3, hidden_dims=[8], device="cpu")
    model.fit(X, **_fit_kwargs())
    Z = model.encode(X, batch_size=4, verbose=0)

    decoded = model.decode(Z[0], verbose=0)

    assert decoded.shape == (1, 10)


def test_standard_autoencoder_reconstruction_convenience_methods():
    """Convenience metrics should match direct reconstruction calculations."""
    X = np.random.randn(16, 10).astype("float32")
    model = StandardAutoencoder(k=3, hidden_dims=[8], device="cpu")
    model.fit(X, **_fit_kwargs())

    prediction = model.predict(X, batch_size=4, verbose=0)
    errors = model.reconstruction_error(
        X,
        metric="mse",
        reduction="sample",
        batch_size=4,
    )
    summary = model.evaluate_reconstruction(
        X,
        metric="rmse",
        batch_size=4,
    )

    manual = np.mean((prediction - X) ** 2, axis=1)
    assert np.allclose(errors, manual)
    assert summary["metric"] == "rmse"
    assert summary["n_samples"] == len(X)


def test_convlstm_metrics_use_last_frame_as_default_target():
    """ConvLSTM metrics should compare predictions with the final input frame."""
    X = np.random.randn(16, 3, 1, 8, 8).astype("float32")
    model = ConvLSTMAutoencoder(k=4, device="cpu")
    model.fit(X, **_fit_kwargs())

    prediction = model.predict(X, batch_size=4, verbose=0)
    errors = model.reconstruction_error(
        X,
        metric="mse",
        reduction="sample",
        batch_size=4,
    )
    manual = np.mean(
        (prediction - X[:, -1]) ** 2,
        axis=(1, 2, 3),
    )

    assert np.allclose(errors, manual)


def test_standard_autoencoder_checkpoint_round_trip(tmp_path):
    """A standard autoencoder checkpoint should restore identical predictions."""
    X = np.random.randn(16, 10).astype("float32")
    model = StandardAutoencoder(k=3, hidden_dims=[8], device="cpu")
    model.fit(X, **_fit_kwargs())

    checkpoint = tmp_path / "standard_autoencoder.pt"
    model.save_pytorch_model(checkpoint)
    loaded = StandardAutoencoder.from_pytorch_model(
        checkpoint,
        device="cpu",
    )
    loaded_into_instance = StandardAutoencoder(device="cpu")
    loaded_into_instance.load_pytorch_model(checkpoint)

    original = model.predict(X, batch_size=4, verbose=0)
    restored = loaded.predict(X, batch_size=4, verbose=0)
    restored_instance = loaded_into_instance.predict(
        X,
        batch_size=4,
        verbose=0,
    )

    assert loaded.is_fitted
    assert loaded_into_instance.is_fitted
    assert np.allclose(original, restored)
    assert np.allclose(original, restored_instance)


def test_convlstm_checkpoint_round_trip(tmp_path):
    """A ConvLSTM checkpoint should restore identical predictions."""
    X = np.random.randn(16, 3, 1, 8, 8).astype("float32")
    model = ConvLSTMAutoencoder(k=4, device="cpu")
    model.fit(X, **_fit_kwargs())

    checkpoint = tmp_path / "convlstm_autoencoder.pt"
    model.save_pytorch_model(checkpoint)
    loaded = ConvLSTMAutoencoder.from_pytorch_model(
        checkpoint,
        device="cpu",
    )

    original = model.predict(X, batch_size=4, verbose=0)
    restored = loaded.predict(X, batch_size=4, verbose=0)

    assert loaded.is_fitted
    assert np.allclose(original, restored)
