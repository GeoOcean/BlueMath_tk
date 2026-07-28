"""Integration tests for the advanced autoencoder public API."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bluemath_tk.deeplearning import autoencoders  # noqa: E402
from bluemath_tk.deeplearning.autoencoders import (  # noqa: E402
    SpatialTokenConvLSTMTransformerAutoencoder,
    VariationalAutoencoder,
)


@pytest.fixture(autouse=True)
def _set_seed():
    np.random.seed(607)
    torch.manual_seed(607)
    torch.set_num_threads(1)


def test_advanced_autoencoders_are_publicly_exported():
    assert autoencoders.VariationalAutoencoder is VariationalAutoencoder
    assert (
        autoencoders.SpatialTokenConvLSTMTransformerAutoencoder
        is SpatialTokenConvLSTMTransformerAutoencoder
    )


def test_advanced_autoencoders_follow_common_encode_decode_contract():
    dense_X = np.random.randn(12, 6).astype("float32")
    sequence_X = np.random.randn(12, 3, 1, 6, 6).astype("float32")

    models_and_data = [
        (
            VariationalAutoencoder(
                k=3,
                hidden_dims=[8],
                beta=0.1,
                device="cpu",
            ),
            dense_X,
        ),
        (
            SpatialTokenConvLSTMTransformerAutoencoder(
                k=3,
                spatial_pool_size=(2, 2),
                d_model=8,
                n_heads=2,
                n_layers=1,
                device="cpu",
            ),
            sequence_X,
        ),
    ]

    for model, X in models_and_data:
        model.fit(
            X,
            validation_split=0.25,
            epochs=1,
            batch_size=4,
            patience=2,
            verbose=0,
        )
        latent = model.encode(X, verbose=0)
        prediction = model.predict(X, verbose=0)
        decoded = model.decode(latent, verbose=0)
        metrics = model.evaluate(X, metric="rmse", verbose=0)

        assert prediction.shape == X.shape
        assert decoded.shape == X.shape
        assert np.allclose(prediction, decoded, rtol=1e-5, atol=1e-6)
        assert metrics["metric"] == "rmse"
        assert metrics["n_samples"] == len(X)


def test_advanced_checkpoint_class_mismatch_is_rejected(tmp_path):
    X = np.random.randn(12, 6).astype("float32")
    vae = VariationalAutoencoder(
        k=3,
        hidden_dims=[8],
        device="cpu",
    )
    vae.fit(
        X,
        validation_split=0.25,
        epochs=1,
        batch_size=4,
        patience=2,
        verbose=0,
    )
    checkpoint = tmp_path / "vae.pt"
    vae.save_pytorch_model(checkpoint)

    with pytest.raises(ValueError, match="Checkpoint contains"):
        SpatialTokenConvLSTMTransformerAutoencoder.from_pytorch_model(
            checkpoint,
            device="cpu",
        )
