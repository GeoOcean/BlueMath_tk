"""Tests for the spatial-token spatiotemporal autoencoder."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bluemath_tk.deeplearning.autoencoders import (  # noqa: E402
    SpatialTokenConvLSTMTransformerAutoencoder,
)


@pytest.fixture(autouse=True)
def _set_seed():
    np.random.seed(503)
    torch.manual_seed(503)
    torch.set_num_threads(1)


def _model():
    return SpatialTokenConvLSTMTransformerAutoencoder(
        k=4,
        spatial_pool_size=(2, 2),
        d_model=8,
        n_heads=2,
        n_layers=1,
        device="cpu",
    )


def _fit_model(X):
    model = _model()
    model.fit(
        X,
        validation_split=0.25,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        patience=2,
        verbose=0,
    )
    return model


def test_spatial_token_autoencoder_reconstructs_complete_sequence():
    X = np.random.randn(12, 3, 1, 8, 10).astype("float32")
    model = _fit_model(X)

    latent = model.encode(X, batch_size=4, verbose=0)
    prediction = model.predict(X, batch_size=4, verbose=0)
    decoded = model.decode(latent, batch_size=4, verbose=0)

    assert latent.shape == (12, 4)
    assert prediction.shape == X.shape
    assert decoded.shape == X.shape
    assert np.allclose(prediction, decoded, rtol=1e-5, atol=1e-6)
    assert np.isfinite(prediction).all()


def test_spatial_token_single_vector_decoding():
    X = np.random.randn(12, 3, 1, 7, 9).astype("float32")
    model = _fit_model(X)
    latent = model.encode(X, verbose=0)

    decoded = model.decode(latent[0], verbose=0)

    assert decoded.shape == (1, 3, 1, 7, 9)


def test_spatial_token_fixed_sample_shape_validation():
    X = np.random.randn(12, 3, 1, 8, 8).astype("float32")
    model = _fit_model(X)

    with pytest.raises(ValueError, match="Expected per-sample shape"):
        model.predict(
            np.random.randn(4, 4, 1, 8, 8).astype("float32"),
            verbose=0,
        )
    with pytest.raises(ValueError, match="Expected per-sample shape"):
        model.predict(
            np.random.randn(4, 3, 2, 8, 8).astype("float32"),
            verbose=0,
        )


def test_spatial_token_rejects_non_sequence_input():
    model = _model()
    X = np.random.randn(12, 1, 8, 8).astype("float32")

    with pytest.raises(ValueError, match="5D input"):
        model.fit(
            X,
            validation_split=0.25,
            epochs=1,
            batch_size=4,
            patience=2,
            verbose=0,
        )


def test_spatial_token_rejects_non_sequence_targets():
    X = np.random.randn(12, 3, 1, 8, 8).astype("float32")
    frame_target = X[:, -1]
    broadcastable_target = X[:, :1]

    for target in (frame_target, broadcastable_target):
        with pytest.raises(ValueError, match="same shape as X"):
            _model().fit(
                X,
                y=target,
                validation_split=0.25,
                epochs=1,
                batch_size=4,
                patience=2,
                verbose=0,
            )


def test_spatial_token_rejects_pool_larger_than_encoded_grid():
    X = np.random.randn(12, 3, 1, 6, 6).astype("float32")
    model = SpatialTokenConvLSTMTransformerAutoencoder(
        k=4,
        spatial_pool_size=(3, 3),
        d_model=8,
        n_heads=2,
        n_layers=1,
        device="cpu",
    )

    with pytest.raises(ValueError, match="spatial_pool_size"):
        model.fit(
            X,
            validation_split=0.25,
            epochs=1,
            batch_size=4,
            patience=2,
            verbose=0,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k": 0}, "k"),
        ({"spatial_pool_size": (0, 2)}, "spatial_pool_size"),
        ({"spatial_pool_size": [2, 2]}, "spatial_pool_size"),
        ({"d_model": 0}, "d_model"),
        ({"n_heads": 0}, "n_heads"),
        ({"d_model": 10, "n_heads": 3}, "divisible"),
        ({"n_layers": 0}, "n_layers"),
    ],
)
def test_spatial_token_rejects_invalid_constructor_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SpatialTokenConvLSTMTransformerAutoencoder(**kwargs)


def test_spatial_token_gradients_reach_space_time_and_latent_paths():
    X = torch.randn(4, 3, 1, 6, 6)
    outer = _model()
    model = outer._build_model(tuple(X.shape))

    reconstruction = model(X)
    reconstruction.square().mean().backward()

    names = {
        "encoder_temporal": (
            model.encoder_blocks[0].temporal_attention.in_proj_weight
        ),
        "encoder_spatial": (
            model.encoder_blocks[0].spatial_attention.in_proj_weight
        ),
        "decoder_temporal": (
            model.decoder_blocks[0].temporal_attention.in_proj_weight
        ),
        "decoder_spatial": (
            model.decoder_blocks[0].spatial_attention.in_proj_weight
        ),
        "latent": model.latent.weight,
        "time_query": model.decoder_time_query,
        "space_query": model.decoder_space_query,
    }
    for name, parameter in names.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name


def test_spatial_token_decoder_depends_on_time_and_space_queries():
    outer = _model()
    model = outer._build_model((4, 3, 1, 8, 8))
    z = torch.zeros(2, 4)

    reconstruction = model.decode_forward(z)
    time_difference = torch.max(
        torch.abs(reconstruction[:, 0] - reconstruction[:, 1])
    )
    spatial_variation = reconstruction.var(dim=(-2, -1)).mean()

    assert time_difference > 1e-7
    assert spatial_variation > 1e-10


def test_spatial_token_checkpoint_round_trip(tmp_path):
    X = np.random.randn(12, 3, 1, 8, 8).astype("float32")
    model = _fit_model(X)
    checkpoint = tmp_path / "spatial_token.pt"

    original = model.predict(X, verbose=0)
    model.save_pytorch_model(checkpoint)
    restored = SpatialTokenConvLSTMTransformerAutoencoder.from_pytorch_model(
        checkpoint,
        device="cpu",
    )

    assert restored.spatial_pool_size == (2, 2)
    assert np.allclose(original, restored.predict(X, verbose=0))


def test_spatial_token_manual_optimization_reduces_loss():
    coordinate = torch.linspace(-1.0, 1.0, 6)
    base = coordinate[:, None] + coordinate[None, :]
    frames = torch.stack([base, base.square(), torch.sin(base)], dim=0)
    X = frames[None, :, None].repeat(4, 1, 1, 1, 1)

    model = _model()._build_model(tuple(X.shape))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    with torch.no_grad():
        initial = torch.mean((model(X) - X) ** 2).item()
    for _ in range(10):
        optimizer.zero_grad()
        loss = torch.mean((model(X) - X) ** 2)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = torch.mean((model(X) - X) ** 2).item()

    assert final < initial
