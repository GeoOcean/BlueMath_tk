"""Tests for the spatial-token spatiotemporal autoencoder."""

import numpy as np
import pytest
import torch

from bluemath_tk.deeplearning.autoencoders import (
    SpatialTokenConvLSTMTransformerAutoencoder,
)
from bluemath_tk.deeplearning.spatiotemporal_autoencoders import (
    _FactorizedSpatiotemporalBlock,
)


@pytest.fixture(autouse=True)
def _set_seed():
    previous_threads = torch.get_num_threads()
    np.random.seed(503)
    torch.manual_seed(503)
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


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
        model.encode(
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
    for target in (X[:, -1], X[:, :1]):
        with pytest.raises(ValueError, match="Target shape"):
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


def test_spatial_token_singleton_tiny_grid_is_safe_and_input_sensitive():
    outer = SpatialTokenConvLSTMTransformerAutoencoder(
        k=2,
        spatial_pool_size=(1, 1),
        d_model=4,
        n_heads=1,
        n_layers=1,
        device="cpu",
    )
    inner = outer._build_model((2, 1, 1, 1, 1)).eval()
    inputs = torch.tensor(
        [
            [[[[0.0]]]],
            [[[[1.0]]]],
        ]
    )
    with torch.no_grad():
        latent = inner.encode_forward(inputs)
        reconstruction = inner(inputs)

    assert reconstruction.shape == (2, 1, 1, 1, 1)
    assert torch.isfinite(reconstruction).all()
    assert not torch.allclose(latent[0], latent[1], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k": 0}, "k"),
        ({"spatial_pool_size": (0, 2)}, "spatial_pool_size"),
        ({"spatial_pool_size": [2, 2]}, "spatial_pool_size"),
        ({"d_model": 0}, "d_model"),
        ({"d_model": 1}, "d_model"),
        ({"d_model": 2}, "d_model"),
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
    model = _model()._build_model(tuple(X.shape))

    reconstruction = model(X)
    reconstruction.square().mean().backward()

    parameters = {
        "encoder_temporal": (model.encoder_blocks[0].temporal_attention.in_proj_weight),
        "encoder_spatial": (model.encoder_blocks[0].spatial_attention.in_proj_weight),
        "decoder_temporal": (model.decoder_blocks[0].temporal_attention.in_proj_weight),
        "decoder_spatial": (model.decoder_blocks[0].spatial_attention.in_proj_weight),
        "latent": model.latent.weight,
        "time_query": model.decoder_time_query,
        "space_query": model.decoder_space_query,
    }
    for name, parameter in parameters.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name


def _set_identity_attention(attention):
    dimension = attention.embed_dim
    identity = torch.eye(dimension)
    with torch.no_grad():
        attention.in_proj_weight.zero_()
        attention.in_proj_weight[:dimension].copy_(identity)
        attention.in_proj_weight[dimension : 2 * dimension].copy_(identity)
        attention.in_proj_weight[2 * dimension :].copy_(identity)
        attention.in_proj_bias.zero_()
        attention.out_proj.weight.copy_(identity)
        attention.out_proj.bias.zero_()


def _zero_module(module):
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()


def test_factorized_block_temporal_attention_mixes_timesteps():
    block = _FactorizedSpatiotemporalBlock(d_model=4, n_heads=1)
    _set_identity_attention(block.temporal_attention)
    _zero_module(block.spatial_attention)
    _zero_module(block.feed_forward)

    values = torch.zeros(1, 3, 2, 4)
    values[0, 0, 0] = torch.tensor([1.0, -1.0, 0.5, -0.5])
    enabled = block(values.clone())
    with torch.no_grad():
        block.temporal_attention.out_proj.weight.zero_()
    disabled = block(values.clone())

    cross_time_change = torch.abs(enabled[0, 1, 0] - disabled[0, 1, 0])
    assert torch.max(cross_time_change) > 1e-4


def test_factorized_block_spatial_attention_mixes_tokens():
    block = _FactorizedSpatiotemporalBlock(d_model=4, n_heads=1)
    _zero_module(block.temporal_attention)
    _set_identity_attention(block.spatial_attention)
    _zero_module(block.feed_forward)

    values = torch.zeros(1, 2, 3, 4)
    values[0, 0, 0] = torch.tensor([1.0, -1.0, 0.5, -0.5])
    enabled = block(values.clone())
    with torch.no_grad():
        block.spatial_attention.out_proj.weight.zero_()
    disabled = block(values.clone())

    cross_space_change = torch.abs(enabled[0, 0, 1] - disabled[0, 0, 1])
    assert torch.max(cross_space_change) > 1e-4


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
