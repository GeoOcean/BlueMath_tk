"""Tests for sequence-only spatiotemporal autoencoders."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bluemath_tk.deeplearning.autoencoders import (  # noqa: E402
    ConvLSTMAutoencoder,
    HybridConvLSTMTransformerAutoencoder,
)


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(123)
    torch.manual_seed(123)
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


def _factories():
    return [
        lambda: ConvLSTMAutoencoder(k=4, device="cpu"),
        lambda: HybridConvLSTMTransformerAutoencoder(
            k=4,
            d_model=8,
            n_heads=2,
            n_layers=1,
            efficient_attention="linear",
            device="cpu",
        ),
    ]


@pytest.mark.parametrize("factory", _factories())
def test_spatiotemporal_autoencoders_reconstruct_full_input(factory):
    X = np.random.randn(8, 3, 1, 8, 8).astype("float32")
    model = factory()
    model.fit(X, **_fit_kwargs())

    prediction = model.predict(X, batch_size=4, verbose=0)
    latent = model.encode(X, batch_size=4, verbose=0)
    decoded = model.decode(latent, batch_size=4, verbose=0)
    errors = model.reconstruction_error(
        X,
        metric="mse",
        reduction="sample",
        batch_size=4,
    )

    manual = ((prediction - X) ** 2).reshape(len(X), -1).mean(axis=1)
    assert prediction.shape == X.shape
    assert latent.shape == (8, 4)
    assert decoded.shape == X.shape
    assert errors.shape == (8,)
    assert np.allclose(errors, manual)
    assert np.allclose(prediction, decoded, rtol=1e-5, atol=1e-6)


def test_reconstruction_mode_argument_is_not_supported():
    with pytest.raises(TypeError, match="reconstruction_mode"):
        ConvLSTMAutoencoder(reconstruction_mode="last_frame")


def test_full_sequence_autoencoder_rejects_frame_target():
    X = np.random.randn(8, 3, 1, 8, 8).astype("float32")
    model = ConvLSTMAutoencoder(k=4, device="cpu")

    with pytest.raises(ValueError, match="Target shape"):
        model.fit(X, y=X[:, -1], **_fit_kwargs())


def test_full_sequence_autoencoder_rejects_different_sequence_shape():
    X = np.random.randn(8, 3, 1, 8, 8).astype("float32")
    model = ConvLSTMAutoencoder(k=4, device="cpu")
    model.fit(X, **_fit_kwargs())

    wrong_length = np.random.randn(8, 4, 1, 8, 8).astype("float32")
    with pytest.raises(ValueError, match="per-sample shape"):
        model.predict(wrong_length, batch_size=4, verbose=0)


@pytest.mark.parametrize(
    ("factory", "model_class"),
    [
        (
            lambda: ConvLSTMAutoencoder(k=4, device="cpu"),
            ConvLSTMAutoencoder,
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
            HybridConvLSTMTransformerAutoencoder,
        ),
    ],
)
def test_sequence_checkpoint_round_trip(tmp_path, factory, model_class):
    X = np.random.randn(8, 3, 1, 8, 8).astype("float32")
    model = factory()
    model.fit(X, **_fit_kwargs())

    path = tmp_path / f"{model_class.__name__}.pt"
    model.save_pytorch_model(path)
    restored = model_class.from_pytorch_model(path, device="cpu")

    original = model.predict(X, batch_size=4, verbose=0)
    reloaded = restored.predict(X, batch_size=4, verbose=0)
    assert np.allclose(original, reloaded)


def test_sequence_decode_accepts_one_latent_vector():
    X = np.random.randn(8, 3, 1, 8, 8).astype("float32")
    model = ConvLSTMAutoencoder(k=4, device="cpu")
    model.fit(X, **_fit_kwargs())
    latent = model.encode(X, batch_size=4, verbose=0)

    decoded = model.decode(latent[0], verbose=0)
    assert decoded.shape == (1, 3, 1, 8, 8)


def test_sequence_decoder_crops_spatial_padding():
    X = np.random.randn(8, 2, 1, 7, 9).astype("float32")
    model = ConvLSTMAutoencoder(k=4, device="cpu")
    model.fit(X, **_fit_kwargs())

    prediction = model.predict(X, batch_size=4, verbose=0)
    assert prediction.shape == X.shape


def test_hybrid_sequence_decoder_supports_standard_attention():
    X = np.random.randn(8, 2, 1, 8, 8).astype("float32")
    model = HybridConvLSTMTransformerAutoencoder(
        k=4,
        d_model=8,
        n_heads=2,
        n_layers=1,
        efficient_attention=None,
        device="cpu",
    )
    model.fit(X, **_fit_kwargs())

    prediction = model.predict(X, batch_size=4, verbose=0)
    assert prediction.shape == X.shape


@pytest.mark.parametrize(
    ("factory", "parameter_names"),
    [
        (
            lambda: ConvLSTMAutoencoder(k=4, device="cpu"),
            [
                "decoder_time_embedding",
                "temporal_decoder.weight_ih_l0",
            ],
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
            [
                "decoder_time_queries",
                "latent_to_decoder.weight",
                "decoder_transformer_blocks.0.attn.Wv.weight",
                "frame_seed.weight",
            ],
        ),
    ],
)
def test_sequence_decoder_parameters_receive_gradients(factory, parameter_names):
    X = np.random.randn(4, 3, 1, 8, 8).astype("float32")
    model = factory()
    model.model = model._build_model(X.shape).to(model.device)
    model.model.train()

    tensor = torch.as_tensor(X, dtype=torch.float32, device=model.device)
    prediction = model.model(tensor)
    loss = torch.mean((prediction - tensor) ** 2)
    loss.backward()

    parameters = dict(model.model.named_parameters())
    for name in parameter_names:
        gradient = parameters[name].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient).item() > 0


@pytest.mark.parametrize("factory", _factories())
def test_sequence_decoder_uses_temporal_position(factory):
    model = factory()
    model.model = model._build_model((4, 3, 1, 8, 8)).to(model.device)
    model.model.eval()

    latent = torch.zeros(2, 4, device=model.device)
    with torch.no_grad():
        decoded = model.model.decode_forward(latent)

    assert decoded.shape == (2, 3, 1, 8, 8)
    assert not torch.allclose(decoded[:, 0], decoded[:, 1])


def test_sequence_decoder_parameters_are_always_present():
    conv = ConvLSTMAutoencoder(k=4, device="cpu")
    hybrid = HybridConvLSTMTransformerAutoencoder(
        k=4,
        d_model=8,
        n_heads=2,
        n_layers=1,
        device="cpu",
    )

    conv_keys = set(conv._build_model((4, 3, 1, 8, 8)).state_dict())
    hybrid_keys = set(hybrid._build_model((4, 3, 1, 8, 8)).state_dict())
    assert any("temporal_decoder" in key for key in conv_keys)
    assert any("decoder_transformer_blocks" in key for key in hybrid_keys)


def test_fit_rejects_non_numpy_input_before_target_validation():
    X = np.random.randn(8, 3, 1, 8, 8).astype("float32")
    model = ConvLSTMAutoencoder(k=4, device="cpu")

    with pytest.raises(TypeError, match="X must be a NumPy array"):
        model.fit(X.tolist(), y=X, **_fit_kwargs())
