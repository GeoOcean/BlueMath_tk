"""Acceptance tests for the PCA-like latent feature.

These tests are intended to be added together with the integration edits
described in docs/PCA_LIKE_LATENT_FEATURE_IMPLEMENTATION.md.

Before those edits they will fail because the public constructors do not yet
accept ``latent_structure``.
"""

import numpy as np
import pytest

from bluemath_tk.deeplearning.autoencoders import (
    CNNAutoencoder,
    ConvLSTMAutoencoder,
    HybridConvLSTMTransformerAutoencoder,
    LSTMAutoencoder,
    SpatialTokenConvLSTMTransformerAutoencoder,
    StandardAutoencoder,
    VariationalAutoencoder,
    VisionTransformerAutoencoder,
)


def _cases():
    return [
        (
            lambda: StandardAutoencoder(
                k=3,
                hidden_dims=[12, 8],
                latent_structure="pca_like",
                device="cpu",
            ),
            np.random.default_rng(1).normal(size=(12, 10)).astype("float32"),
        ),
        (
            lambda: LSTMAutoencoder(
                k=3,
                hidden=(8, 6),
                latent_structure="pca_like",
                device="cpu",
            ),
            np.random.default_rng(2).normal(size=(12, 3, 4)).astype("float32"),
        ),
        (
            lambda: CNNAutoencoder(
                k=3,
                latent_structure="pca_like",
                device="cpu",
            ),
            np.random.default_rng(3).normal(size=(10, 1, 8, 8)).astype("float32"),
        ),
        (
            lambda: VisionTransformerAutoencoder(
                k=3,
                patch_size=4,
                d_model=16,
                depth_enc=1,
                depth_dec=1,
                heads=4,
                latent_structure="pca_like",
                device="cpu",
            ),
            np.random.default_rng(4).normal(size=(10, 1, 8, 8)).astype("float32"),
        ),
        (
            lambda: ConvLSTMAutoencoder(
                k=3,
                latent_structure="pca_like",
                device="cpu",
            ),
            np.random.default_rng(5).normal(size=(12, 2, 1, 8, 8)).astype("float32"),
        ),
        (
            lambda: HybridConvLSTMTransformerAutoencoder(
                k=3,
                d_model=16,
                n_heads=4,
                n_layers=1,
                latent_structure="pca_like",
                device="cpu",
            ),
            np.random.default_rng(6).normal(size=(12, 2, 1, 8, 8)).astype("float32"),
        ),
        (
            lambda: SpatialTokenConvLSTMTransformerAutoencoder(
                k=3,
                spatial_pool_size=(1, 1),
                d_model=16,
                n_heads=4,
                n_layers=1,
                latent_structure="pca_like",
                device="cpu",
            ),
            np.random.default_rng(7).normal(size=(12, 2, 1, 8, 8)).astype("float32"),
        ),
        (
            lambda: VariationalAutoencoder(
                k=3,
                hidden_dims=[12, 8],
                beta=0.01,
                validation_mc_samples=1,
                latent_structure="pca_like",
                device="cpu",
            ),
            np.random.default_rng(8).normal(size=(12, 10)).astype("float32"),
        ),
    ]


@pytest.mark.parametrize(("factory", "X"), _cases())
def test_pca_like_mode_fits_predicts_and_encodes(factory, X):
    np.random.seed(17)
    model = factory()
    model.fit(
        X[:8],
        epochs=1,
        batch_size=4,
        patience=1,
        verbose=0,
        validation_data=(X[8:], None),
    )
    reconstruction = model.predict(X[8:], batch_size=4, verbose=0)
    latent = model.encode(X[8:], batch_size=4, verbose=0)
    assert reconstruction.shape == X[8:].shape
    assert latent.shape == (len(X[8:]), model.k)
    assert np.isfinite(reconstruction).all()
    assert np.isfinite(latent).all()


@pytest.mark.parametrize(("factory", "X"), _cases())
def test_default_none_mode_remains_available(factory, X):
    configured = factory()
    default_model = type(configured)(k=configured.k, device="cpu")
    assert default_model.latent_structure == "none"


def test_invalid_mode_rejected_early():
    with pytest.raises(ValueError):
        StandardAutoencoder(k=2, latent_structure="not-a-mode")

def test_legacy_positional_device_calls_remain_valid():
    """New latent options must not consume the historical device position."""
    models = [
        StandardAutoencoder(3, [8], "cpu"),
        LSTMAutoencoder(3, (8, 4), "cpu"),
        CNNAutoencoder(3, "cpu"),
        VisionTransformerAutoencoder(3, 4, 16, 1, 1, 4, "cpu"),
        ConvLSTMAutoencoder(3, "cpu"),
        HybridConvLSTMTransformerAutoencoder(
            3,
            16,
            4,
            1,
            "linear",
            "cpu",
        ),
        SpatialTokenConvLSTMTransformerAutoencoder(
            3,
            (2, 2),
            16,
            4,
            1,
            "cpu",
        ),
        VariationalAutoencoder(3, [8], 1.0, 1, "cpu"),
    ]

    for model in models:
        assert model.device.type == "cpu"
        assert model.latent_structure == "none"


def test_new_latent_constructor_options_are_keyword_only():
    """All public latent options should be keyword-only compatibility additions."""
    import inspect

    classes = (
        StandardAutoencoder,
        LSTMAutoencoder,
        CNNAutoencoder,
        VisionTransformerAutoencoder,
        ConvLSTMAutoencoder,
        HybridConvLSTMTransformerAutoencoder,
        SpatialTokenConvLSTMTransformerAutoencoder,
        VariationalAutoencoder,
    )
    latent_names = (
        "latent_structure",
        "latent_orthogonality_weight",
        "latent_decorrelation_weight",
        "latent_ordering_probability",
    )

    for cls in classes:
        parameters = inspect.signature(cls.__init__).parameters
        assert parameters["device"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for name in latent_names:
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_vae_direct_training_forward_applies_ordered_prefix_mask():
    """The ordinary VAE model forward must match custom-fit ordering semantics."""
    import torch

    outer = VariationalAutoencoder(
        k=3,
        hidden_dims=[8],
        beta=0.0,
        validation_mc_samples=1,
        latent_structure="pca_like",
        latent_orthogonality_weight=0.0,
        latent_decorrelation_weight=0.0,
        latent_ordering_probability=1.0,
        device="cpu",
    )
    inner = outer._build_model((4, 6))
    inner.train()

    captured = {}
    original_decode = inner.decode_forward

    def capture_decode(z):
        captured["z"] = z.detach().clone()
        return original_decode(z)

    inner.decode_forward = capture_decode
    x = torch.randn(4, 6)

    torch.manual_seed(123)
    _ = inner(x)

    assert "z" in captured
    assert torch.all(captured["z"][:, -1] == 0)

    mu, _ = inner.encode_distribution_forward(x)
    encoded = inner.encode_forward(x)
    assert torch.equal(mu, encoded)
    assert torch.any(mu[:, -1] != 0)


def test_built_model_rejects_latent_checkpoint_config_mismatch(tmp_path):
    """A built receiver must not silently retain a conflicting latent mode."""
    source = StandardAutoencoder(
        k=2,
        hidden_dims=[4],
        latent_structure="pca_like",
        latent_orthogonality_weight=1.0,
        latent_decorrelation_weight=0.1,
        latent_ordering_probability=0.5,
        device="cpu",
    )
    build_shape = (4, 6)
    source._build_input_shape = build_shape
    source.model = source._build_model(build_shape).to(source.device)
    source.is_fitted = True

    checkpoint = tmp_path / "structured.pt"
    source.save_pytorch_model(checkpoint)

    receiver = StandardAutoencoder(
        k=2,
        hidden_dims=[4],
        latent_structure="none",
        device="cpu",
    )
    receiver._build_input_shape = build_shape
    receiver.model = receiver._build_model(build_shape).to(receiver.device)

    with pytest.raises(ValueError, match="latent configuration"):
        receiver.load_pytorch_model(checkpoint, weights_only=False)


def test_structured_checkpoint_restores_config_when_rebuilt(tmp_path):
    """Self-describing loading must restore the saved latent configuration."""
    source = StandardAutoencoder(
        k=2,
        hidden_dims=[4],
        latent_structure="pca_like",
        latent_orthogonality_weight=1.0,
        latent_decorrelation_weight=0.1,
        latent_ordering_probability=0.5,
        device="cpu",
    )
    build_shape = (4, 6)
    source._build_input_shape = build_shape
    source.model = source._build_model(build_shape).to(source.device)
    source.is_fitted = True

    checkpoint = tmp_path / "structured-roundtrip.pt"
    source.save_pytorch_model(checkpoint)

    loaded = StandardAutoencoder.from_pytorch_model(
        checkpoint,
        device="cpu",
        weights_only=False,
    )

    assert loaded.latent_structure == "pca_like"
    assert loaded.latent_orthogonality_weight == pytest.approx(1.0)
    assert loaded.latent_decorrelation_weight == pytest.approx(0.1)
    assert loaded.latent_ordering_probability == pytest.approx(0.5)
