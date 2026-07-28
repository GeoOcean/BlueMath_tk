"""Tests for the dense variational autoencoder."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bluemath_tk.deeplearning.autoencoders import (  # noqa: E402
    VariationalAutoencoder,
)


@pytest.fixture(autouse=True)
def _set_seed():
    np.random.seed(401)
    torch.manual_seed(401)
    torch.set_num_threads(1)


def _fit_model(X, beta=0.1, epochs=2):
    model = VariationalAutoencoder(
        k=3,
        hidden_dims=[12, 8],
        beta=beta,
        device="cpu",
    )
    history = model.fit(
        X,
        validation_split=0.25,
        epochs=epochs,
        batch_size=4,
        learning_rate=1e-3,
        patience=max(epochs, 2),
        verbose=0,
    )
    return model, history


def test_vae_reconstructs_original_multidimensional_shape():
    X = np.random.randn(16, 2, 5).astype("float32")
    model, history = _fit_model(X)

    prediction = model.predict(X, batch_size=4, verbose=0)
    latent = model.encode(X, batch_size=4, verbose=0)
    decoded = model.decode(latent, batch_size=4, verbose=0)

    assert prediction.shape == X.shape
    assert decoded.shape == X.shape
    assert latent.shape == (16, 3)
    assert np.allclose(prediction, decoded, rtol=1e-5, atol=1e-6)
    assert set(history) == {
        "train_loss",
        "train_reconstruction_loss",
        "train_kl_loss",
        "val_loss",
        "val_reconstruction_loss",
        "val_kl_loss",
    }


def test_vae_distribution_and_deterministic_inference():
    X = np.random.randn(16, 7).astype("float32")
    model, _ = _fit_model(X)

    mu, log_var = model.encode_distribution(X, batch_size=4)
    first = model.predict(X, batch_size=4, verbose=0)
    second = model.predict(X, batch_size=4, verbose=0)

    assert mu.shape == (16, 3)
    assert log_var.shape == (16, 3)
    assert np.allclose(model.encode(X, verbose=0), mu)
    assert np.array_equal(first, second)
    assert np.isfinite(mu).all()
    assert np.isfinite(log_var).all()


def test_vae_stochastic_operations_are_explicit():
    X = np.random.randn(16, 7).astype("float32")
    model, _ = _fit_model(X)

    latent_1 = model.sample_latent(X, batch_size=4)
    latent_2 = model.sample_latent(X, batch_size=4)
    prediction_1 = model.predict(X, batch_size=4, verbose=0, stochastic=True)
    prediction_2 = model.predict(X, batch_size=4, verbose=0, stochastic=True)

    assert latent_1.shape == (16, 3)
    assert not np.array_equal(latent_1, latent_2)
    assert not np.array_equal(prediction_1, prediction_2)


def test_vae_kl_divergence_matches_manual_calculation():
    X = np.random.randn(8, 5).astype("float32")
    model, _ = _fit_model(X)
    X_tensor = torch.as_tensor(X)

    mu, log_var = model.model.encode_distribution_forward(X_tensor)
    calculated = model.model.kl_divergence(mu, log_var)
    manual = -0.5 * torch.sum(
        1.0 + log_var - mu.pow(2) - log_var.exp(),
        dim=1,
    ).mean()

    assert torch.allclose(calculated, manual)
    assert calculated.ndim == 0
    assert calculated >= 0


def test_vae_reparameterization_keeps_gradients():
    mu = torch.randn(4, 3, requires_grad=True)
    log_var = torch.randn(4, 3, requires_grad=True)

    sample = VariationalAutoencoder(
        k=3,
        hidden_dims=[4],
        device="cpu",
    )._build_model((4, 5)).reparameterize(mu, log_var)
    sample.square().mean().backward()

    assert mu.grad is not None
    assert log_var.grad is not None
    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(log_var.grad).all()


def test_vae_beta_zero_removes_kl_from_total_loss():
    X = np.random.randn(16, 6).astype("float32")
    _, history = _fit_model(X, beta=0.0, epochs=1)

    assert history["train_loss"] == pytest.approx(
        history["train_reconstruction_loss"]
    )
    assert history["val_loss"] == pytest.approx(
        history["val_reconstruction_loss"]
    )


def test_vae_sampling_and_single_vector_decoding():
    X = np.random.randn(16, 6).astype("float32")
    model, _ = _fit_model(X)
    latent = model.encode(X, verbose=0)

    decoded = model.decode(latent[0], verbose=0)
    generated = model.sample(5, batch_size=2)

    assert decoded.shape == (1, 6)
    assert generated.shape == (5, 6)
    assert np.isfinite(generated).all()


def test_vae_checkpoint_round_trip(tmp_path):
    X = np.random.randn(16, 6).astype("float32")
    model, _ = _fit_model(X)
    checkpoint = tmp_path / "vae.pt"

    original = model.predict(X, verbose=0)
    model.save_pytorch_model(checkpoint)
    restored = VariationalAutoencoder.from_pytorch_model(
        checkpoint,
        device="cpu",
    )

    assert restored.beta == model.beta
    assert restored.hidden_dims == model.hidden_dims
    assert np.allclose(original, restored.predict(X, verbose=0))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k": 0}, "k"),
        ({"hidden_dims": []}, "hidden_dims"),
        ({"hidden_dims": [8, 0]}, "hidden dimension"),
        ({"beta": -1.0}, "beta"),
        ({"beta": float("nan")}, "beta"),
        ({"beta": float("inf")}, "beta"),
    ],
)
def test_vae_rejects_invalid_constructor_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        VariationalAutoencoder(**kwargs)


def test_vae_rejects_invalid_sampling_requests():
    model = VariationalAutoencoder(k=2, hidden_dims=[4], device="cpu")

    with pytest.raises(ValueError, match="fitted"):
        model.sample(1)
    with pytest.raises(TypeError, match="integer"):
        model.sample(1.5)
    with pytest.raises(TypeError, match="batch_size"):
        model.sample(1, batch_size=1.5)


def test_vae_rejects_zero_sized_sample_dimensions():
    X = np.empty((8, 0), dtype="float32")
    model = VariationalAutoencoder(
        k=2,
        hidden_dims=[4],
        device="cpu",
    )

    with pytest.raises(ValueError, match="dimension"):
        model.fit(
            X,
            validation_split=0.25,
            epochs=1,
            batch_size=2,
            patience=2,
            verbose=0,
        )


def test_vae_training_reduces_structured_reconstruction_loss():
    coordinate = np.linspace(-1.0, 1.0, 32, dtype="float32")[:, None]
    X = np.concatenate(
        [coordinate, coordinate**2, np.sin(np.pi * coordinate)],
        axis=1,
    )
    _, history = _fit_model(X, beta=0.0, epochs=12)

    assert history["train_reconstruction_loss"][-1] < (
        history["train_reconstruction_loss"][0]
    )
