"""Tests for the dense variational autoencoder."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from bluemath_tk.deeplearning.autoencoders import VariationalAutoencoder


@pytest.fixture(autouse=True)
def _set_seed():
    previous_threads = torch.get_num_threads()
    np.random.seed(401)
    torch.manual_seed(401)
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


def _fit_model(X, beta=0.1, epochs=2, validation_mc_samples=3):
    model = VariationalAutoencoder(
        k=3,
        hidden_dims=[12, 8],
        beta=beta,
        validation_mc_samples=validation_mc_samples,
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
        "val_deterministic_reconstruction_loss",
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
    prediction_1 = model.predict(X, verbose=0, stochastic=True)
    prediction_2 = model.predict(X, verbose=0, stochastic=True)

    assert latent_1.shape == (16, 3)
    assert not np.array_equal(latent_1, latent_2)
    assert not np.array_equal(prediction_1, prediction_2)


def test_vae_kl_has_known_analytic_values():
    inner = VariationalAutoencoder(
        k=3,
        hidden_dims=[4],
        device="cpu",
    )._build_model((4, 5))

    zero_mu = torch.zeros(4, 3)
    zero_log_var = torch.zeros(4, 3)
    assert inner.kl_divergence(zero_mu, zero_log_var).item() == pytest.approx(0.0)

    unit_mu = torch.ones(4, 3)
    expected = 0.5 * 3
    assert inner.kl_divergence(unit_mu, zero_log_var).item() == pytest.approx(expected)


def test_vae_kl_is_stable_at_float32_mean_boundary():
    outer = VariationalAutoencoder(k=1, hidden_dims=[2], device="cpu")
    inner = outer._build_model((2, 1))
    maximum = np.finfo(np.float32).max
    mu = torch.tensor([[maximum]], dtype=torch.float32, requires_grad=True)
    log_var = torch.zeros_like(mu)

    loss = inner.kl_divergence(mu, log_var)
    loss.backward()

    assert loss.dtype == torch.float64
    assert torch.isfinite(loss)
    assert mu.grad is not None
    assert torch.isfinite(mu.grad).all()
    assert torch.count_nonzero(mu.grad) > 0


def test_vae_reparameterization_matches_standard_normal_moments():
    inner = VariationalAutoencoder(
        k=1,
        hidden_dims=[4],
        device="cpu",
    )._build_model((4, 2))
    mu = torch.full((20000, 1), 2.0)
    log_var = torch.log(torch.full((20000, 1), 9.0))

    sample = inner.reparameterize(mu, log_var)

    assert sample.mean().item() == pytest.approx(2.0, abs=0.08)
    assert sample.std(unbiased=False).item() == pytest.approx(3.0, abs=0.08)


def test_vae_full_objective_gradients_reach_all_paths():
    outer = VariationalAutoencoder(
        k=2,
        hidden_dims=[6],
        beta=0.2,
        device="cpu",
    )
    inner = outer._build_model((8, 4))
    X = torch.randn(8, 4)

    mu, log_var = inner.encode_distribution_forward(X)
    reconstruction = inner.decode_forward(inner.reparameterize(mu, log_var))
    loss = nn.functional.mse_loss(reconstruction, X)
    loss = loss + outer.beta * inner.kl_divergence(mu, log_var)
    loss.backward()

    for parameter in (
        inner.mu_layer.weight,
        inner.variance_layer.weight,
        inner.decoder[-1].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


@pytest.mark.parametrize(
    "raw_bias",
    [-1000.0, 1000.0, np.finfo(np.float32).max],
)
def test_vae_extreme_variance_logits_keep_corrective_gradients(raw_bias):
    outer = VariationalAutoencoder(
        k=2,
        hidden_dims=[4],
        beta=1.0,
        device="cpu",
    )
    inner = outer._build_model((6, 3))
    with torch.no_grad():
        inner.variance_layer.weight.zero_()
        inner.variance_layer.bias.fill_(raw_bias)

    X = torch.zeros(6, 3)
    mu, log_var = inner.encode_distribution_forward(X)
    loss = inner.kl_divergence(mu, log_var)
    loss.backward()

    gradient = inner.variance_layer.bias.grad
    assert torch.isfinite(log_var).all()
    assert torch.isfinite(loss)
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_vae_validation_reports_stochastic_and_deterministic_metrics():
    outer = VariationalAutoencoder(
        k=1,
        hidden_dims=[2],
        beta=0.0,
        validation_mc_samples=64,
        device="cpu",
    )

    class KnownPosterior(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))

        def encode_distribution_forward(self, x):
            mu = torch.zeros(len(x), 1) + self.anchor * 0
            log_var = torch.zeros_like(mu)
            return mu, log_var

        @staticmethod
        def reparameterize(mu, log_var):
            return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

        @staticmethod
        def kl_divergence(mu, log_var):
            return (
                0.5
                * torch.sum(
                    mu.pow(2) + log_var.exp() - 1.0 - log_var,
                    dim=1,
                ).mean()
            )

        @staticmethod
        def decode_forward(z):
            return torch.relu(z)

    outer.model = KnownPosterior()
    X = torch.zeros(32, 1)
    totals = outer._run_vae_epoch(
        X,
        X,
        batch_size=8,
        criterion=nn.MSELoss(),
        optimizer=None,
        stochastic_samples=64,
        report_deterministic=True,
    )

    assert totals["deterministic_reconstruction_loss"] == pytest.approx(0.0)
    assert totals["reconstruction_loss"] > 0.2
    assert totals["loss"] == pytest.approx(totals["reconstruction_loss"])


def test_vae_early_stopping_uses_stochastic_validation_objective(monkeypatch):
    X = np.zeros((8, 3), dtype="float32")
    model = VariationalAutoencoder(
        k=2,
        hidden_dims=[4],
        beta=0.1,
        validation_mc_samples=5,
        device="cpu",
    )
    validation_losses = iter([3.0, 1.0, 2.0])
    training_epoch = {"value": 0}

    def fake_epoch(
        X_tensor,
        y_tensor,
        batch_size,
        criterion,
        optimizer,
        stochastic_samples,
        report_deterministic,
    ):
        if optimizer is not None:
            training_epoch["value"] += 1
            with torch.no_grad():
                model.model.mu_layer.bias.fill_(training_epoch["value"])
            return {
                "loss": 1.0,
                "reconstruction_loss": 1.0,
                "kl_loss": 0.0,
                "deterministic_reconstruction_loss": 0.0,
            }

        assert stochastic_samples == model.validation_mc_samples
        assert report_deterministic
        validation_loss = next(validation_losses)
        return {
            "loss": validation_loss,
            "reconstruction_loss": validation_loss,
            "kl_loss": 0.0,
            "deterministic_reconstruction_loss": 100.0 - validation_loss,
        }

    monkeypatch.setattr(model, "_run_vae_epoch", fake_epoch)
    model.fit(
        X,
        validation_split=0.25,
        epochs=3,
        batch_size=4,
        patience=3,
        verbose=0,
    )

    expected = torch.full_like(model.model.mu_layer.bias, 2.0)
    assert torch.allclose(model.model.mu_layer.bias, expected)


def test_vae_validation_sampling_does_not_advance_training_rng(monkeypatch):
    X = np.zeros((8, 3), dtype="float32")
    model = VariationalAutoencoder(
        k=2,
        hidden_dims=[4],
        validation_mc_samples=7,
        device="cpu",
    )
    model.model = model._build_model(X.shape).to(model.device)
    optimizer = torch.optim.SGD(model.model.parameters(), lr=0.0)
    training_draws = []

    def fake_epoch(
        X_tensor,
        y_tensor,
        batch_size,
        criterion,
        optimizer,
        stochastic_samples,
        report_deterministic,
    ):
        if optimizer is not None:
            training_draws.append(torch.rand(()).item())
        else:
            torch.rand(128)
        return {
            "loss": 1.0,
            "reconstruction_loss": 1.0,
            "kl_loss": 0.0,
            "deterministic_reconstruction_loss": 1.0,
        }

    monkeypatch.setattr(model, "_run_vae_epoch", fake_epoch)
    torch.manual_seed(811)
    model.fit(
        X,
        validation_split=0.25,
        epochs=3,
        batch_size=4,
        optimizer=optimizer,
        patience=3,
        verbose=0,
    )

    torch.manual_seed(811)
    expected_draws = [torch.rand(()).item() for _ in range(3)]
    assert training_draws == pytest.approx(expected_draws)


def test_vae_beta_zero_removes_kl_from_total_loss():
    X = np.random.randn(16, 6).astype("float32")
    _, history = _fit_model(X, beta=0.0, epochs=1)

    assert history["train_loss"] == pytest.approx(history["train_reconstruction_loss"])
    assert history["val_loss"] == pytest.approx(history["val_reconstruction_loss"])


def test_vae_custom_inference_rejects_float32_overflow():
    X = np.zeros((16, 6), dtype="float32")
    model, _ = _fit_model(X)
    huge = np.full((2, 6), 1e100, dtype="float64")

    for method in (
        model.encode_distribution,
        model.sample_latent,
    ):
        with pytest.raises(ValueError, match="converted to float32"):
            method(huge)
    with pytest.raises(ValueError, match="converted to float32"):
        model.predict(huge, verbose=0, stochastic=True)


def test_vae_custom_inference_rejects_nonfinite_model_outputs(
    monkeypatch,
):
    X = np.zeros((16, 6), dtype="float32")
    model, _ = _fit_model(X)

    def nonfinite_distribution(batch):
        mu = torch.full(
            (len(batch), model.k),
            float("inf"),
            dtype=batch.dtype,
            device=batch.device,
        )
        return mu, torch.zeros_like(mu)

    monkeypatch.setattr(
        model.model,
        "encode_distribution_forward",
        nonfinite_distribution,
    )
    for method in (
        model.encode_distribution,
        model.sample_latent,
    ):
        with pytest.raises(FloatingPointError, match="not finite"):
            method(X[:2])
    with pytest.raises(FloatingPointError, match="not finite"):
        model.predict(X[:2], verbose=0, stochastic=True)

    sample_shape = tuple(X.shape[1:])

    def nonfinite_decode(z):
        return torch.full(
            (len(z), *sample_shape),
            float("inf"),
            dtype=z.dtype,
            device=z.device,
        )

    monkeypatch.setattr(model.model, "decode_forward", nonfinite_decode)
    with pytest.raises(FloatingPointError, match="not finite"):
        model.sample(2)


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
    assert restored.validation_mc_samples == model.validation_mc_samples
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
        ({"validation_mc_samples": 0}, "validation_mc_samples"),
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


def test_vae_training_reduces_structured_reconstruction_loss():
    coordinate = np.linspace(-1.0, 1.0, 32, dtype="float32")[:, None]
    X = np.concatenate(
        [coordinate, coordinate**2, np.sin(np.pi * coordinate)],
        axis=1,
    )
    _, history = _fit_model(X, beta=0.0, epochs=12)

    assert (
        history["train_reconstruction_loss"][-1]
        < (history["train_reconstruction_loss"][0])
    )
