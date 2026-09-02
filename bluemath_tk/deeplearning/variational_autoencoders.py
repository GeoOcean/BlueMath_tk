"""Variational autoencoders for BlueMath_tk."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from tqdm import tqdm

from ._base_model import BaseDeepLearningModel


class VariationalAutoencoder(BaseDeepLearningModel):
    """Dense variational autoencoder for arbitrary per-sample shapes.

    The encoder parameterizes a diagonal Gaussian posterior. Public
    :meth:`encode` and deterministic :meth:`predict` use the posterior mean.
    Stochastic posterior sampling is explicit through ``stochastic=True`` or
    :meth:`sample_latent`.

    The default objective is an elementwise mean reconstruction loss plus
    ``beta`` times a KL term summed over latent dimensions and averaged over
    samples. Therefore, ``beta`` depends on data normalization, per-sample
    dimensionality, reconstruction-loss scaling, and latent dimension.

    Parameters
    ----------
    k : int, optional
        Number of latent dimensions, by default 20.
    hidden_dims : list of int, optional
        Encoder hidden dimensions. The decoder uses the reversed sequence.
        By default ``[512, 256, 128]``.
    beta : float, optional
        Weight applied to the KL-divergence term, by default 1.0.
    validation_mc_samples : int, optional
        Posterior samples per validation batch for the stochastic objective
        used by early stopping, by default 4.
    device : str or torch.device, optional
        Device on which to run the model.
    **kwargs
        Additional keyword arguments passed to ``BaseDeepLearningModel``.
    """

    def __init__(
        self,
        k: int = 20,
        hidden_dims: list[int] | None = None,
        beta: float = 1.0,
        validation_mc_samples: int = 4,
        device: str | torch.device | None = None,
        **kwargs,
    ):
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer.")
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        if not isinstance(hidden_dims, list) or not hidden_dims:
            raise ValueError("hidden_dims must be a non-empty list of integers.")
        if any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 1
            for dim in hidden_dims
        ):
            raise ValueError("Every hidden dimension must be a positive integer.")
        if (
            not isinstance(beta, (int, float))
            or isinstance(beta, bool)
            or not math.isfinite(float(beta))
            or beta < 0
        ):
            raise ValueError("beta must be a finite non-negative number.")
        if (
            not isinstance(validation_mc_samples, int)
            or isinstance(validation_mc_samples, bool)
            or validation_mc_samples < 1
        ):
            raise ValueError("validation_mc_samples must be a positive integer.")

        self.k = k
        self.hidden_dims = list(hidden_dims)
        self.beta = float(beta)
        self.validation_mc_samples = validation_mc_samples
        super().__init__(device=device, **kwargs)

    def _build_model(self, input_shape: tuple, **kwargs) -> nn.Module:
        """Build the encoder, posterior parameterization, and decoder."""
        if len(input_shape) < 2:
            raise ValueError(
                "VariationalAutoencoder requires a leading sample dimension."
            )

        sample_shape = tuple(input_shape[1:])
        if any(dimension < 1 for dimension in sample_shape):
            raise ValueError("Every per-sample dimension must be positive.")
        n_features = int(np.prod(sample_shape))
        hidden_dims = tuple(self.hidden_dims)
        latent_dim = self.k

        class VariationalAutoencoderModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.sample_shape = sample_shape
                self.n_features = n_features
                self.latent_dim = latent_dim

                encoder_layers: list[nn.Module] = []
                previous_dim = n_features
                for hidden_dim in hidden_dims:
                    encoder_layers.extend(
                        [
                            nn.Linear(previous_dim, hidden_dim),
                            nn.ReLU(),
                        ]
                    )
                    previous_dim = hidden_dim
                self.encoder = nn.Sequential(*encoder_layers)
                self.mu_layer = nn.Linear(previous_dim, latent_dim)
                self.variance_layer = nn.Linear(previous_dim, latent_dim)

                decoder_layers: list[nn.Module] = []
                previous_dim = latent_dim
                for hidden_dim in reversed(hidden_dims):
                    decoder_layers.extend(
                        [
                            nn.Linear(previous_dim, hidden_dim),
                            nn.ReLU(),
                        ]
                    )
                    previous_dim = hidden_dim
                decoder_layers.append(nn.Linear(previous_dim, n_features))
                self.decoder = nn.Sequential(*decoder_layers)

            def _flatten(self, x: torch.Tensor) -> torch.Tensor:
                if x.dim() < 2:
                    raise ValueError("Input must include a leading batch dimension.")
                actual = tuple(x.shape[1:])
                if actual != self.sample_shape:
                    raise ValueError(
                        f"Expected per-sample shape {self.sample_shape}, got {actual}."
                    )
                return x.reshape(x.size(0), self.n_features)

            def encode_distribution_forward(
                self,
                x: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                flat = self._flatten(x)
                hidden = self.encoder(flat)
                mu = self.mu_layer(hidden)
                raw_variance = self.variance_layer(hidden)
                safe_raw = torch.clamp_min(raw_variance, -20.0)
                central_log_var = torch.log(functional.softplus(safe_raw))
                log_var = torch.where(
                    raw_variance < -20.0,
                    raw_variance,
                    central_log_var,
                )
                return mu, log_var

            @staticmethod
            def reparameterize(
                mu: torch.Tensor,
                log_var: torch.Tensor,
            ) -> torch.Tensor:
                standard_deviation = torch.exp(0.5 * log_var)
                noise = torch.randn_like(standard_deviation)
                return mu + standard_deviation * noise

            @staticmethod
            def kl_divergence(
                mu: torch.Tensor,
                log_var: torch.Tensor,
            ) -> torch.Tensor:
                """Return a numerically stable diagonal-Gaussian KL mean."""
                mu_stable = mu.to(dtype=torch.float64)
                log_var_stable = log_var.to(dtype=torch.float64)
                per_sample = 0.5 * torch.sum(
                    mu_stable.pow(2) + log_var_stable.exp() - 1.0 - log_var_stable,
                    dim=1,
                )
                return per_sample.mean()

            def decode_forward(self, z: torch.Tensor) -> torch.Tensor:
                if z.dim() != 2 or z.shape[1] != self.latent_dim:
                    raise ValueError(
                        f"Latent input must have shape (batch, {self.latent_dim})."
                    )
                reconstruction = self.decoder(z)
                return reconstruction.reshape(
                    reconstruction.size(0),
                    *self.sample_shape,
                )

            def encode_forward(self, x: torch.Tensor) -> torch.Tensor:
                mu, _ = self.encode_distribution_forward(x)
                return mu

            def forward(
                self,
                x: torch.Tensor,
                stochastic: bool | None = None,
            ) -> torch.Tensor:
                mu, log_var = self.encode_distribution_forward(x)
                if stochastic is None:
                    stochastic = self.training
                z = self.reparameterize(mu, log_var) if stochastic else mu
                return self.decode_forward(z)

        return VariationalAutoencoderModel()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        validation_split: float = 0.2,
        epochs: int = 500,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        optimizer: torch.optim.Optimizer | None = None,
        criterion: nn.Module | None = None,
        patience: int = 20,
        verbose: int = 1,
        validation_data: tuple[np.ndarray, np.ndarray | None] | None = None,
        **kwargs,
    ) -> dict[str, list]:
        """Fit the VAE with stochastic train and validation objectives.

        ``val_loss`` is a Monte Carlo estimate of the same beta-VAE objective
        used for training and controls early stopping. The separate
        ``val_deterministic_reconstruction_loss`` reports posterior-mean
        reconstruction for stable scientific comparison.

        Parameters
        ----------
        validation_data : tuple, optional
            An explicit ``(X_validation, y_validation)`` pair. When supplied,
            ``validation_split`` is ignored and exactly these samples drive the
            validation objective and early stopping. Default is None.
        """
        learning_rate = self._validate_learning_rate(learning_rate)
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        if y is None:
            y = self._get_reconstruction_target(X)

        self._validate_fit_inputs(
            X,
            y,
            validation_split,
            batch_size,
            epochs,
            patience,
            validation_data=validation_data,
        )
        (
            X_train_array,
            y_train_array,
            X_validation_array,
            y_validation_array,
        ) = self._resolve_fit_partitions(X, y, validation_split, validation_data)
        self._validate_or_set_build_input_shape(tuple(X.shape))
        self.is_fitted = False

        if self.model is None:
            self.model = self._build_model(X.shape, **kwargs).to(self.device)

        if optimizer is None:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
            )
        if criterion is None:
            criterion = nn.MSELoss()
        reduction = getattr(criterion, "reduction", "mean")
        if reduction not in {"mean", None}:
            raise ValueError(
                "VariationalAutoencoder requires a mean-reduced scalar "
                "reconstruction criterion."
            )

        X_train = torch.as_tensor(
            X_train_array,
            dtype=torch.float32,
            device=self.device,
        )
        y_train = torch.as_tensor(
            y_train_array,
            dtype=torch.float32,
            device=self.device,
        )
        X_validation = torch.as_tensor(
            X_validation_array,
            dtype=torch.float32,
            device=self.device,
        )
        y_validation = torch.as_tensor(
            y_validation_array,
            dtype=torch.float32,
            device=self.device,
        )

        history = {
            "train_loss": [],
            "train_reconstruction_loss": [],
            "train_kl_loss": [],
            "val_loss": [],
            "val_reconstruction_loss": [],
            "val_kl_loss": [],
            "val_deterministic_reconstruction_loss": [],
        }
        best_validation_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        epoch_range = range(epochs)
        progress_bar = None
        if verbose > 0:
            progress_bar = tqdm(epoch_range, desc="Training", unit="epoch")
            epoch_range = progress_bar

        for epoch in epoch_range:
            self.model.train()
            train_totals = self._run_vae_epoch(
                X_train,
                y_train,
                batch_size,
                criterion,
                optimizer=optimizer,
                stochastic_samples=1,
                report_deterministic=False,
            )
            history["train_loss"].append(train_totals["loss"])
            history["train_reconstruction_loss"].append(
                train_totals["reconstruction_loss"]
            )
            history["train_kl_loss"].append(train_totals["kl_loss"])

            self.model.eval()
            validation_devices: list[int] = []
            if self.device.type == "cuda":
                device_index = self.device.index
                if device_index is None:
                    device_index = torch.cuda.current_device()
                validation_devices = [device_index]
            with torch.random.fork_rng(devices=validation_devices):
                with torch.no_grad():
                    validation_totals = self._run_vae_epoch(
                        X_validation,
                        y_validation,
                        batch_size,
                        criterion,
                        optimizer=None,
                        stochastic_samples=self.validation_mc_samples,
                        report_deterministic=True,
                    )
            history["val_loss"].append(validation_totals["loss"])
            history["val_reconstruction_loss"].append(
                validation_totals["reconstruction_loss"]
            )
            history["val_kl_loss"].append(validation_totals["kl_loss"])
            history["val_deterministic_reconstruction_loss"].append(
                validation_totals["deterministic_reconstruction_loss"]
            )

            validation_loss = validation_totals["loss"]
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if progress_bar is not None:
                        progress_bar.set_postfix_str(
                            f"Early stopping at epoch {epoch + 1}"
                        )
                    break

            if progress_bar is not None:
                progress_bar.set_postfix_str(
                    f"Train: {train_totals['loss']:.6f}, "
                    f"Val: {validation_loss:.6f}, "
                    f"Patience: {patience_counter}/{patience}"
                )

        if best_model_state is None:
            raise FloatingPointError(
                "Training completed without a finite validation objective."
            )
        self.model.load_state_dict(best_model_state)
        self.is_fitted = True
        return history

    def _run_vae_epoch(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        stochastic_samples: int,
        report_deterministic: bool,
    ) -> dict[str, float]:
        if self.model is None:
            raise ValueError("Model must be built before training.")

        totals = {
            "loss": 0.0,
            "reconstruction_loss": 0.0,
            "kl_loss": 0.0,
            "deterministic_reconstruction_loss": 0.0,
        }
        total_samples = 0

        for start, stop in self._batch_slices(len(X), batch_size):
            batch_X = X[start:stop]
            batch_y = y[start:stop]
            current_batch_size = stop - start
            if optimizer is not None:
                optimizer.zero_grad()

            mu, log_var = self.model.encode_distribution_forward(batch_X)
            self._require_finite_tensor(mu, "VAE posterior mean")
            self._require_finite_tensor(log_var, "VAE posterior log variance")
            reconstruction_losses = []
            for _ in range(stochastic_samples):
                z = self.model.reparameterize(mu, log_var)
                reconstruction = self.model.decode_forward(z)
                self._require_matching_output_shape(
                    reconstruction, batch_y, "VAE reconstruction"
                )
                self._require_finite_tensor(reconstruction, "VAE reconstruction output")
                reconstruction_loss = criterion(reconstruction, batch_y)
                self._require_scalar_loss(reconstruction_loss)
                self._require_finite_loss(
                    reconstruction_loss,
                    "VAE reconstruction",
                )
                reconstruction_losses.append(reconstruction_loss)

            mean_reconstruction_loss = torch.stack(reconstruction_losses).mean()
            kl_loss = self.model.kl_divergence(mu, log_var)
            self._require_finite_loss(kl_loss, "VAE KL")
            loss = mean_reconstruction_loss + self.beta * kl_loss
            self._require_finite_loss(loss, "VAE total")

            deterministic_loss = None
            if report_deterministic:
                deterministic = self.model.decode_forward(mu)
                self._require_matching_output_shape(
                    deterministic,
                    batch_y,
                    "VAE deterministic reconstruction",
                )
                self._require_finite_tensor(
                    deterministic,
                    "VAE deterministic reconstruction output",
                )
                deterministic_loss = criterion(deterministic, batch_y)
                self._require_scalar_loss(deterministic_loss)
                self._require_finite_loss(
                    deterministic_loss,
                    "VAE deterministic reconstruction",
                )

            if optimizer is None:
                self._require_finite_parameters()
            else:
                self._require_finite_buffers()

            if optimizer is not None:
                loss.backward()
                self._require_finite_gradients()
                optimizer.step()
                self._require_finite_parameters()

            totals["loss"] += float(loss.item()) * current_batch_size
            totals["reconstruction_loss"] += (
                float(mean_reconstruction_loss.item()) * current_batch_size
            )
            totals["kl_loss"] += float(kl_loss.item()) * current_batch_size
            if deterministic_loss is not None:
                totals["deterministic_reconstruction_loss"] += (
                    float(deterministic_loss.item()) * current_batch_size
                )
            total_samples += current_batch_size

        return {name: value / total_samples for name, value in totals.items()}

    def predict(
        self,
        X: np.ndarray,
        batch_size: int = 64,
        verbose: int = 1,
        stochastic: bool = False,
    ) -> np.ndarray:
        """Reconstruct inputs from posterior means or posterior samples."""
        if not stochastic:
            return super().predict(
                X,
                batch_size=batch_size,
                verbose=verbose,
            )
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction.")
        self._validate_inference_inputs(
            X,
            batch_size,
            check_expected_shape=True,
        )

        self.model.eval()
        X_tensor = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=self.device,
        )
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                output = self.model(
                    X_tensor[start : start + batch_size],
                    stochastic=True,
                )
                self._require_finite_tensor(output, "Stochastic prediction output")
                self._require_finite_parameters()
                outputs.append(output.cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def encode_distribution(
        self,
        X: np.ndarray,
        batch_size: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior means and log variances."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before encoding.")
        self._validate_inference_inputs(
            X,
            batch_size,
            check_expected_shape=True,
        )

        self.model.eval()
        X_tensor = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=self.device,
        )
        means = []
        log_variances = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                mu, log_var = self.model.encode_distribution_forward(
                    X_tensor[start : start + batch_size]
                )
                self._require_finite_tensor(mu, "Posterior mean output")
                self._require_finite_tensor(log_var, "Posterior log-variance output")
                self._require_finite_parameters()
                means.append(mu.cpu().numpy())
                log_variances.append(log_var.cpu().numpy())
        return (
            np.concatenate(means, axis=0),
            np.concatenate(log_variances, axis=0),
        )

    def sample_latent(
        self,
        X: np.ndarray,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Draw one posterior latent sample for every input sample."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before sampling latents.")
        self._validate_inference_inputs(
            X,
            batch_size,
            check_expected_shape=True,
        )

        self.model.eval()
        X_tensor = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=self.device,
        )
        samples = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                mu, log_var = self.model.encode_distribution_forward(
                    X_tensor[start : start + batch_size]
                )
                self._require_finite_tensor(mu, "Posterior mean output")
                self._require_finite_tensor(log_var, "Posterior log-variance output")
                sample = self.model.reparameterize(mu, log_var)
                self._require_finite_tensor(sample, "Posterior latent sample")
                self._require_finite_parameters()
                samples.append(sample.cpu().numpy())
        return np.concatenate(samples, axis=0)

    def sample(
        self,
        n_samples: int,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Decode standard-normal prior samples.

        Prior samples are generatively meaningful only when KL regularization
        has aligned the learned posterior with the standard-normal prior.
        """
        if not isinstance(n_samples, int) or isinstance(n_samples, bool):
            raise TypeError("n_samples must be an integer.")
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1.")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before sampling.")

        outputs = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                current_batch = min(batch_size, n_samples - start)
                z = torch.randn(
                    current_batch,
                    self.k,
                    dtype=torch.float32,
                    device=self.device,
                )
                output = self.model.decode_forward(z)
                self._require_finite_tensor(output, "Prior sample output")
                self._require_finite_parameters()
                outputs.append(output.cpu().numpy())
        return np.concatenate(outputs, axis=0)
