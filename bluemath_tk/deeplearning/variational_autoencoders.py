"""Variational autoencoders for BlueMath_tk."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ._base_model import BaseDeepLearningModel


class VariationalAutoencoder(BaseDeepLearningModel):
    """Dense variational autoencoder for arbitrary per-sample shapes.

    The model flattens every input sample internally, learns a diagonal
    Gaussian posterior in a latent space of size ``k``, and restores the
    original per-sample shape during decoding.

    Deterministic public inference is intentional: :meth:`encode` returns the
    posterior mean and :meth:`predict` reconstructs from that mean unless
    ``stochastic=True`` is requested explicitly.

    Parameters
    ----------
    k : int, optional
        Number of latent dimensions, by default 20.
    hidden_dims : list of int, optional
        Encoder hidden dimensions. The decoder uses the reversed sequence.
        By default ``[512, 256, 128]``.
    beta : float, optional
        Weight applied to the KL-divergence term, by default 1.0.
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

        self.k = k
        self.hidden_dims = list(hidden_dims)
        self.beta = float(beta)
        super().__init__(device=device, **kwargs)

    def _build_model(self, input_shape: tuple, **kwargs) -> nn.Module:
        """Build the encoder, Gaussian latent distribution, and decoder."""
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
                self.log_var_layer = nn.Linear(previous_dim, latent_dim)

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
                    raise ValueError(
                        "Input must include a leading batch dimension."
                    )
                sample_shape_actual = tuple(x.shape[1:])
                if sample_shape_actual != self.sample_shape:
                    raise ValueError(
                        f"Expected per-sample shape {self.sample_shape}, "
                        f"got {sample_shape_actual}."
                    )
                return x.reshape(x.size(0), self.n_features)

            def encode_distribution_forward(
                self,
                x: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                flat = self._flatten(x)
                hidden = self.encoder(flat)
                mu = self.mu_layer(hidden)
                log_var = self.log_var_layer(hidden)
                log_var = torch.clamp(log_var, min=-30.0, max=20.0)
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
                per_sample = -0.5 * torch.sum(
                    1.0 + log_var - mu.pow(2) - log_var.exp(),
                    dim=1,
                )
                return per_sample.mean()

            def decode_forward(self, z: torch.Tensor) -> torch.Tensor:
                if z.dim() != 2 or z.shape[1] != self.latent_dim:
                    raise ValueError(
                        "Latent input must have shape "
                        f"(batch, {self.latent_dim})."
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
        **kwargs,
    ) -> dict[str, list]:
        """Fit the VAE using a beta-weighted normalized reconstruction loss.

        The default reconstruction term is elementwise mean squared error,
        while the KL term is summed over latent dimensions and averaged over
        the batch. Consequently, ``beta`` controls their relative scale and
        should be selected using validation data for each data normalization
        and sample dimensionality.
        """
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
        )
        if tuple(X.shape) != tuple(y.shape):
            raise ValueError(
                "VariationalAutoencoder targets must have the same shape as X."
            )
        self._validate_or_set_build_input_shape(tuple(X.shape))

        if self.model is None:
            self.model = self._build_model(X.shape, **kwargs).to(self.device)

        if optimizer is None:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
            )
        if criterion is None:
            criterion = nn.MSELoss()

        indices = np.arange(len(X))
        np.random.shuffle(indices)
        split = int((1 - validation_split) * len(X))
        train_indices = indices[:split]
        validation_indices = indices[split:]

        X_train = torch.as_tensor(
            X[train_indices],
            dtype=torch.float32,
            device=self.device,
        )
        y_train = torch.as_tensor(
            y[train_indices],
            dtype=torch.float32,
            device=self.device,
        )
        X_validation = torch.as_tensor(
            X[validation_indices],
            dtype=torch.float32,
            device=self.device,
        )
        y_validation = torch.as_tensor(
            y[validation_indices],
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
                stochastic=True,
            )
            history["train_loss"].append(train_totals["loss"])
            history["train_reconstruction_loss"].append(
                train_totals["reconstruction_loss"]
            )
            history["train_kl_loss"].append(train_totals["kl_loss"])

            self.model.eval()
            with torch.no_grad():
                validation_totals = self._run_vae_epoch(
                    X_validation,
                    y_validation,
                    batch_size,
                    criterion,
                    optimizer=None,
                    stochastic=False,
                )
            history["val_loss"].append(validation_totals["loss"])
            history["val_reconstruction_loss"].append(
                validation_totals["reconstruction_loss"]
            )
            history["val_kl_loss"].append(validation_totals["kl_loss"])

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

        if best_model_state is not None:
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
        stochastic: bool,
    ) -> dict[str, float]:
        if self.model is None:
            raise ValueError("Model must be built before training.")

        totals = {
            "loss": 0.0,
            "reconstruction_loss": 0.0,
            "kl_loss": 0.0,
        }
        slices = self._batch_slices(len(X), batch_size)

        for start, stop in slices:
            batch_X = X[start:stop]
            batch_y = y[start:stop]
            if optimizer is not None:
                optimizer.zero_grad()

            mu, log_var = self.model.encode_distribution_forward(batch_X)
            z = self.model.reparameterize(mu, log_var) if stochastic else mu
            reconstruction = self.model.decode_forward(z)
            reconstruction_loss = criterion(reconstruction, batch_y)
            self._require_scalar_loss(reconstruction_loss)
            kl_loss = self.model.kl_divergence(mu, log_var)
            loss = reconstruction_loss + self.beta * kl_loss

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            totals["loss"] += float(loss.item())
            totals["reconstruction_loss"] += float(
                reconstruction_loss.item()
            )
            totals["kl_loss"] += float(kl_loss.item())

        number_of_batches = len(slices)
        return {
            name: value / number_of_batches
            for name, value in totals.items()
        }

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
        self._validate_inference_inputs(X, batch_size)

        self.model.eval()
        X_tensor = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=self.device,
        )
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                outputs.append(
                    self.model(
                        X_tensor[start : start + batch_size],
                        stochastic=True,
                    )
                    .cpu()
                    .numpy()
                )
        return np.concatenate(outputs, axis=0)

    def encode_distribution(
        self,
        X: np.ndarray,
        batch_size: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior means and log variances."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before encoding.")
        self._validate_inference_inputs(X, batch_size)

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
        self._validate_inference_inputs(X, batch_size)

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
                samples.append(
                    self.model.reparameterize(mu, log_var).cpu().numpy()
                )
        return np.concatenate(samples, axis=0)

    def sample(
        self,
        n_samples: int,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Decode samples drawn from the standard-normal latent prior."""
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
                outputs.append(self.model.decode_forward(z).cpu().numpy())
        return np.concatenate(outputs, axis=0)
