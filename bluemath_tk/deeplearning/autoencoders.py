"""
Autoencoders module.

This module is a pytorch translation from a tensorflow implementation developed by Sergio López Dubón.

This module contains the following autoencoders:
- StandardAutoencoder
- OrthogonalAutoencoder
- VariationalAutoencoder
- LSTMAutoencoder
- CNNAutoencoder
- VisionTransformerAutoencoder
- ConvLSTMAutoencoder
- HybridConvLSTMTransformerAutoencoder

Each autoencoder is a subclass of BaseDeepLearningModel and implements the following methods:
- fit(X, y=None, epochs=10, batch_size=32, verbose=1)
- predict(X)
- encode(X)
- decode(X)
- evaluate(X)
"""

import copy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ._base_model import BaseDeepLearningModel
from .layers import (
    LatentDecorr,
    LinearSelfAttention,
    Patchify,
    PositionalEmbedding,
    TimePositionalEncoding,
    Unpatchify,
)
from .variational_autoencoders import VariationalAutoencoder



class StandardAutoencoder(BaseDeepLearningModel):
    """
    Standard fully-connected autoencoder.

    A simple feedforward autoencoder with symmetric encoder-decoder architecture.
    Designed for tabular/flattened data (not images or sequences).

    Input Shape
    -----------
    X : np.ndarray
        Input data with a leading sample dimension.
        For tabular data use (n_samples, n_features). Higher-dimensional
        per-sample inputs are flattened internally and reconstructed to
        their original sample shape.

    Examples
    --------
    >>> # Tabular data (e.g., flattened features)
    >>> X = np.random.randn(1000, 784)  # 1000 samples, 784 features
    >>> ae = StandardAutoencoder(k=20, hidden_dims=[256, 128, 64])
    >>> history = ae.fit(X, epochs=10)
    >>> X_recon = ae.predict(X)
    >>> Z = ae.encode(X)  # Get latent representations (1000, 20)

    Parameters
    ----------
    k : int, optional
        Number of latent dimensions. Default is 20.
    hidden_dims : list, optional
        List of hidden layer dimensions for encoder (decoder is symmetric).
        Default is [512, 256, 128, 64].
    device : str or torch.device, optional
        Device to run the model on. Default is None.
    **kwargs
        Additional keyword arguments passed to BaseDeepLearningModel.
    """

    def __init__(
        self,
        k: int = 20,
        hidden_dims: Optional[list] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        self.hidden_dims = hidden_dims
        self.k = k
        super().__init__(device=device, **kwargs)

    def _build_model(self, input_shape: Tuple, **kwargs) -> nn.Module:
        """Build the standard fully-connected autoencoder model."""
        # Handle input shape: (n_samples, ...) or a single sample shape.
        # For fit(X), input_shape includes the batch dimension, so the
        # per-sample shape is input_shape[1:].
        if len(input_shape) == 1:
            sample_shape = (input_shape[0],)
        else:
            sample_shape = tuple(input_shape[1:])
        n_features = int(np.prod(sample_shape))

        class StandardAutoencoderModel(nn.Module):
            def __init__(self, n_features, hidden_dims, k, sample_shape):
                super().__init__()
                self.n_features = n_features
                self.sample_shape = tuple(sample_shape)
                # Encoder
                encoder_layers = []
                prev_dim = n_features
                for dim in hidden_dims:
                    encoder_layers.append(nn.Linear(prev_dim, dim))
                    encoder_layers.append(nn.BatchNorm1d(dim))
                    encoder_layers.append(nn.ReLU())
                    prev_dim = dim
                encoder_layers.append(nn.Linear(prev_dim, k))
                self.encoder = nn.Sequential(*encoder_layers)

                # Decoder
                decoder_layers = []
                prev_dim = k
                for dim in reversed(hidden_dims):
                    decoder_layers.append(nn.Linear(prev_dim, dim))
                    decoder_layers.append(nn.BatchNorm1d(dim))
                    decoder_layers.append(nn.ReLU())
                    prev_dim = dim
                decoder_layers.append(nn.Linear(prev_dim, n_features))
                self.decoder = nn.Sequential(*decoder_layers)

            def forward(self, x):
                # Flatten input if needed: (B, ...) -> (B, n_features)
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                elif x.dim() == 1:
                    x = x.unsqueeze(0)
                z = self.encoder(x)
                x_recon = self.decoder(z)
                return x_recon.view(x_recon.size(0), *self.sample_shape)

            def encode_forward(self, x):
                """Encode input to latent space."""
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                elif x.dim() == 1:
                    x = x.unsqueeze(0)
                return self.encoder(x)


            def decode_forward(self, z):
                """Decode latent vectors to the original sample shape."""
                x_recon = self.decoder(z)
                return x_recon.view(x_recon.size(0), *self.sample_shape)

        return StandardAutoencoderModel(n_features, self.hidden_dims, self.k, sample_shape)


class OrthogonalAutoencoder(BaseDeepLearningModel):
    """
    Orthogonal autoencoder with orthogonal regularization.

    Adds orthogonality constraints on encoder weights and latent decorrelation
    to encourage more interpretable latent representations.
    Designed for tabular/flattened data (not images or sequences).

    Input Shape
    -----------
    X : np.ndarray
        Input data with a leading sample dimension.
        For tabular data use (n_samples, n_features). Higher-dimensional
        per-sample inputs are flattened internally and reconstructed to
        their original sample shape.

    Examples
    --------
    >>> # Tabular data with orthogonal constraints
    >>> X = np.random.randn(1000, 784)  # 1000 samples, 784 features
    >>> ae = OrthogonalAutoencoder(k=20, lambda_W=1e-3, lambda_Z=1e-2)
    >>> history = ae.fit(X, epochs=10)
    >>> Z = ae.encode(X)  # Decorrelated latent representations

    Parameters
    ----------
    k : int, optional
        Number of latent dimensions. Default is 20.
    hidden_dims : list, optional
        List of hidden layer dimensions. Default is [512, 256, 128, 64].
    lambda_W : float, optional
        Weight orthogonality penalty strength. Default is 1e-3.
    lambda_Z : float, optional
        Latent decorrelation penalty strength. Default is 1e-2.
    device : str or torch.device, optional
        Device to run the model on. Default is None.
    **kwargs
        Additional keyword arguments passed to BaseDeepLearningModel.
    """

    def __init__(
        self,
        k: int = 20,
        hidden_dims: Optional[list] = None,
        lambda_W: float = 1e-3,
        lambda_Z: float = 1e-2,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        self.hidden_dims = hidden_dims
        self.k = k
        self.lambda_W = lambda_W
        self.lambda_Z = lambda_Z
        super().__init__(device=device, **kwargs)

    def _build_model(self, input_shape: Tuple, **kwargs) -> nn.Module:
        """Build the orthogonal autoencoder model."""
        # Handle input shape: (n_samples, ...) or a single sample shape.
        # For fit(X), input_shape includes the batch dimension, so the
        # per-sample shape is input_shape[1:].
        if len(input_shape) == 1:
            sample_shape = (input_shape[0],)
        else:
            sample_shape = tuple(input_shape[1:])
        n_features = int(np.prod(sample_shape))

        class OrthogonalAutoencoderModel(nn.Module):
            def __init__(
                self, n_features, hidden_dims, k, lambda_W, lambda_Z, sample_shape
            ):
                super().__init__()
                self.n_features = n_features
                self.sample_shape = tuple(sample_shape)
                self.lambda_W = lambda_W
                self.lambda_Z = lambda_Z

                # Encoder
                encoder_layers = []
                prev_dim = n_features
                for dim in hidden_dims:
                    encoder_layers.append(nn.Linear(prev_dim, dim))
                    encoder_layers.append(nn.BatchNorm1d(dim))
                    encoder_layers.append(nn.ReLU())
                    prev_dim = dim

                self.encoder_layers = nn.ModuleList(encoder_layers)
                self.latent_layer = nn.Linear(prev_dim, k)
                self.latent_decorr = LatentDecorr(strength=lambda_Z)

                # Decoder
                decoder_layers = []
                prev_dim = k
                for dim in reversed(hidden_dims):
                    decoder_layers.append(nn.Linear(prev_dim, dim))
                    decoder_layers.append(nn.BatchNorm1d(dim))
                    decoder_layers.append(nn.ReLU())
                    prev_dim = dim
                decoder_layers.append(nn.Linear(prev_dim, n_features))
                self.decoder = nn.Sequential(*decoder_layers)

            def forward(self, x):
                # Flatten input if needed: (B, ...) -> (B, n_features)
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                elif x.dim() == 1:
                    x = x.unsqueeze(0)
                h = x
                for layer in self.encoder_layers:
                    h = layer(h)
                z = self.latent_layer(h)
                z = self.latent_decorr(z)

                # Orthogonality regularization
                W = self.latent_layer.weight  # (k, in_dim)
                WT_W = torch.matmul(W, W.t())  # (k, k)
                I_k = torch.eye(WT_W.size(0), device=WT_W.device, dtype=WT_W.dtype)
                ortho_loss = self.lambda_W * torch.sum((WT_W - I_k) ** 2)

                # Store losses for retrieval during training
                # Keep in computation graph by adding to z (doesn't change z value)
                self._ortho_loss = ortho_loss
                z = z + 0 * ortho_loss

                x_recon = self.decoder(z)
                return x_recon.view(x_recon.size(0), *self.sample_shape)

            def encode_forward(self, x):
                """Encode input to latent space."""
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                elif x.dim() == 1:
                    x = x.unsqueeze(0)
                h = x
                for layer in self.encoder_layers:
                    h = layer(h)
                z = self.latent_layer(h)
                z = self.latent_decorr(z)
                return z

            def get_regularization_losses(self):
                """Get current regularization losses."""
                ortho_loss = getattr(self, "_ortho_loss", None)
                decorr_loss = getattr(self.latent_decorr, "_loss", None)
                return ortho_loss, decorr_loss


            def decode_forward(self, z):
                """Decode latent vectors to the original sample shape."""
                x_recon = self.decoder(z)
                return x_recon.view(x_recon.size(0), *self.sample_shape)

        return OrthogonalAutoencoderModel(
            n_features,
            self.hidden_dims,
            self.k,
            self.lambda_W,
            self.lambda_Z,
            sample_shape,
        )

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        validation_split: float = 0.2,
        epochs: int = 500,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        patience: int = 20,
        verbose: int = 1,
        **kwargs,
    ) -> Dict[str, list]:
        """
        Fit the orthogonal autoencoder with regularization losses.

        This method overrides the base fit() to properly add orthogonality
        and decorrelation regularization losses during training.
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
        self._validate_or_set_build_input_shape(tuple(X.shape))

        if self.model is None:
            self.model = self._build_model(X.shape, **kwargs)
            self.model = self.model.to(self.device)

        avoid_singleton = self._requires_non_singleton_training_batches()

        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        if criterion is None:
            criterion = nn.MSELoss()

        # Train/validation split
        n_samples = len(X)
        idx = np.arange(n_samples)
        np.random.shuffle(idx)
        split = int((1 - validation_split) * n_samples)
        train_idx, val_idx = idx[:split], idx[split:]
        Xtr, Xval = X[train_idx], X[val_idx]

        if y is None:
            # Autoencoder case
            ytr, yval = Xtr, Xval
        else:
            ytr, yval = y[train_idx], y[val_idx]

        # Convert to tensors
        Xtr_tensor = torch.FloatTensor(Xtr).to(self.device)
        Xval_tensor = torch.FloatTensor(Xval).to(self.device)
        ytr_tensor = torch.FloatTensor(ytr).to(self.device)
        yval_tensor = torch.FloatTensor(yval).to(self.device)

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        # Create progress bar if verbose > 0
        use_progress_bar = verbose > 0
        epoch_range = range(epochs)
        pbar = None
        if use_progress_bar:
            pbar = tqdm(epoch_range, desc="Training", unit="epoch")
            epoch_range = pbar

        for epoch in epoch_range:
            # Training
            self.model.train()
            train_loss = 0.0
            train_slices = self._batch_slices(
                len(Xtr),
                batch_size,
                avoid_singleton=avoid_singleton,
            )
            n_batches = len(train_slices)
            for start, stop in train_slices:
                batch_X = Xtr_tensor[start:stop]
                batch_y = ytr_tensor[start:stop]

                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)

                # Add regularization losses
                ortho_loss, decorr_loss = self.model.get_regularization_losses()
                if ortho_loss is not None:
                    loss = loss + ortho_loss
                if decorr_loss is not None:
                    loss = loss + decorr_loss

                self._require_scalar_loss(loss)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= n_batches
            history["train_loss"].append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                val_slices = self._batch_slices(
                    len(Xval),
                    batch_size,
                    avoid_singleton=False,
                )
                n_val_batches = len(val_slices)
                for start, stop in val_slices:
                    batch_X = Xval_tensor[start:stop]
                    batch_y = yval_tensor[start:stop]

                    output = self.model(batch_X)
                    loss = criterion(output, batch_y)

                    # Add regularization losses for validation
                    ortho_loss, decorr_loss = self.model.get_regularization_losses()
                    if ortho_loss is not None:
                        loss = loss + ortho_loss
                    if decorr_loss is not None:
                        loss = loss + decorr_loss

                    self._require_scalar_loss(loss)
                    val_loss += loss.item()

                val_loss /= n_val_batches
                history["val_loss"].append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose > 0:
                        if pbar is not None:
                            pbar.set_postfix_str(f"Early stopping at epoch {epoch + 1}")
                        self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            # Update progress bar with current losses
            if pbar is not None:
                pbar.set_postfix_str(
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Patience: {patience_counter}/{patience}"
                )
            elif verbose > 0 and (epoch + 1) % max(1, epochs // 10) == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.is_fitted = True

        return history


class LSTMAutoencoder(BaseDeepLearningModel):
    """
    LSTM-based autoencoder for sequential/temporal data.

    Uses LSTM cells for encoding and decoding temporal sequences.
    Designed for time series data (not images or tabular data).

    Input Shape
    -----------
    X : np.ndarray
        Input data with shape (n_samples, seq_len, n_features).
        - n_samples: number of sequences
        - seq_len: length of each sequence (automatically inferred from X.shape[1])
        - n_features: number of features per timestep

    Examples
    --------
    >>> # Time series data (e.g., sensor readings over time)
    >>> X = np.random.randn(100, 10, 5)  # 100 sequences, 10 timesteps, 5 features
    >>> ae = LSTMAutoencoder(k=20, hidden=(256, 128))
    >>> history = ae.fit(X, epochs=10)
    >>> X_recon = ae.predict(X)  # Shape: (100, 10, 5)
    >>> Z = ae.encode(X)  # Latent representations: (100, 20)

    Parameters
    ----------
    k : int, optional
        Number of latent dimensions, by default 20.
    hidden : tuple, optional
        Hidden layer dimensions for LSTM, by default (256, 128).
    device : str or torch.device, optional
        Device to run the model on.
    **kwargs
        Additional keyword arguments passed to BaseDeepLearningModel.
    """

    def __init__(
        self,
        k: int = 20,
        hidden: Tuple[int, int] = (256, 128),
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        self.hidden = hidden
        self.k = k
        super().__init__(device=device, **kwargs)

    def _build_model(self, input_shape: Tuple, **kwargs) -> nn.Module:
        """Build the LSTM autoencoder model."""
        # Input shape should be (n_samples, seq_len, n_features)
        if len(input_shape) != 3:
            raise ValueError(
                f"LSTMAutoencoder expects 3D input (n_samples, seq_len, n_features), "
                f"got shape {input_shape}"
            )
        n_features = input_shape[-1]
        seq_len = input_shape[1]  # Infer from input shape

        class LSTMAutoencoderModel(nn.Module):
            def __init__(self, seq_len, n_features, hidden, k):
                super().__init__()
                self.seq_len = seq_len
                self.n_features = n_features

                # Encoder
                self.lstm1 = nn.LSTM(n_features, hidden[0], batch_first=True)
                self.lstm2 = nn.LSTM(hidden[0], hidden[1], batch_first=True)
                self.latent = nn.Linear(hidden[1], k)

                # Decoder
                self.latent_to_seq = nn.Linear(k, hidden[1])
                self.lstm3 = nn.LSTM(hidden[1], hidden[0], batch_first=True)
                self.lstm4 = nn.LSTM(hidden[0], n_features, batch_first=True)

            def forward(self, x):
                # x: (B, T, F)
                if x.dim() != 3:
                    raise ValueError(
                        f"Expected 3D input (batch, seq_len, features), got {x.shape}"
                    )
                # Encoder
                x, _ = self.lstm1(x)
                x, _ = self.lstm2(x)
                z = self.latent(x[:, -1, :])  # Take last timestep

                # Decoder
                z_expanded = (
                    self.latent_to_seq(z).unsqueeze(1).repeat(1, self.seq_len, 1)
                )
                x, _ = self.lstm3(z_expanded)
                x, _ = self.lstm4(x)

                return x

            def encode_forward(self, x):
                """Encode input to latent space."""
                if x.dim() != 3:
                    raise ValueError(
                        f"Expected 3D input (batch, seq_len, features), got {x.shape}"
                    )
                x, _ = self.lstm1(x)
                x, _ = self.lstm2(x)
                z = self.latent(x[:, -1, :])  # Take last timestep
                return z


            def decode_forward(self, z):
                """Decode latent vectors to full temporal sequences."""
                z_expanded = (
                    self.latent_to_seq(z)
                    .unsqueeze(1)
                    .repeat(1, self.seq_len, 1)
                )
                x, _ = self.lstm3(z_expanded)
                x, _ = self.lstm4(x)
                return x

        return LSTMAutoencoderModel(seq_len, n_features, self.hidden, self.k)


class CNNAutoencoder(BaseDeepLearningModel):
    """
    Convolutional autoencoder for spatial grid data (images).

    Uses 2D convolutions for encoding and transposed convolutions for decoding.
    Designed for 2D spatial data like images or gridded data.

    Input Shape
    -----------
    X : np.ndarray
        Input data with shape (n_samples, C, H, W) - channels-first format.
        - n_samples: number of images
        - C: number of channels (e.g., 1 for grayscale, 3 for RGB)
        - H, W: height and width of the image
        Note: Only channels-first format is supported for consistency.

    Examples
    --------
    >>> # Single images (channels-first format required)
    >>> X = np.random.randn(100, 3, 64, 64)  # 100 images, 3 channels, 64x64
    >>> ae = CNNAutoencoder(k=20)
    >>> history = ae.fit(X, epochs=10)
    >>> X_recon = ae.predict(X)  # Shape: (100, 3, 64, 64)
    >>> Z = ae.encode(X)  # Latent representations: (100, 20)

    Parameters
    ----------
    k : int, optional
        Number of latent dimensions. Default is 20.
    device : str or torch.device, optional
        Device to run the model on. Default is None.
    **kwargs
        Additional keyword arguments passed to BaseDeepLearningModel.
    """

    def __init__(
        self,
        k: int = 20,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        self.k = k
        super().__init__(device=device, **kwargs)

    def _build_model(self, input_shape: Tuple, **kwargs) -> nn.Module:
        """Build the CNN autoencoder model."""
        # Parse input shape: (n_samples, C, H, W) or (C, H, W)
        if len(input_shape) == 4:
            # (n_samples, C, H, W) - channels-first format
            C, H, W = input_shape[1], input_shape[2], input_shape[3]
        elif len(input_shape) == 3:
            # (C, H, W) - single sample without batch dimension
            C, H, W = input_shape[0], input_shape[1], input_shape[2]
        else:
            raise ValueError(
                f"CNNAutoencoder expects 3D (C, H, W) or 4D (n_samples, C, H, W) input shape, "
                f"got {input_shape} with {len(input_shape)} dimensions"
            )

        # Pad to make H, W divisible by 4
        pad_h = (4 - (H % 4)) % 4
        pad_w = (4 - (W % 4)) % 4

        class CNNAutoencoderModel(nn.Module):
            def __init__(self, H, W, C, k, pad_h, pad_w):
                super().__init__()
                self.pad_h = pad_h
                self.pad_w = pad_w
                self.C = C
                self.H = H
                self.W = W

                # Encoder
                self.encoder = nn.Sequential(
                    nn.ZeroPad2d((0, pad_w, 0, pad_h)),
                    nn.Conv2d(C, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                )

                # Calculate flattened size
                H_enc = (H + pad_h) // 4
                W_enc = (W + pad_w) // 4
                self.flat_size = H_enc * W_enc * 64

                self.fc1 = nn.Linear(self.flat_size, 256)
                self.fc2 = nn.Linear(256, k)

                # Decoder
                self.fc3 = nn.Linear(k, 256)
                self.fc4 = nn.Linear(256, self.flat_size)

                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(
                        64, 64, 3, stride=2, padding=1, output_padding=1
                    ),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.ConvTranspose2d(
                        64, 32, 3, stride=2, padding=1, output_padding=1
                    ),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, C, 3, padding=1),
                )

            def forward(self, x):
                # Only accept (B, C, H, W) format - channels-first
                if x.dim() != 4:
                    raise ValueError(
                        f"CNNAutoencoder expects 4D input (B, C, H, W), got shape {x.shape}"
                    )

                # Validate channels are in the correct position
                if x.shape[1] != self.C:
                    raise ValueError(
                        f"CNNAutoencoder expects channels-first format (B, C, H, W). "
                        f"Expected C={self.C} at position 1, but got shape {x.shape}. "
                        f"If your data is channels-last (B, H, W, C), please permute it: "
                        f"X = np.transpose(X, (0, 3, 1, 2))"
                    )

                # Encoder
                x = self.encoder(x)
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                z = self.fc2(x)

                # Decoder
                x = F.relu(self.fc3(z))
                x = F.relu(self.fc4(x))
                x = x.view(
                    x.size(0),
                    64,
                    (self.H + self.pad_h) // 4,
                    (self.W + self.pad_w) // 4,
                )
                x = self.decoder(x)

                # Crop padding
                if self.pad_h > 0 or self.pad_w > 0:
                    x = x[:, :, : self.H, : self.W]

                return x

            def encode_forward(self, x):
                """Encode input to latent space."""

                # Only accept (B, C, H, W) format - channels-first
                if x.dim() != 4:
                    raise ValueError(
                        f"CNNAutoencoder expects 4D input (B, C, H, W), got shape {x.shape}"
                    )

                # Validate channels are in the correct position
                if x.shape[1] != self.C:
                    raise ValueError(
                        f"CNNAutoencoder expects channels-first format (B, C, H, W). "
                        f"Expected C={self.C} at position 1, but got shape {x.shape}. "
                        f"If your data is channels-last (B, H, W, C), please permute it: "
                        f"X = np.transpose(X, (0, 3, 1, 2))"
                    )

                # Encoder only
                x = self.encoder(x)
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                z = self.fc2(x)

                return z


            def decode_forward(self, z):
                """Decode latent vectors to channels-first spatial grids."""
                batch_size = z.size(0)
                x = F.relu(self.fc3(z))
                x = F.relu(self.fc4(x))
                x = x.view(
                    batch_size,
                    64,
                    (self.H + self.pad_h) // 4,
                    (self.W + self.pad_w) // 4,
                )
                x = self.decoder(x)
                if self.pad_h > 0 or self.pad_w > 0:
                    x = x[:, :, : self.H, : self.W]
                return x

        return CNNAutoencoderModel(H, W, C, self.k, pad_h, pad_w)


class VisionTransformerAutoencoder(BaseDeepLearningModel):
    """
    Vision Transformer (ViT) autoencoder for spatial grid data (images).

    Uses patch-based processing with transformer architecture.
    Designed for 2D spatial data like images or gridded data.

    Input Shape
    -----------
    X : np.ndarray
        Input data with shape (n_samples, C, H, W) - channels-first format.
        - n_samples: number of images
        - C: number of channels (e.g., 1 for grayscale, 3 for RGB)
        - H, W: height and width of the image
        Note: Only channels-first format is supported.

    Examples
    --------
    >>> # Single images (channels-first format required)
    >>> X = np.random.randn(100, 3, 64, 64)  # 100 images, 3 channels, 64x64
    >>> ae = VisionTransformerAutoencoder(k=20, patch_size=8, d_model=256)
    >>> history = ae.fit(X, epochs=10)
    >>> X_recon = ae.predict(X)  # Shape: (100, 3, 64, 64)
    >>> Z = ae.encode(X)  # Latent representations: (100, 20)

    Parameters
    ----------
    k : int, optional
        Number of latent dimensions, by default 20.
    patch_size : int, optional
        Size of each patch, by default 8.
    d_model : int, optional
        Model dimension, by default 256.
    depth_enc : int, optional
        Number of encoder transformer blocks, by default 4.
    depth_dec : int, optional
        Number of decoder transformer blocks, by default 2.
    heads : int, optional
        Number of attention heads, by default 4.
    device : str or torch.device, optional
        Device to run the model on.
    **kwargs
        Additional keyword arguments passed to BaseDeepLearningModel.
    """

    def __init__(
        self,
        k: int = 20,
        patch_size: int = 8,
        d_model: int = 256,
        depth_enc: int = 4,
        depth_dec: int = 2,
        heads: int = 4,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.d_model = d_model
        self.depth_enc = depth_enc
        self.depth_dec = depth_dec
        self.heads = heads
        self.k = k
        super().__init__(device=device, **kwargs)

    def _build_model(self, input_shape: Tuple, **kwargs) -> nn.Module:
        """Build the ViT autoencoder model."""
        # Parse input shape: (n_samples, C, H, W) or (C, H, W)
        if len(input_shape) == 4:
            # (n_samples, C, H, W) - channels-first format
            C, H, W = input_shape[1], input_shape[2], input_shape[3]
        elif len(input_shape) == 3:
            # (C, H, W) - single sample without batch dimension
            C, H, W = input_shape[0], input_shape[1], input_shape[2]
        else:
            raise ValueError(
                f"VisionTransformerAutoencoder expects 3D (C, H, W) or 4D (n_samples, C, H, W) input shape, "
                f"got {input_shape} with {len(input_shape)} dimensions"
            )

        # Pad to make H, W divisible by patch_size
        pad_h = (self.patch_size - (H % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (W % self.patch_size)) % self.patch_size
        Hp, Wp = (H + pad_h) // self.patch_size, (W + pad_w) // self.patch_size
        N = Hp * Wp
        Pdim = self.patch_size * self.patch_size * C

        class ViTAutoencoderModel(nn.Module):
            def __init__(
                self,
                H,
                W,
                C,
                patch_size,
                d_model,
                depth_enc,
                depth_dec,
                heads,
                k,
                pad_h,
                pad_w,
                N,
                Pdim,
            ):
                super().__init__()
                self.patch_size = patch_size
                self.pad_h = pad_h
                self.pad_w = pad_w
                self.H = H
                self.W = W
                self.C = C
                self.d_model = d_model

                # Patchify + embed + pos
                self.patchify = Patchify(patch_size)
                self.patch_embed = nn.Linear(Pdim, d_model)
                self.pos_embed = PositionalEmbedding(N, d_model)

                # Encoder blocks
                encoder_blocks = []
                for _ in range(depth_enc):
                    encoder_blocks.append(
                        nn.TransformerEncoderLayer(
                            d_model,
                            heads,
                            dim_feedforward=d_model * 4,
                            activation="gelu",
                            batch_first=True,
                        )
                    )
                self.encoder_blocks = nn.Sequential(*encoder_blocks)

                # Global bottleneck (latent k)
                self.global_pool = nn.AdaptiveAvgPool1d(1)
                self.latent_k = nn.Linear(d_model, k)

                # Project back to token space for decoding
                self.dec_seed = nn.Linear(k, N * d_model)
                self.dec_pos_embed = PositionalEmbedding(N, d_model)

                # Decoder blocks
                decoder_blocks = []
                for _ in range(depth_dec):
                    decoder_blocks.append(
                        nn.TransformerEncoderLayer(
                            d_model,
                            heads,
                            dim_feedforward=d_model * 4,
                            activation="gelu",
                            batch_first=True,
                        )
                    )
                self.decoder_blocks = nn.Sequential(*decoder_blocks)
                self.patch_reconstruct = nn.Linear(d_model, Pdim)

                # Reconstruct patches
                self.unpatchify = Unpatchify(patch_size, Hp, Wp, C)

            def forward(self, x):
                # Only accept (B, C, H, W) format - channels-first
                if x.dim() != 4:
                    raise ValueError(
                        f"VisionTransformerAutoencoder expects 4D input (B, C, H, W), "
                        f"got shape {x.shape}"
                    )

                # Validate channels are in the correct position
                if x.shape[1] != self.C:
                    raise ValueError(
                        f"VisionTransformerAutoencoder expects channels-first format (B, C, H, W). "
                        f"Expected C={self.C} at position 1, but got shape {x.shape}. "
                        f"If your data is channels-last (B, H, W, C), please permute it: "
                        f"X = np.transpose(X, (0, 3, 1, 2))"
                    )

                B = x.size(0)

                # Pad
                if self.pad_h > 0 or self.pad_w > 0:
                    x = F.pad(x, (0, self.pad_w, 0, self.pad_h))

                # Patchify + embed + pos
                tokens = self.patchify(x)  # (B, N, Pdim)
                tok_emb = self.patch_embed(tokens)  # (B, N, d_model)
                x = self.pos_embed(tok_emb)  # (B, N, d_model)

                # Encoder blocks
                for block in self.encoder_blocks:
                    x = block(x)

                # Global bottleneck
                z = x.mean(dim=1)  # (B, d_model) - GlobalAveragePooling1D
                z_k = self.latent_k(z)  # (B, k)

                # Project back to token space
                dec_seed = F.relu(self.dec_seed(z_k))  # (B, N*d_model)
                dec_tokens = dec_seed.view(B, N, self.d_model)  # (B, N, d_model)
                dec_tokens = self.dec_pos_embed(dec_tokens)

                # Decoder blocks
                y = dec_tokens
                for block in self.decoder_blocks:
                    y = block(y)

                # Reconstruct patches
                patch_tokens = self.patch_reconstruct(y)  # (B, N, Pdim)
                rec_patches = self.unpatchify(patch_tokens)  # (B, C, H+pad_h, W+pad_w)

                # Crop padding
                if self.pad_h > 0 or self.pad_w > 0:
                    rec_patches = rec_patches[:, :, : self.H, : self.W]

                return rec_patches

            def encode_forward(self, x):
                """Encode input to latent space."""
                # Only accept (B, C, H, W) format - channels-first
                if x.dim() != 4:
                    raise ValueError(
                        f"VisionTransformerAutoencoder expects 4D input (B, C, H, W), "
                        f"got shape {x.shape}"
                    )

                # Validate channels are in the correct position
                if x.shape[1] != self.C:
                    raise ValueError(
                        f"VisionTransformerAutoencoder expects channels-first format (B, C, H, W). "
                        f"Expected C={self.C} at position 1, but got shape {x.shape}. "
                        f"If your data is channels-last (B, H, W, C), please permute it: "
                        f"X = np.transpose(X, (0, 3, 1, 2))"
                    )

                # Pad
                if self.pad_h > 0 or self.pad_w > 0:
                    x = F.pad(x, (0, self.pad_w, 0, self.pad_h))

                # Patchify + embed + pos
                tokens = self.patchify(x)
                tok_emb = self.patch_embed(tokens)
                x = self.pos_embed(tok_emb)

                # Encoder blocks
                for block in self.encoder_blocks:
                    x = block(x)

                # Global bottleneck
                z = x.mean(dim=1)  # (B, d_model)
                z_k = self.latent_k(z)  # (B, k)

                return z_k


            def decode_forward(self, z):
                """Decode latent vectors to channels-first spatial grids."""
                batch_size = z.size(0)
                dec_seed = F.relu(self.dec_seed(z))
                dec_tokens = dec_seed.view(batch_size, N, self.d_model)
                dec_tokens = self.dec_pos_embed(dec_tokens)
                y = dec_tokens
                for block in self.decoder_blocks:
                    y = block(y)
                patch_tokens = self.patch_reconstruct(y)
                reconstruction = self.unpatchify(patch_tokens)
                if self.pad_h > 0 or self.pad_w > 0:
                    reconstruction = reconstruction[
                        :, :, : self.H, : self.W
                    ]
                return reconstruction

        return ViTAutoencoderModel(
            H,
            W,
            C,
            self.patch_size,
            self.d_model,
            self.depth_enc,
            self.depth_dec,
            self.heads,
            self.k,
            pad_h,
            pad_w,
            N,
            Pdim,
        )

class ConvLSTMAutoencoder(BaseDeepLearningModel):
    """ConvLSTM autoencoder for complete spatiotemporal sequences.

    The model compresses each input window to one latent vector and
    reconstructs the complete window. For input shape ``(B, T, C, H, W)``,
    ``predict`` and ``decode`` return ``(B, T, C, H, W)``.

    Parameters
    ----------
    k : int, optional
        Number of latent dimensions, by default 20.
    device : str or torch.device, optional
        Device on which to run the model.
    **kwargs
        Additional keyword arguments passed to ``BaseDeepLearningModel``.
    """

    def __init__(
        self,
        k: int = 20,
        device: str | torch.device | None = None,
        **kwargs,
    ):
        if "reconstruction_mode" in kwargs:
            raise TypeError(
                "reconstruction_mode is no longer supported; "
                "ConvLSTMAutoencoder always reconstructs the full sequence."
            )
        self.k = k
        super().__init__(device=device, **kwargs)

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
        """Fit the model to reconstruct the complete input sequence."""
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        target = self._get_reconstruction_target(X) if y is None else y
        self._validate_target_shape(X, target)
        return super().fit(
            X,
            y=target,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            criterion=criterion,
            patience=patience,
            verbose=verbose,
            **kwargs,
        )

    def _get_reconstruction_target(self, X: np.ndarray) -> np.ndarray:
        """Use the complete input sequence as the reconstruction target."""
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        if X.ndim != 5:
            raise ValueError(
                "ConvLSTMAutoencoder expects 5D input "
                "(n_samples, seq_len, C, H, W)."
            )
        return X

    def _validate_target_shape(
        self,
        X: np.ndarray,
        target: np.ndarray,
    ) -> None:
        if not isinstance(target, np.ndarray):
            raise TypeError("y must be a NumPy array.")
        expected = X.shape
        if target.shape != expected:
            raise ValueError(
                f"Target shape {target.shape} is incompatible with full "
                f"sequence reconstruction; expected {expected}."
            )

    def _build_model(self, input_shape: tuple, **kwargs) -> nn.Module:
        """Build the ConvLSTM encoder and full-sequence decoder."""
        if len(input_shape) != 5:
            raise ValueError(
                "ConvLSTMAutoencoder expects input shape "
                "(n_samples, seq_len, C, H, W)."
            )

        seq_len = input_shape[1]
        channels, height, width = input_shape[2:]
        pad_h = (-height) % 4
        pad_w = (-width) % 4
        latent_dim = self.k

        class ConvLSTMAutoencoderModel(nn.Module):
            def __init__(self):
                super().__init__()
                from .layers import ConvLSTM

                self.seq_len = seq_len
                self.H = height
                self.W = width
                self.C = channels
                self.pad_h = pad_h
                self.pad_w = pad_w

                self.convlstm1 = ConvLSTM(
                    input_dim=channels,
                    hidden_dim=32,
                    kernel_size=(3, 3),
                    num_layers=1,
                    batch_first=True,
                    return_all_layers=False,
                )
                self.bn1 = nn.BatchNorm3d(32)
                self.convlstm2 = ConvLSTM(
                    input_dim=32,
                    hidden_dim=32,
                    kernel_size=(3, 3),
                    num_layers=1,
                    batch_first=True,
                    return_all_layers=False,
                )

                self.conv1 = nn.Conv2d(32, 32, 3, padding=1)
                self.pool1 = nn.MaxPool2d(2)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool2 = nn.MaxPool2d(2)

                encoded_h = (height + pad_h) // 4
                encoded_w = (width + pad_w) // 4
                self.flat_size = encoded_h * encoded_w * 64
                self.latent = nn.Linear(self.flat_size, latent_dim)

                self.fc_dec = nn.Linear(latent_dim, self.flat_size)
                self.upsample1 = nn.Upsample(
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=True,
                )
                self.deconv1 = nn.ConvTranspose2d(
                    64,
                    64,
                    3,
                    padding=1,
                )
                self.upsample2 = nn.Upsample(
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=True,
                )
                self.deconv2 = nn.ConvTranspose2d(
                    64,
                    channels,
                    3,
                    padding=1,
                )

                self.decoder_time_embedding = nn.Parameter(
                    torch.zeros(1, seq_len, latent_dim)
                )
                nn.init.normal_(self.decoder_time_embedding, std=0.02)
                self.temporal_decoder = nn.LSTM(
                    input_size=latent_dim,
                    hidden_size=latent_dim,
                    batch_first=True,
                )

            def _validate_input(self, x: torch.Tensor) -> None:
                if x.dim() != 5:
                    raise ValueError(
                        "ConvLSTMAutoencoder expects 5D input "
                        "(B, T, C, H, W)."
                    )
                sample_shape = tuple(x.shape[1:])
                expected = (self.seq_len, self.C, self.H, self.W)
                if sample_shape != expected:
                    raise ValueError(
                        f"Expected per-sample shape {expected}, "
                        f"got {sample_shape}."
                    )

            def _encode(self, x: torch.Tensor) -> torch.Tensor:
                self._validate_input(x)
                batch_size = x.size(0)

                if self.pad_h > 0 or self.pad_w > 0:
                    x = F.pad(x, (0, self.pad_w, 0, self.pad_h))

                x_list, _ = self.convlstm1(x)
                x = x_list[0]
                x = x.permute(0, 2, 1, 3, 4)
                x = self.bn1(x)
                x = x.permute(0, 2, 1, 3, 4)
                x_list, _ = self.convlstm2(x)
                x = x_list[0][:, -1]

                x = F.relu(self.conv1(x))
                x = self.pool1(x)
                x = F.relu(self.conv2(x))
                x = self.pool2(x)
                return self.latent(x.reshape(batch_size, -1))

            def _decode_spatial(self, codes: torch.Tensor) -> torch.Tensor:
                frame_count = codes.size(0)
                x = F.relu(self.fc_dec(codes))
                x = x.view(
                    frame_count,
                    64,
                    (self.H + self.pad_h) // 4,
                    (self.W + self.pad_w) // 4,
                )
                x = self.upsample1(x)
                x = F.relu(self.deconv1(x))
                x = self.upsample2(x)
                x = self.deconv2(x)
                if self.pad_h > 0 or self.pad_w > 0:
                    x = x[:, :, : self.H, : self.W]
                return x

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Encode and reconstruct the complete input sequence."""
                return self.decode_forward(self.encode_forward(x))

            def encode_forward(self, x: torch.Tensor) -> torch.Tensor:
                """Encode a complete input sequence to one latent vector."""
                return self._encode(x)

            def decode_forward(self, z: torch.Tensor) -> torch.Tensor:
                """Decode latent vectors to complete sequences."""
                if z.dim() != 2 or z.size(1) != latent_dim:
                    raise ValueError(
                        f"Expected latent shape (B, {latent_dim}), "
                        f"got {tuple(z.shape)}."
                    )

                temporal_input = (
                    z.unsqueeze(1) + self.decoder_time_embedding
                )
                temporal_codes, _ = self.temporal_decoder(temporal_input)
                batch_size = z.size(0)
                frames = self._decode_spatial(
                    temporal_codes.reshape(batch_size * self.seq_len, -1)
                )
                return frames.view(
                    batch_size,
                    self.seq_len,
                    self.C,
                    self.H,
                    self.W,
                )

        return ConvLSTMAutoencoderModel()


class HybridConvLSTMTransformerAutoencoder(BaseDeepLearningModel):
    """ConvLSTM-Transformer autoencoder for complete sequences.

    The encoder combines ConvLSTM features with temporal attention and maps
    the complete input window to one latent vector. The decoder uses
    latent-conditioned temporal queries to reconstruct every input frame.

    Parameters
    ----------
    k : int, optional
        Number of latent dimensions, by default 20.
    d_model : int, optional
        Transformer embedding dimension, by default 256.
    n_heads : int, optional
        Number of attention heads, by default 4.
    n_layers : int, optional
        Number of Transformer layers, by default 2.
    efficient_attention : {"linear", None}, optional
        Attention implementation, by default ``"linear"``.
    device : str or torch.device, optional
        Device on which to run the model.
    **kwargs
        Additional keyword arguments passed to ``BaseDeepLearningModel``.
    """

    def __init__(
        self,
        k: int = 20,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        efficient_attention: str | None = "linear",
        device: str | torch.device | None = None,
        **kwargs,
    ):
        if "reconstruction_mode" in kwargs:
            raise TypeError(
                "reconstruction_mode is no longer supported; "
                "HybridConvLSTMTransformerAutoencoder always reconstructs "
                "the full sequence."
            )
        self.k = k
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.efficient_attention = efficient_attention
        super().__init__(device=device, **kwargs)

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
        """Fit the model to reconstruct the complete input sequence."""
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        target = self._get_reconstruction_target(X) if y is None else y
        self._validate_target_shape(X, target)
        return super().fit(
            X,
            y=target,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            criterion=criterion,
            patience=patience,
            verbose=verbose,
            **kwargs,
        )

    def _get_reconstruction_target(self, X: np.ndarray) -> np.ndarray:
        """Use the complete input sequence as the reconstruction target."""
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        if X.ndim != 5:
            raise ValueError(
                "HybridConvLSTMTransformerAutoencoder expects 5D input "
                "(n_samples, seq_len, C, H, W)."
            )
        return X

    def _validate_target_shape(
        self,
        X: np.ndarray,
        target: np.ndarray,
    ) -> None:
        if not isinstance(target, np.ndarray):
            raise TypeError("y must be a NumPy array.")
        expected = X.shape
        if target.shape != expected:
            raise ValueError(
                f"Target shape {target.shape} is incompatible with full "
                f"sequence reconstruction; expected {expected}."
            )

    def _build_model(self, input_shape: tuple, **kwargs) -> nn.Module:
        """Build the hybrid encoder and full-sequence decoder."""
        if len(input_shape) != 5:
            raise ValueError(
                "HybridConvLSTMTransformerAutoencoder expects input shape "
                "(n_samples, seq_len, C, H, W)."
            )

        seq_len = input_shape[1]
        channels, height, width = input_shape[2:]
        pad_h = (-height) % 4
        pad_w = (-width) % 4
        latent_dim = self.k
        d_model = self.d_model
        n_heads = self.n_heads
        n_layers = self.n_layers
        efficient_attention = self.efficient_attention

        class HybridAutoencoderModel(nn.Module):
            def __init__(self):
                super().__init__()
                from .layers import ConvLSTM

                self.seq_len = seq_len
                self.H = height
                self.W = width
                self.C = channels
                self.pad_h = pad_h
                self.pad_w = pad_w
                self.efficient_attention = efficient_attention

                self.convlstm1 = ConvLSTM(
                    input_dim=channels,
                    hidden_dim=32,
                    kernel_size=(3, 3),
                    num_layers=1,
                    batch_first=True,
                    return_all_layers=True,
                )
                self.convlstm2 = ConvLSTM(
                    input_dim=32,
                    hidden_dim=32,
                    kernel_size=(3, 3),
                    num_layers=1,
                    batch_first=True,
                    return_all_layers=True,
                )

                self.conv1 = nn.Conv2d(32, 32, 3, padding=1)
                self.pool1 = nn.MaxPool2d(2)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool2 = nn.MaxPool2d(2)
                self.global_pool = nn.AdaptiveAvgPool2d(1)
                self.frame_embed = nn.Linear(64, d_model)
                self.time_pos_enc = TimePositionalEncoding()
                self.transformer_blocks = self._make_blocks()

                self.global_pool_time = nn.AdaptiveAvgPool1d(1)
                self.latent = nn.Linear(d_model, latent_dim)

                encoded_h = (height + pad_h) // 4
                encoded_w = (width + pad_w) // 4
                self.flat_size = encoded_h * encoded_w * 64

                self.latent_to_decoder = nn.Linear(
                    latent_dim,
                    d_model,
                )
                self.decoder_time_queries = nn.Parameter(
                    torch.zeros(1, seq_len, d_model)
                )
                nn.init.normal_(self.decoder_time_queries, std=0.02)
                self.decoder_time_pos_enc = TimePositionalEncoding()
                self.decoder_transformer_blocks = self._make_blocks()
                self.frame_seed = nn.Linear(d_model, self.flat_size)

                self.upsample1 = nn.Upsample(
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=True,
                )
                self.deconv1 = nn.ConvTranspose2d(
                    64,
                    64,
                    3,
                    padding=1,
                )
                self.upsample2 = nn.Upsample(
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=True,
                )
                self.deconv2 = nn.ConvTranspose2d(
                    64,
                    channels,
                    3,
                    padding=1,
                )

            def _make_blocks(self) -> nn.ModuleList:
                if self.efficient_attention == "linear":
                    return nn.ModuleList(
                        [
                            nn.ModuleDict(
                                {
                                    "norm1": nn.LayerNorm(d_model),
                                    "attn": LinearSelfAttention(
                                        d_model,
                                        n_heads,
                                    ),
                                    "norm2": nn.LayerNorm(d_model),
                                    "mlp": nn.Sequential(
                                        nn.Linear(d_model, d_model * 4),
                                        nn.GELU(),
                                        nn.Dropout(0.0),
                                        nn.Linear(d_model * 4, d_model),
                                        nn.Dropout(0.0),
                                    ),
                                }
                            )
                            for _ in range(n_layers)
                        ]
                    )
                return nn.ModuleList(
                    [
                        nn.TransformerEncoderLayer(
                            d_model,
                            n_heads,
                            dim_feedforward=d_model * 4,
                            activation="gelu",
                            batch_first=True,
                        )
                        for _ in range(n_layers)
                    ]
                )

            def _run_blocks(
                self,
                x: torch.Tensor,
                blocks: nn.ModuleList,
            ) -> torch.Tensor:
                if self.efficient_attention == "linear":
                    for block in blocks:
                        x_norm = block["norm1"](x)
                        x = x + block["attn"](x_norm)
                        x = x + block["mlp"](block["norm2"](x))
                    return x
                for block in blocks:
                    x = block(x)
                return x

            def _validate_input(self, x: torch.Tensor) -> None:
                if x.dim() != 5:
                    raise ValueError(
                        "HybridConvLSTMTransformerAutoencoder expects 5D "
                        "input (B, T, C, H, W)."
                    )
                sample_shape = tuple(x.shape[1:])
                expected = (self.seq_len, self.C, self.H, self.W)
                if sample_shape != expected:
                    raise ValueError(
                        f"Expected per-sample shape {expected}, "
                        f"got {sample_shape}."
                    )

            def _encode(self, x: torch.Tensor) -> torch.Tensor:
                self._validate_input(x)
                batch_size, seq_size = x.shape[:2]

                if self.pad_h > 0 or self.pad_w > 0:
                    x = F.pad(x, (0, self.pad_w, 0, self.pad_h))

                x_list, _ = self.convlstm1(x)
                x = x_list[0]
                x_list, _ = self.convlstm2(x)
                x = x_list[0]

                frame_features = []
                for index in range(seq_size):
                    frame = F.relu(self.conv1(x[:, index]))
                    frame = self.pool1(frame)
                    frame = F.relu(self.conv2(frame))
                    frame = self.pool2(frame)
                    frame = self.global_pool(frame).flatten(1)
                    frame_features.append(self.frame_embed(frame))

                x = torch.stack(frame_features, dim=1)
                x = self.time_pos_enc(x)
                x = self._run_blocks(x, self.transformer_blocks)
                x = self.global_pool_time(x.transpose(1, 2)).squeeze(-1)
                return self.latent(x).view(batch_size, latent_dim)

            def _decode_spatial(
                self,
                codes: torch.Tensor,
                projection: nn.Linear,
            ) -> torch.Tensor:
                frame_count = codes.size(0)
                x = F.relu(projection(codes))
                x = x.view(
                    frame_count,
                    64,
                    (self.H + self.pad_h) // 4,
                    (self.W + self.pad_w) // 4,
                )
                x = self.upsample1(x)
                x = F.relu(self.deconv1(x))
                x = self.upsample2(x)
                x = self.deconv2(x)
                if self.pad_h > 0 or self.pad_w > 0:
                    x = x[:, :, : self.H, : self.W]
                return x

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Encode and reconstruct the complete input sequence."""
                return self.decode_forward(self.encode_forward(x))

            def encode_forward(self, x: torch.Tensor) -> torch.Tensor:
                """Encode a complete input sequence to one latent vector."""
                return self._encode(x)

            def decode_forward(self, z: torch.Tensor) -> torch.Tensor:
                """Decode latent vectors to complete sequences."""
                if z.dim() != 2 or z.size(1) != latent_dim:
                    raise ValueError(
                        f"Expected latent shape (B, {latent_dim}), "
                        f"got {tuple(z.shape)}."
                    )

                decoder_tokens = self.latent_to_decoder(z).unsqueeze(1)
                decoder_tokens = (
                    decoder_tokens + self.decoder_time_queries
                )
                decoder_tokens = self.decoder_time_pos_enc(decoder_tokens)
                decoder_tokens = self._run_blocks(
                    decoder_tokens,
                    self.decoder_transformer_blocks,
                )

                batch_size = z.size(0)
                frames = self._decode_spatial(
                    decoder_tokens.reshape(
                        batch_size * self.seq_len,
                        d_model,
                    ),
                    self.frame_seed,
                )
                return frames.view(
                    batch_size,
                    self.seq_len,
                    self.C,
                    self.H,
                    self.W,
                )

        return HybridAutoencoderModel()
