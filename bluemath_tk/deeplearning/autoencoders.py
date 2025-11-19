"""
Autoencoders module.

This module is a pytorch translation from a tensorflow implementation developed by Sergio López Dubón.

This module contains the following autoencoders:
- StandardAutoencoder
- OrthogonalAutoencoder
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


class StandardAutoencoder(BaseDeepLearningModel):
    """
    Standard fully-connected autoencoder.

    A simple feedforward autoencoder with symmetric encoder-decoder architecture.
    Designed for tabular/flattened data (not images or sequences).

    Input Shape
    -----------
    X : np.ndarray
        Input data with shape (n_samples, n_features) or (n_samples,).
        - For 2D arrays: (n_samples, n_features) - each row is a sample
        - For 1D arrays: (n_features,) - single sample (will be reshaped)
        The model automatically flattens multi-dimensional inputs.

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
        # Handle input shape: (n_samples, n_features) or (n_features,)
        if len(input_shape) == 1:
            n_features = input_shape[0]
        else:
            # Take last dimension as features (handles (n_samples, n_features))
            n_features = input_shape[-1]

        class StandardAutoencoderModel(nn.Module):
            def __init__(self, n_features, hidden_dims, k):
                super().__init__()
                self.n_features = n_features
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
                return x_recon

            def encode_forward(self, x):
                """Encode input to latent space."""
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                elif x.dim() == 1:
                    x = x.unsqueeze(0)
                return self.encoder(x)

        return StandardAutoencoderModel(n_features, self.hidden_dims, self.k)


class OrthogonalAutoencoder(BaseDeepLearningModel):
    """
    Orthogonal autoencoder with orthogonal regularization.

    Adds orthogonality constraints on encoder weights and latent decorrelation
    to encourage more interpretable latent representations.
    Designed for tabular/flattened data (not images or sequences).

    Input Shape
    -----------
    X : np.ndarray
        Input data with shape (n_samples, n_features) or (n_samples,).
        - For 2D arrays: (n_samples, n_features) - each row is a sample
        - For 1D arrays: (n_features,) - single sample (will be reshaped)
        The model automatically flattens multi-dimensional inputs.

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
        # Handle input shape: (n_samples, n_features) or (n_features,)
        if len(input_shape) == 1:
            n_features = input_shape[0]
        else:
            n_features = input_shape[-1]

        class OrthogonalAutoencoderModel(nn.Module):
            def __init__(self, n_features, hidden_dims, k, lambda_W, lambda_Z):
                super().__init__()
                self.n_features = n_features
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
                return x_recon

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

        return OrthogonalAutoencoderModel(
            n_features, self.hidden_dims, self.k, self.lambda_W, self.lambda_Z
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
        if self.model is None:
            self.model = self._build_model(X.shape, **kwargs)
            self.model = self.model.to(self.device)

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
            n_batches = (len(Xtr) + batch_size - 1) // batch_size

            for i in range(0, len(Xtr), batch_size):
                batch_X = Xtr_tensor[i : i + batch_size]
                batch_y = ytr_tensor[i : i + batch_size]

                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)

                # Add regularization losses
                ortho_loss, decorr_loss = self.model.get_regularization_losses()
                if ortho_loss is not None:
                    loss = loss + ortho_loss
                if decorr_loss is not None:
                    loss = loss + decorr_loss

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= n_batches
            history["train_loss"].append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                n_val_batches = (len(Xval) + batch_size - 1) // batch_size
                for i in range(0, len(Xval), batch_size):
                    batch_X = Xval_tensor[i : i + batch_size]
                    batch_y = yval_tensor[i : i + batch_size]

                    output = self.model(batch_X)
                    loss = criterion(output, batch_y)

                    # Add regularization losses for validation
                    ortho_loss, decorr_loss = self.model.get_regularization_losses()
                    if ortho_loss is not None:
                        loss = loss + ortho_loss
                    if decorr_loss is not None:
                        loss = loss + decorr_loss

                    val_loss += loss.item()

                val_loss /= n_val_batches
                history["val_loss"].append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
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
                rec_patches = self.unpatchify(y)  # (B, C, H+pad_h, W+pad_w)

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
    """
    ConvLSTM autoencoder for spatiotemporal data (image sequences).

    Combines convolutional and LSTM layers for spatiotemporal sequences.
    Designed for video-like data or time series of images.

    Input Shape
    -----------
    X : np.ndarray
        Input data with shape (n_samples, seq_len, C, H, W).
        - n_samples: number of sequences
        - seq_len: number of frames in each sequence (automatically inferred from X.shape[1])
        - C: number of channels (e.g., 1 for grayscale, 3 for RGB)
        - H, W: height and width of each frame

    Examples
    --------
    >>> # Video-like data (time series of images)
    >>> X = np.random.randn(100, 10, 3, 64, 64)  # 100 sequences, 10 frames, 3 channels, 64x64
    >>> ae = ConvLSTMAutoencoder(k=20)
    >>> history = ae.fit(X, epochs=10)
    >>> X_recon = ae.predict(X)  # Shape: (100, 3, 64, 64) - single frame reconstruction
    >>> Z = ae.encode(X)  # Latent representations: (100, 20)

    Parameters
    ----------
    k : int, optional
        Number of latent dimensions, by default 20.
    device : str or torch.device, optional
        Device to run the model on.
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
        """Build the ConvLSTM autoencoder model."""
        # Parse input shape: (n_samples, seq_len, C, H, W) - channels-first format
        if len(input_shape) != 5:
            raise ValueError(
                f"ConvLSTMAutoencoder expects 5D input shape (n_samples, seq_len, C, H, W), "
                f"got {input_shape} with {len(input_shape)} dimensions"
            )
        # (n_samples, seq_len, C, H, W)
        seq_len = input_shape[1]  # Infer from input shape
        C, H, W = input_shape[2], input_shape[3], input_shape[4]

        # Compute padding so (H+pad) and (W+pad) are divisible by 4
        pad_h = (-H) % 4
        pad_w = (-W) % 4

        class ConvLSTMAutoencoderModel(nn.Module):
            def __init__(self, seq_len, H, W, C, k, pad_h, pad_w):
                super().__init__()
                self.seq_len = seq_len
                self.pad_h = pad_h
                self.pad_w = pad_w
                self.H = H
                self.W = W
                self.C = C

                # ConvLSTM layers
                from .layers import ConvLSTM

                self.convlstm1 = ConvLSTM(
                    input_dim=C,
                    hidden_dim=32,
                    kernel_size=3,
                    num_layers=1,
                    batch_first=True,
                    return_all_layers=False,
                )
                self.bn1 = nn.BatchNorm3d(32)
                self.convlstm2 = ConvLSTM(
                    input_dim=32,
                    hidden_dim=32,
                    kernel_size=3,
                    num_layers=1,
                    batch_first=True,
                    return_all_layers=False,
                )

                # Spatial downsample
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1)
                self.pool1 = nn.MaxPool2d(2)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool2 = nn.MaxPool2d(2)

                # Flatten and latent
                H_enc = (H + pad_h) // 4
                W_enc = (W + pad_w) // 4
                self.flat_size = H_enc * W_enc * 64
                self.latent = nn.Linear(self.flat_size, k)

                # Decoder
                self.fc_dec = nn.Linear(k, self.flat_size)
                self.upsample1 = nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=True
                )
                self.deconv1 = nn.ConvTranspose2d(64, 64, 3, padding=1)
                self.upsample2 = nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=True
                )
                self.deconv2 = nn.ConvTranspose2d(64, C, 3, padding=1)

            def forward(self, x):
                # Only accept (B, T, C, H, W) format - channels-first
                if x.dim() != 5:
                    raise ValueError(
                        f"ConvLSTMAutoencoder expects 5D input (B, T, C, H, W), "
                        f"got shape {x.shape}"
                    )
                B, T, C_in, H, W = x.shape

                # Validate channels are in the correct position
                if C_in != self.C:
                    raise ValueError(
                        f"ConvLSTMAutoencoder expects channels-first format (B, T, C, H, W). "
                        f"Expected C={self.C} at position 2, but got shape {x.shape}. "
                        f"If your data is channels-last (B, T, H, W, C), please permute it: "
                        f"X = np.transpose(X, (0, 1, 4, 2, 3))"
                    )

                # Pad
                if self.pad_h > 0 or self.pad_w > 0:
                    x = F.pad(x, (0, self.pad_w, 0, self.pad_h))

                # ConvLSTM
                x_list, _ = self.convlstm1(x)  # Returns list
                x = x_list[0]  # (B, T, 32, H+pad, W+pad)
                x = x.permute(0, 2, 1, 3, 4)  # (B, 32, T, H+pad, W+pad)
                x = self.bn1(x)
                x = x.permute(0, 2, 1, 3, 4)  # (B, T, 32, H+pad, W+pad)
                x_list, _ = self.convlstm2(x)
                x = x_list[0]  # (B, T, 32, H+pad, W+pad)

                # Take last timestep
                x = x[:, -1]  # (B, 32, H+pad, W+pad)

                # Spatial downsample
                x = F.relu(self.conv1(x))
                x = self.pool1(x)
                x = F.relu(self.conv2(x))
                x = self.pool2(x)

                # Flatten and latent
                x = x.view(B, -1)
                z = self.latent(x)

                # Decoder
                x = F.relu(self.fc_dec(z))
                x = x.view(B, 64, (H + self.pad_h) // 4, (W + self.pad_w) // 4)
                x = self.upsample1(x)
                x = F.relu(self.deconv1(x))
                x = self.upsample2(x)
                x = self.deconv2(x)

                # Crop padding
                if self.pad_h > 0 or self.pad_w > 0:
                    x = x[:, :, : self.H, : self.W]

                return x

            def encode_forward(self, x):
                """Encode input to latent space."""
                # Only accept (B, T, C, H, W) format - channels-first
                if x.dim() != 5:
                    raise ValueError(
                        f"ConvLSTMAutoencoder expects 5D input (B, T, C, H, W), "
                        f"got shape {x.shape}"
                    )
                B, T, C_in, H, W = x.shape

                # Validate channels are in the correct position
                if C_in != self.C:
                    raise ValueError(
                        f"ConvLSTMAutoencoder expects channels-first format (B, T, C, H, W). "
                        f"Expected C={self.C} at position 2, but got shape {x.shape}. "
                        f"If your data is channels-last (B, T, H, W, C), please permute it: "
                        f"X = np.transpose(X, (0, 1, 4, 2, 3))"
                    )

                # Pad
                if self.pad_h > 0 or self.pad_w > 0:
                    x = F.pad(x, (0, self.pad_w, 0, self.pad_h))

                # ConvLSTM
                x_list, _ = self.convlstm1(x)
                x = x_list[0]
                x = x.permute(0, 2, 1, 3, 4)
                x = self.bn1(x)
                x = x.permute(0, 2, 1, 3, 4)
                x_list, _ = self.convlstm2(x)
                x = x_list[0]

                # Take last timestep
                x = x[:, -1]

                # Spatial downsample
                x = F.relu(self.conv1(x))
                x = self.pool1(x)
                x = F.relu(self.conv2(x))
                x = self.pool2(x)

                # Flatten and latent
                x = x.view(B, -1)
                z = self.latent(x)

                return z

        return ConvLSTMAutoencoderModel(seq_len, H, W, C, self.k, pad_h, pad_w)


class HybridConvLSTMTransformerAutoencoder(BaseDeepLearningModel):
    """
    Hybrid ConvLSTM + Transformer autoencoder for spatiotemporal data.

    Combines ConvLSTM for spatiotemporal encoding with Transformer attention
    for temporal modeling.
    Designed for complex spatiotemporal patterns (video-like data or time series of images).

    Input Shape
    -----------
    X : np.ndarray
        Input data with shape (n_samples, seq_len, C, H, W).
        - n_samples: number of sequences
        - seq_len: number of frames in each sequence (automatically inferred from X.shape[1])
        - C: number of channels (e.g., 1 for grayscale, 3 for RGB)
        - H, W: height and width of each frame

    Examples
    --------
    >>> # Complex spatiotemporal data
    >>> X = np.random.randn(100, 10, 3, 64, 64)  # 100 sequences, 10 frames, 3 channels, 64x64
    >>> ae = HybridConvLSTMTransformerAutoencoder(k=20, d_model=256)
    >>> history = ae.fit(X, epochs=10)
    >>> X_recon = ae.predict(X)  # Shape: (100, 3, 64, 64) - single frame reconstruction
    >>> Z = ae.encode(X)  # Latent representations: (100, 20)

    Parameters
    ----------
    k : int, optional
        Number of latent dimensions, by default 20.
    d_model : int, optional
        Model dimension, by default 256.
    n_heads : int, optional
        Number of attention heads, by default 4.
    n_layers : int, optional
        Number of transformer layers, by default 2.
    efficient_attention : str, optional
        Use 'linear' for efficient linear attention, None for standard MHA,
        by default 'linear'.
    device : str or torch.device, optional
        Device to run the model on.
    **kwargs
        Additional keyword arguments passed to BaseDeepLearningModel.
    """

    def __init__(
        self,
        k: int = 20,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        efficient_attention: Optional[str] = "linear",
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.efficient_attention = efficient_attention
        self.k = k
        super().__init__(device=device, **kwargs)

    def _build_model(self, input_shape: Tuple, **kwargs) -> nn.Module:
        """Build the hybrid autoencoder model."""
        # Parse input shape: (n_samples, seq_len, C, H, W) - channels-first format
        if len(input_shape) != 5:
            raise ValueError(
                f"HybridConvLSTMTransformerAutoencoder expects 5D input shape (n_samples, seq_len, C, H, W), "
                f"got {input_shape} with {len(input_shape)} dimensions"
            )
        # (n_samples, seq_len, C, H, W)
        seq_len = input_shape[1]  # Infer from input shape
        C, H, W = input_shape[2], input_shape[3], input_shape[4]

        # Compute padding
        pad_h = (-H) % 4
        pad_w = (-W) % 4

        class HybridAutoencoderModel(nn.Module):
            def __init__(
                self,
                seq_len,
                H,
                W,
                C,
                k,
                d_model,
                n_heads,
                n_layers,
                efficient_attention,
                pad_h,
                pad_w,
            ):
                super().__init__()
                self.seq_len = seq_len
                self.pad_h = pad_h
                self.pad_w = pad_w
                self.H = H
                self.W = W
                self.C = C

                # ConvLSTM stack
                from .layers import ConvLSTM

                self.convlstm1 = ConvLSTM(
                    input_dim=C,
                    hidden_dim=32,
                    kernel_size=3,
                    num_layers=1,
                    batch_first=True,
                    return_all_layers=True,
                )
                self.convlstm2 = ConvLSTM(
                    input_dim=32,
                    hidden_dim=32,
                    kernel_size=3,
                    num_layers=1,
                    batch_first=True,
                    return_all_layers=True,
                )

                # Spatial downsample + per-frame embedding
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1)
                self.pool1 = nn.MaxPool2d(2)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool2 = nn.MaxPool2d(2)

                H_enc = (H + pad_h) // 4
                W_enc = (W + pad_w) // 4
                self.global_pool = nn.AdaptiveAvgPool2d(1)
                self.frame_embed = nn.Linear(64, d_model)
                self.time_pos_enc = TimePositionalEncoding()

                # Temporal Transformer / Linear attention
                if efficient_attention == "linear":
                    self.transformer_blocks = nn.ModuleList(
                        [
                            nn.ModuleDict(
                                {
                                    "norm1": nn.LayerNorm(d_model),
                                    "attn": LinearSelfAttention(d_model, n_heads),
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
                else:
                    self.transformer_blocks = nn.ModuleList(
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

                # Pool time to one latent vector
                self.global_pool_time = nn.AdaptiveAvgPool1d(1)
                self.latent = nn.Linear(d_model, k)

                # Decoder
                self.fc_dec = nn.Linear(k, H_enc * W_enc * 64)
                self.upsample1 = nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=True
                )
                self.deconv1 = nn.ConvTranspose2d(64, 64, 3, padding=1)
                self.upsample2 = nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=True
                )
                self.deconv2 = nn.ConvTranspose2d(64, C, 3, padding=1)

            def forward(self, x):
                # Only accept (B, T, C, H, W) format - channels-first
                if x.dim() != 5:
                    raise ValueError(
                        f"HybridConvLSTMTransformerAutoencoder expects 5D input "
                        f"(B, T, C, H, W), got shape {x.shape}"
                    )
                B, T, C_in, H, W = x.shape

                # Validate channels are in the correct position
                if C_in != self.C:
                    raise ValueError(
                        f"HybridConvLSTMTransformerAutoencoder expects channels-first format (B, T, C, H, W). "
                        f"Expected C={self.C} at position 2, but got shape {x.shape}. "
                        f"If your data is channels-last (B, T, H, W, C), please permute it: "
                        f"X = np.transpose(X, (0, 1, 4, 2, 3))"
                    )

                # Pad
                if self.pad_h > 0 or self.pad_w > 0:
                    x = F.pad(x, (0, self.pad_w, 0, self.pad_h))

                # ConvLSTM
                x_list, _ = self.convlstm1(x)
                x = x_list[0]  # Take output (B, T, 32, H+pad, W+pad)
                x_list, _ = self.convlstm2(x)
                x = x_list[0]  # (B, T, 32, H+pad, W+pad)

                # Spatial downsample per frame
                frame_features = []
                for t in range(T):
                    frame = x[:, t]  # (B, 32, H+pad, W+pad)
                    frame = F.relu(self.conv1(frame))
                    frame = self.pool1(frame)
                    frame = F.relu(self.conv2(frame))
                    frame = self.pool2(frame)  # (B, 64, H_enc, W_enc)
                    frame = self.global_pool(frame).squeeze(-1).squeeze(-1)  # (B, 64)
                    frame = self.frame_embed(frame)  # (B, d_model)
                    frame_features.append(frame)

                x = torch.stack(frame_features, dim=1)  # (B, T, d_model)
                x = self.time_pos_enc(x)

                # Transformer blocks
                if self.efficient_attention == "linear":
                    for block in self.transformer_blocks:
                        x_norm = block["norm1"](x)
                        attn_out = block["attn"](x_norm)
                        x = x + attn_out
                        x = x + block["mlp"](block["norm2"](x))
                else:
                    for block in self.transformer_blocks:
                        x = block(x)

                # Pool time to one latent vector
                x = x.transpose(1, 2)  # (B, d_model, T)
                x = self.global_pool_time(x).squeeze(-1)  # (B, d_model)
                z = self.latent(x)  # (B, k)

                # Decoder
                x = F.relu(self.fc_dec(z))
                H_enc = (H + self.pad_h) // 4
                W_enc = (W + self.pad_w) // 4
                x = x.view(B, 64, H_enc, W_enc)
                x = self.upsample1(x)
                x = F.relu(self.deconv1(x))
                x = self.upsample2(x)
                x = self.deconv2(x)

                # Crop padding
                if self.pad_h > 0 or self.pad_w > 0:
                    x = x[:, :, : self.H, : self.W]

                return x

            def encode_forward(self, x):
                """Encode input to latent space."""
                # Only accept (B, T, C, H, W) format - channels-first
                if x.dim() != 5:
                    raise ValueError(
                        f"HybridConvLSTMTransformerAutoencoder expects 5D input "
                        f"(B, T, C, H, W), got shape {x.shape}"
                    )
                B, T, C_in, H, W = x.shape

                # Validate channels are in the correct position
                if C_in != self.C:
                    raise ValueError(
                        f"HybridConvLSTMTransformerAutoencoder expects channels-first format (B, T, C, H, W). "
                        f"Expected C={self.C} at position 2, but got shape {x.shape}. "
                        f"If your data is channels-last (B, T, H, W, C), please permute it: "
                        f"X = np.transpose(X, (0, 1, 4, 2, 3))"
                    )

                # Pad
                if self.pad_h > 0 or self.pad_w > 0:
                    x = F.pad(x, (0, self.pad_w, 0, self.pad_h))

                # ConvLSTM
                x_list, _ = self.convlstm1(x)
                x = x_list[0]
                x_list, _ = self.convlstm2(x)
                x = x_list[0]

                # Spatial downsample per frame
                frame_features = []
                for t in range(T):
                    frame = x[:, t]
                    frame = F.relu(self.conv1(frame))
                    frame = self.pool1(frame)
                    frame = F.relu(self.conv2(frame))
                    frame = self.pool2(frame)
                    frame = self.global_pool(frame).squeeze(-1).squeeze(-1)
                    frame = self.frame_embed(frame)
                    frame_features.append(frame)

                x = torch.stack(frame_features, dim=1)
                x = self.time_pos_enc(x)

                # Transformer blocks
                if self.efficient_attention == "linear":
                    for block in self.transformer_blocks:
                        x_norm = block["norm1"](x)
                        attn_out = block["attn"](x_norm)
                        x = x + attn_out
                        x = x + block["mlp"](block["norm2"](x))
                else:
                    for block in self.transformer_blocks:
                        x = block(x)

                # Pool time to one latent vector
                x = x.transpose(1, 2)
                x = self.global_pool_time(x).squeeze(-1)
                z = self.latent(x)

                return z

        return HybridAutoencoderModel(
            seq_len,
            H,
            W,
            C,
            self.k,
            self.d_model,
            self.n_heads,
            self.n_layers,
            self.efficient_attention,
            pad_h,
            pad_w,
        )
