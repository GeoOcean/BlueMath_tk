"""Spatially explicit spatiotemporal autoencoders for BlueMath_tk."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from ._base_model import BaseDeepLearningModel
from .layers import ConvLSTM


class _FactorizedSpatiotemporalBlock(nn.Module):
    """Apply temporal attention, spatial attention, and a feed-forward block."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.spatial_norm = nn.LayerNorm(d_model)
        self.spatial_attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_tokens, d_model = x.shape

        temporal = x.permute(0, 2, 1, 3).reshape(
            batch_size * n_tokens,
            seq_len,
            d_model,
        )
        temporal_normalized = self.temporal_norm(temporal)
        temporal_update, _ = self.temporal_attention(
            temporal_normalized,
            temporal_normalized,
            temporal_normalized,
            need_weights=False,
        )
        temporal = temporal + temporal_update
        x = temporal.reshape(
            batch_size,
            n_tokens,
            seq_len,
            d_model,
        ).permute(0, 2, 1, 3)

        spatial = x.reshape(batch_size * seq_len, n_tokens, d_model)
        spatial_normalized = self.spatial_norm(spatial)
        spatial_update, _ = self.spatial_attention(
            spatial_normalized,
            spatial_normalized,
            spatial_normalized,
            need_weights=False,
        )
        spatial = spatial + spatial_update
        x = spatial.reshape(batch_size, seq_len, n_tokens, d_model)

        return x + self.feed_forward(self.feed_forward_norm(x))


class SpatialTokenConvLSTMTransformerAutoencoder(BaseDeepLearningModel):
    """ConvLSTM-Transformer autoencoder with explicit spatial tokens.

    The model keeps several spatial tokens at every timestep, applies
    factorized temporal and spatial attention, compresses the full input
    window to one vector of size ``k``, and reconstructs the complete input
    sequence.

    Parameters
    ----------
    k : int, optional
        Number of latent dimensions, by default 20.
    spatial_pool_size : tuple of int, optional
        Number of pooled token rows and columns, by default ``(4, 4)``.
        Each value must not exceed the corresponding spatial-encoder output
        dimension, approximately ``ceil(H / 4)`` and ``ceil(W / 4)``.
    d_model : int, optional
        Token dimension, by default 128.
    n_heads : int, optional
        Number of attention heads, by default 4.
    n_layers : int, optional
        Number of factorized attention blocks in both encoder and decoder,
        by default 2.
    device : str or torch.device, optional
        Device on which to run the model.
    **kwargs
        Additional keyword arguments passed to ``BaseDeepLearningModel``.
    """

    def __init__(
        self,
        k: int = 20,
        spatial_pool_size: tuple[int, int] = (4, 4),
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        device: str | torch.device | None = None,
        **kwargs,
    ):
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer.")
        if (
            not isinstance(spatial_pool_size, tuple)
            or len(spatial_pool_size) != 2
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
                for value in spatial_pool_size
            )
        ):
            raise ValueError(
                "spatial_pool_size must be a tuple of two positive integers."
            )
        if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model < 1:
            raise ValueError("d_model must be a positive integer.")
        if not isinstance(n_heads, int) or isinstance(n_heads, bool) or n_heads < 1:
            raise ValueError("n_heads must be a positive integer.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if not isinstance(n_layers, int) or isinstance(n_layers, bool) or n_layers < 1:
            raise ValueError("n_layers must be a positive integer.")

        self.k = k
        self.spatial_pool_size = tuple(spatial_pool_size)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
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
        if not isinstance(target, np.ndarray):
            raise TypeError("y must be a NumPy array.")
        if tuple(target.shape) != tuple(X.shape):
            raise ValueError(
                "Spatial-token autoencoder targets must have the same "
                "shape as X for full-sequence reconstruction."
            )
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
        """Use the complete spatiotemporal sequence as the target."""
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        if X.ndim != 5:
            raise ValueError(
                "SpatialTokenConvLSTMTransformerAutoencoder expects 5D input "
                "(n_samples, seq_len, C, H, W)."
            )
        return X

    def _build_model(self, input_shape: tuple, **kwargs) -> nn.Module:
        """Build the spatial-token encoder and full-sequence decoder."""
        if len(input_shape) != 5:
            raise ValueError(
                "SpatialTokenConvLSTMTransformerAutoencoder expects input "
                "shape (n_samples, seq_len, C, H, W)."
            )

        seq_len = input_shape[1]
        channels, height, width = input_shape[2:]
        if any(value < 1 for value in (seq_len, channels, height, width)):
            raise ValueError("All sequence and spatial dimensions must be positive.")

        pooled_height, pooled_width = self.spatial_pool_size
        encoded_height = (height + 3) // 4
        encoded_width = (width + 3) // 4
        if pooled_height > encoded_height or pooled_width > encoded_width:
            raise ValueError(
                "spatial_pool_size cannot exceed the spatial-encoder output "
                f"size {(encoded_height, encoded_width)} for input "
                f"shape {(height, width)}."
            )
        n_tokens = pooled_height * pooled_width
        d_model = self.d_model
        latent_dim = self.k
        n_heads = self.n_heads
        n_layers = self.n_layers

        class SpatialTokenModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.seq_len = seq_len
                self.channels = channels
                self.height = height
                self.width = width
                self.pooled_height = pooled_height
                self.pooled_width = pooled_width
                self.n_tokens = n_tokens
                self.d_model = d_model
                self.latent_dim = latent_dim

                self.convlstm1 = ConvLSTM(
                    input_dim=channels,
                    hidden_dim=32,
                    kernel_size=(3, 3),
                    num_layers=1,
                    batch_first=True,
                    return_all_layers=False,
                )
                self.convlstm2 = ConvLSTM(
                    input_dim=32,
                    hidden_dim=32,
                    kernel_size=(3, 3),
                    num_layers=1,
                    batch_first=True,
                    return_all_layers=False,
                )
                self.spatial_encoder = nn.Sequential(
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.GroupNorm(8, 64),
                    nn.GELU(),
                    nn.Conv2d(64, d_model, 3, stride=2, padding=1),
                    nn.GroupNorm(_group_count(d_model), d_model),
                    nn.GELU(),
                )

                self.encoder_time_embedding = nn.Parameter(
                    torch.zeros(1, seq_len, 1, d_model)
                )
                self.encoder_space_embedding = nn.Parameter(
                    torch.zeros(1, 1, n_tokens, d_model)
                )
                self.encoder_blocks = nn.ModuleList(
                    [
                        _FactorizedSpatiotemporalBlock(d_model, n_heads)
                        for _ in range(n_layers)
                    ]
                )
                self.latent_norm = nn.LayerNorm(d_model)
                self.latent = nn.Linear(d_model, latent_dim)

                self.latent_to_tokens = nn.Linear(latent_dim, d_model)
                self.decoder_time_query = nn.Parameter(
                    torch.zeros(1, seq_len, 1, d_model)
                )
                self.decoder_space_query = nn.Parameter(
                    torch.zeros(1, 1, n_tokens, d_model)
                )
                self.decoder_blocks = nn.ModuleList(
                    [
                        _FactorizedSpatiotemporalBlock(d_model, n_heads)
                        for _ in range(n_layers)
                    ]
                )
                self.spatial_decoder = nn.Sequential(
                    nn.Conv2d(d_model, 64, 3, padding=1),
                    nn.GroupNorm(8, 64),
                    nn.GELU(),
                    nn.Conv2d(64, 32, 3, padding=1),
                    nn.GroupNorm(8, 32),
                    nn.GELU(),
                    nn.Conv2d(32, channels, 3, padding=1),
                )

                for parameter in (
                    self.encoder_time_embedding,
                    self.encoder_space_embedding,
                    self.decoder_time_query,
                    self.decoder_space_query,
                ):
                    nn.init.normal_(parameter, std=0.02)

            def _validate_input(self, x: torch.Tensor) -> None:
                if x.dim() != 5:
                    raise ValueError(
                        "Expected 5D input with shape (B, T, C, H, W)."
                    )
                expected = (
                    self.seq_len,
                    self.channels,
                    self.height,
                    self.width,
                )
                actual = tuple(x.shape[1:])
                if actual != expected:
                    raise ValueError(
                        f"Expected per-sample shape {expected}, got {actual}."
                    )

            def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
                self._validate_input(x)
                batch_size = x.size(0)
                features, _ = self.convlstm1(x)
                features, _ = self.convlstm2(features[0])
                features = features[0]

                features = features.reshape(
                    batch_size * self.seq_len,
                    32,
                    self.height,
                    self.width,
                )
                features = self.spatial_encoder(features)
                features = functional.adaptive_avg_pool2d(
                    features,
                    (self.pooled_height, self.pooled_width),
                )
                tokens = features.flatten(2).transpose(1, 2)
                tokens = tokens.reshape(
                    batch_size,
                    self.seq_len,
                    self.n_tokens,
                    self.d_model,
                )
                tokens = (
                    tokens
                    + self.encoder_time_embedding
                    + self.encoder_space_embedding
                )
                for block in self.encoder_blocks:
                    tokens = block(tokens)
                return tokens

            def encode_forward(self, x: torch.Tensor) -> torch.Tensor:
                tokens = self._encode_tokens(x)
                pooled = tokens.mean(dim=(1, 2))
                return self.latent(self.latent_norm(pooled))

            def decode_forward(self, z: torch.Tensor) -> torch.Tensor:
                if z.dim() != 2 or z.shape[1] != self.latent_dim:
                    raise ValueError(
                        "Latent input must have shape "
                        f"(batch, {self.latent_dim})."
                    )
                batch_size = z.size(0)
                seed = self.latent_to_tokens(z).reshape(
                    batch_size,
                    1,
                    1,
                    self.d_model,
                )
                tokens = (
                    seed
                    + self.decoder_time_query
                    + self.decoder_space_query
                )
                for block in self.decoder_blocks:
                    tokens = block(tokens)

                features = tokens.permute(0, 1, 3, 2).reshape(
                    batch_size * self.seq_len,
                    self.d_model,
                    self.pooled_height,
                    self.pooled_width,
                )
                features = functional.interpolate(
                    features,
                    size=(self.height, self.width),
                    mode="bilinear",
                    align_corners=False,
                )
                reconstruction = self.spatial_decoder(features)
                return reconstruction.reshape(
                    batch_size,
                    self.seq_len,
                    self.channels,
                    self.height,
                    self.width,
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.decode_forward(self.encode_forward(x))

        return SpatialTokenModel()


def _group_count(n_channels: int) -> int:
    """Return a GroupNorm group count that divides ``n_channels``."""
    for candidate in (8, 4, 2, 1):
        if n_channels % candidate == 0:
            return candidate
    return 1
