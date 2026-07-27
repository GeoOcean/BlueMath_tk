import copy
import inspect
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..core.models import BlueMathModel
from .metrics import evaluate_reconstruction as evaluate_reconstruction_metric
from .metrics import reconstruction_error as reconstruction_error_metric


class BaseDeepLearningModel(BlueMathModel):
    """
    Base class for all PyTorch deep learning BlueMath models.

    This class provides the basic structure for all deep learning models,
    including common functionality for training, evaluation, and prediction.

    Attributes
    ----------
    model : torch.nn.Module
        The PyTorch model.
    device : torch.device
        The device (CPU/GPU) the model is on.
    is_fitted : bool
        Whether the model has been fitted.
    """

    @abstractmethod
    def __init__(self, device: str | torch.device | None = None, **kwargs):
        """
        Initialize the base deep learning model.

        Parameters
        ----------
        device : str or torch.device, optional
            Device to run the model on ('cpu', 'cuda', etc.).
            If None, uses 'cuda' if available, else 'cpu'.
            Default is None.
        **kwargs
            Additional keyword arguments passed to BlueMathModel.
        """

        super().__init__(**kwargs)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model: nn.Module | None = None
        self.is_fitted = False
        self._build_input_shape: tuple | None = None

        self._exclude_attributes = [
            "model",
        ]

    @abstractmethod
    def _build_model(self, *args, **kwargs) -> nn.Module:
        """
        Build the PyTorch model.

        Returns
        -------
        torch.nn.Module
            The PyTorch model.
        """

        pass


    def _get_reconstruction_target(self, X: np.ndarray) -> np.ndarray:
        """Return the default reconstruction target for ``X``."""
        return X

    @staticmethod
    def _batch_slices(
        n_samples: int,
        batch_size: int,
        avoid_singleton: bool = False,
    ) -> list[tuple[int, int]]:
        """Return batch slices, optionally avoiding a final singleton batch."""
        if avoid_singleton and batch_size == 1:
            raise ValueError(
                "batch_size=1 is incompatible with this model's BatchNorm1d "
                "layers. Use batch_size>=2."
            )

        slices = [
            (start, min(start + batch_size, n_samples))
            for start in range(0, n_samples, batch_size)
        ]

        if (
            avoid_singleton
            and len(slices) > 1
            and slices[-1][1] - slices[-1][0] == 1
        ):
            previous_start, previous_stop = slices[-2]
            final_stop = slices[-1][1]
            previous_size = previous_stop - previous_start

            if previous_size > 2:
                slices[-2] = (previous_start, previous_stop - 1)
                slices[-1] = (previous_stop - 1, final_stop)
            else:
                slices[-2] = (previous_start, final_stop)
                slices.pop()

        return slices

    def _requires_non_singleton_training_batches(self) -> bool:
        """Return whether BatchNorm1d requires at least two training samples."""
        if self.model is None:
            return False
        return any(
            isinstance(module, nn.BatchNorm1d)
            for module in self.model.modules()
        )

    def _validate_or_set_build_input_shape(self, input_shape: tuple) -> None:
        """Store the build shape or reject incompatible repeated fitting."""
        input_shape = tuple(input_shape)
        if self._build_input_shape is None:
            self._build_input_shape = input_shape
            return

        if tuple(self._build_input_shape[1:]) != tuple(input_shape[1:]):
            raise ValueError(
                "The fitted model input shape is incompatible with the new data. "
                f"Expected per-sample shape {self._build_input_shape[1:]}, "
                f"got {input_shape[1:]}."
            )

    @staticmethod
    def _validate_inference_inputs(
        X: np.ndarray,
        batch_size: int,
        name: str = "X",
    ) -> None:
        """Validate common prediction, encoding, and decoding inputs."""
        if not isinstance(X, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array.")
        if X.ndim < 1:
            raise ValueError(f"{name} must have at least one dimension.")
        if len(X) == 0:
            raise ValueError(f"{name} must contain at least one sample.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

    @staticmethod
    def _validate_fit_inputs(
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float,
        batch_size: int,
        epochs: int,
        patience: int,
    ) -> None:
        """Validate common training inputs before building the model."""
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a NumPy array.")
        if X.ndim < 2:
            raise ValueError(
                "X must include a leading sample dimension. "
                "For tabular data use shape (n_samples, n_features)."
            )
        if len(X) != len(y):
            raise ValueError(
                f"X and y must contain the same number of samples; "
                f"got {len(X)} and {len(y)}."
            )
        if not 0.0 < validation_split < 1.0:
            raise ValueError("validation_split must be strictly between 0 and 1.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if patience < 1:
            raise ValueError("patience must be at least 1.")

        split = int((1 - validation_split) * len(X))
        if split < 2:
            raise ValueError(
                "The training split must contain at least two samples. "
                "Increase the dataset size or reduce validation_split."
            )
        if len(X) - split < 1:
            raise ValueError(
                "The validation split must contain at least one sample."
            )

    def _get_init_config(self) -> dict:
        """Collect constructor parameters needed to recreate this model."""
        config = {}
        missing = []
        signature = inspect.signature(self.__class__.__init__)
        for name, parameter in signature.parameters.items():
            if name in {"self", "device"}:
                continue
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                continue
            if hasattr(self, name):
                config[name] = getattr(self, name)
            else:
                missing.append(name)

        if missing:
            raise ValueError(
                f"{self.__class__.__name__} cannot create a self-describing "
                f"checkpoint because these constructor parameters are not "
                f"stored as attributes: {missing}. Override _get_init_config()."
            )
        return config

    @staticmethod
    def _require_scalar_loss(loss: torch.Tensor) -> None:
        """Raise when a criterion returns a non-scalar training loss."""
        if loss.ndim != 0:
            raise ValueError(
                "The training criterion must return a scalar loss. "
                "Use reduction='mean' or reduction='sum'."
            )

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
        """
        Fit the model.

        Parameters
        ----------
        X : np.ndarray
            Training data.
        y : np.ndarray, optional
            Target data. If None, assumes autoencoder (X is target). Default is None.
        validation_split : float, optional
            Fraction of data to use for validation. Default is 0.2.
        epochs : int, optional
            Maximum number of epochs. Default is 500.
        batch_size : int, optional
            Batch size. Default is 64.
        learning_rate : float, optional
            Learning rate. Default is 1e-3.
        optimizer : torch.optim.Optimizer, optional
            Optimizer to use. If None, uses Adam. Default is None.
        criterion : torch.nn.Module, optional
            Loss function. If None, uses MSE. Default is None.
        patience : int, optional
            Early stopping patience. Default is 20.
        verbose : int, optional
            Verbosity level. Default is 1.
        **kwargs
            Additional keyword arguments for model building.

        Returns
        -------
        dict[str, list]
            Training history with 'train_loss' and 'val_loss' keys.
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
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"Patience: {patience_counter}/{patience}"
                )
            elif verbose > 0 and (epoch + 1) % max(1, epochs // 10) == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}"
                )

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.is_fitted = True

        return history

    def predict(
        self, X: np.ndarray, batch_size: int = 64, verbose: int = 1
    ) -> np.ndarray:
        """
        Make predictions.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        batch_size : int, optional
            Batch size for prediction. Default is 64.
        verbose : int, optional
            Verbosity level. If > 0, shows progress bar. Default is 1.

        Returns
        -------
        np.ndarray
            Predictions.

        Raises
        ------
        ValueError
            If model is not fitted.
        """

        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction.")
        self._validate_inference_inputs(X, batch_size)

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        predictions = []

        n_batches = (len(X) + batch_size - 1) // batch_size
        batch_range = range(0, len(X), batch_size)

        if verbose > 0 and n_batches > 1:
            batch_range = tqdm(
                batch_range, desc="Predicting", unit="batch", total=n_batches
            )

        with torch.no_grad():
            for i in batch_range:
                batch_X = X_tensor[i : i + batch_size]
                output = self.model(batch_X)
                predictions.append(output.cpu().numpy())

        return np.concatenate(predictions, axis=0)

    def encode(
        self, X: np.ndarray, batch_size: int = 64, verbose: int = 1
    ) -> np.ndarray:
        """
        Encode input data to latent space.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        batch_size : int, optional
            Batch size for encoding. Default is 64.
        verbose : int, optional
            Verbosity level. If > 0, shows progress bar. Default is 1.

        Returns
        -------
        np.ndarray
            Latent representations.

        Raises
        ------
        ValueError
            If model is not fitted or doesn't support encoding.
        """

        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before encoding.")
        self._validate_inference_inputs(X, batch_size)

        # Check if model has encode_forward method
        if not hasattr(self.model, "encode_forward"):
            raise ValueError(
                f"Model {self.__class__.__name__} does not support encoding. "
                "The model must have an 'encode_forward' method."
            )

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        encodings = []

        n_batches = (len(X) + batch_size - 1) // batch_size
        batch_range = range(0, len(X), batch_size)

        if verbose > 0 and n_batches > 1:
            batch_range = tqdm(
                batch_range, desc="Encoding", unit="batch", total=n_batches
            )

        with torch.no_grad():
            for i in batch_range:
                batch_X = X_tensor[i : i + batch_size]
                encoding = self.model.encode_forward(batch_X)
                encodings.append(encoding.cpu().numpy())

        return np.concatenate(encodings, axis=0)


    def decode(
        self,
        Z: np.ndarray,
        batch_size: int = 64,
        verbose: int = 1,
    ) -> np.ndarray:
        """Decode latent vectors into the reconstruction space."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before decoding.")
        if not hasattr(self.model, "decode_forward"):
            raise ValueError(
                f"Model {self.__class__.__name__} does not support decoding."
            )

        Z = np.asarray(Z)
        if Z.ndim == 1:
            Z = Z[None, :]
        self._validate_inference_inputs(Z, batch_size, name="Z")

        self.model.eval()
        Z_tensor = torch.as_tensor(Z, dtype=torch.float32, device=self.device)
        outputs = []
        batch_range = range(0, len(Z), batch_size)
        n_batches = (len(Z) + batch_size - 1) // batch_size

        if verbose > 0 and n_batches > 1:
            batch_range = tqdm(
                batch_range,
                desc="Decoding",
                unit="batch",
                total=n_batches,
            )

        with torch.no_grad():
            for start in batch_range:
                output = self.model.decode_forward(
                    Z_tensor[start : start + batch_size]
                )
                outputs.append(output.cpu().numpy())

        return np.concatenate(outputs, axis=0)

    def reconstruction_error(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        metric: str = "mse",
        reduction: str = "sample",
        batch_size: int = 64,
        verbose: int = 0,
        eps: float = 0.0,
    ):
        """Compute reconstruction error with the shared metrics module."""
        self._validate_inference_inputs(X, batch_size)
        target = self._get_reconstruction_target(X) if y is None else y
        prediction = self.predict(X, batch_size=batch_size, verbose=verbose)
        return reconstruction_error_metric(
            target,
            prediction,
            metric=metric,
            reduction=reduction,
            eps=eps,
        )

    def evaluate_reconstruction(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        metric: str = "mse",
        batch_size: int = 64,
        verbose: int = 0,
        eps: float = 0.0,
    ) -> dict[str, float]:
        """Return summary statistics for reconstruction error."""
        self._validate_inference_inputs(X, batch_size)
        target = self._get_reconstruction_target(X) if y is None else y
        prediction = self.predict(X, batch_size=batch_size, verbose=verbose)
        return evaluate_reconstruction_metric(
            target,
            prediction,
            metric=metric,
            eps=eps,
        )

    def evaluate(self, X: np.ndarray, **kwargs) -> dict[str, float]:
        """Alias for :meth:`evaluate_reconstruction`."""
        return self.evaluate_reconstruction(X, **kwargs)

    def save_pytorch_model(self, model_path: str, **kwargs):
        """Save weights and metadata required to rebuild the model."""
        if self.model is None:
            raise ValueError("PyTorch model must be built before saving.")
        if self._build_input_shape is None:
            raise ValueError(
                "The model input shape is unknown. Fit or load the model before saving."
            )

        torch.save(
            {
                "checkpoint_version": 2,
                "model_state_dict": self.model.state_dict(),
                "is_fitted": self.is_fitted,
                "model_class": self.__class__.__name__,
                "init_config": self._get_init_config(),
                "build_input_shape": self._build_input_shape,
            },
            model_path,
            **kwargs,
        )
        self.logger.info(f"PyTorch model saved to {model_path}")

    def load_pytorch_model(
        self,
        model_path: str,
        map_location=None,
        **kwargs,
    ):
        """Load a checkpoint into this instance."""
        if map_location is None:
            map_location = self.device

        checkpoint = torch.load(
            model_path,
            map_location=map_location,
            **kwargs,
        )
        checkpoint_class = checkpoint.get("model_class")
        if checkpoint_class and checkpoint_class != self.__class__.__name__:
            raise ValueError(
                f"Checkpoint contains {checkpoint_class}, "
                f"not {self.__class__.__name__}."
            )

        if self.model is None:
            build_input_shape = checkpoint.get("build_input_shape")
            if build_input_shape is None:
                raise ValueError(
                    "This legacy checkpoint does not include build_input_shape. "
                    "Build the model manually before loading it."
                )

            init_config = checkpoint.get("init_config", {})
            for name, value in init_config.items():
                if hasattr(self, name):
                    setattr(self, name, value)

            self._build_input_shape = tuple(build_input_shape)
            self.model = self._build_model(self._build_input_shape)
            self.model = self.model.to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.is_fitted = checkpoint.get("is_fitted", False)
        if self._build_input_shape is None:
            shape = checkpoint.get("build_input_shape")
            if shape is not None:
                self._build_input_shape = tuple(shape)
        self.logger.info(f"PyTorch model loaded from {model_path}")
        return self

    @classmethod
    def from_pytorch_model(
        cls,
        model_path: str,
        device: str | torch.device | None = "cpu",
        map_location="cpu",
        **kwargs,
    ):
        """Create a fitted model from a self-describing checkpoint."""
        checkpoint = torch.load(
            model_path,
            map_location=map_location,
            **kwargs,
        )
        checkpoint_class = checkpoint.get("model_class")
        if checkpoint_class and checkpoint_class != cls.__name__:
            raise ValueError(
                f"Checkpoint contains {checkpoint_class}, not {cls.__name__}."
            )

        init_config = checkpoint.get("init_config")
        build_input_shape = checkpoint.get("build_input_shape")
        if init_config is None or build_input_shape is None:
            raise ValueError(
                "Automatic loading requires a version-2 checkpoint containing "
                "init_config and build_input_shape."
            )

        instance = cls(device=device, **dict(init_config))
        instance._build_input_shape = tuple(build_input_shape)
        instance.model = instance._build_model(instance._build_input_shape)
        instance.model = instance.model.to(instance.device)
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance.is_fitted = checkpoint.get("is_fitted", False)
        return instance
