from abc import abstractmethod
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..core.models import BlueMathModel


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
    def __init__(self, device: Optional[Union[str, torch.device]] = None, **kwargs):
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

        self.model: Optional[nn.Module] = None
        self.is_fitted = False

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
        Dict[str, list]
            Training history with 'train_loss' and 'val_loss' keys.
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

    def save_pytorch_model(self, model_path: str, **kwargs):
        """
        Save the PyTorch model to a file.

        Parameters
        ----------
        model_path : str
            Path to the file where the model will be saved.
        **kwargs
            Additional arguments for torch.save.
        """

        if self.model is None:
            raise ValueError("PyTorch model must be built before saving.")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "is_fitted": self.is_fitted,
                "model_class": self.__class__.__name__,
            },
            model_path,
            **kwargs,
        )
        self.logger.info(f"PyTorch model saved to {model_path}")

    def load_pytorch_model(self, model_path: str, **kwargs):
        """
        Load a PyTorch model from a file.

        Parameters
        ----------
        model_path : str
            Path to the file where the model is saved.
        **kwargs
            Additional arguments for torch.load.
        """

        if self.model is None:
            raise ValueError("PyTorch model must be built before loading.")

        checkpoint = torch.load(model_path, **kwargs)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.is_fitted = checkpoint.get("is_fitted", False)
        self.logger.info(f"PyTorch model loaded from {model_path}")
