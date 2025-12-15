"""
Gaussian Process models module.

This module contains Gaussian Process Regression models using GPyTorch.

Classes:
- BaseGPRModel: Base class for all GP models
- ExactGPModel: Exact Gaussian Process Regression model

1. Wang, Z., Leung, M., Mukhopadhyay, S., et al. (2024). "A hybrid statistical–dynamical framework for compound coastal flooding analysis." *Environmental Research Letters*, 20(1), 014005.
2. Wang, Z., Leung, M., Mukhopadhyay, S., et al. (2025). "Compound coastal flooding in San Francisco Bay under climate change." *npj Natural Hazards*, 2(1), 3.
"""

from abc import abstractmethod
from typing import Dict, Optional, Tuple, Union

import gpytorch
import numpy as np
import torch
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from tqdm import tqdm

from ..core.models import BlueMathModel


class BaseGPRModel(BlueMathModel):
    """
    Base class for Gaussian Process Regression models.

    This class provides common functionality for all GP models, including:
    - GP-specific training with marginal log likelihood
    - Prediction with uncertainty quantification
    - Model save/load with likelihood handling

    GP models differ from standard deep learning models in several ways:
    - Use marginal log likelihood (MLL) instead of standard loss functions
    - Require explicit training data setting via set_train_data()
    - Return distributions (mean + variance) rather than point estimates
    - Typically train on full dataset (no batching during training)

    GP models inherit directly from BlueMathModel (not BaseDeepLearningModel)
    because their training and prediction workflows are fundamentally different
    from standard neural networks.

    Attributes
    ----------
    model : gpytorch.models.GP
        The GPyTorch model.
    device : torch.device
        The device (CPU/GPU) the model is on.
    is_fitted : bool
        Whether the model has been fitted.
    likelihood : gpytorch.likelihoods.Likelihood
        The GP likelihood module.
    mll : gpytorch.mlls.MarginalLogLikelihood
        The marginal log likelihood objective.
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        """
        Initialize the base GP model.

        Parameters
        ----------
        device : str or torch.device, optional
            Device to run the model on. Default is None (auto-detect GPU/CPU).
        **kwargs
            Additional keyword arguments passed to BlueMathModel.
        """
        super().__init__(**kwargs)

        # Device management (similar to BaseDeepLearningModel but GP-specific)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # GP-specific attributes
        self.model: Optional[gpytorch.models.GP] = None
        self.is_fitted = False
        self.likelihood: Optional[gpytorch.likelihoods.Likelihood] = None
        self.mll: Optional[gpytorch.mlls.MarginalLogLikelihood] = None

        # Exclude from pickling (GPyTorch objects need special handling)
        self._exclude_attributes = [
            "model",
            "likelihood",
            "mll",
        ]

    @abstractmethod
    def _build_kernel(self, input_dim: int) -> Kernel:
        """
        Build the covariance kernel.

        Parameters
        ----------
        input_dim : int
            Number of input dimensions.

        Returns
        -------
        gpytorch.kernels.Kernel
            The covariance kernel.
        """

        pass

    @abstractmethod
    def _build_model(self, input_shape: Tuple, **kwargs) -> gpytorch.models.GP:
        """
        Build the GPyTorch model.

        Parameters
        ----------
        input_shape : Tuple
            Shape of input data.

        Returns
        -------
        gpytorch.models.GP
            The GPyTorch model.
        """

        pass

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        learning_rate: float = 0.1,
        optimizer: Optional[torch.optim.Optimizer] = None,
        patience: int = 30,
        verbose: int = 1,
        **kwargs,
    ) -> Dict[str, list]:
        """
        Fit the Gaussian Process model.

        GP models use marginal log likelihood (MLL) optimization, which is
        fundamentally different from standard deep learning training.

        Parameters
        ----------
        X : np.ndarray
            Training input data with shape (n_samples, n_features).
        y : np.ndarray
            Training target data with shape (n_samples,) or (n_samples, 1).
        epochs : int, optional
            Maximum number of training epochs. Default is 200.
        learning_rate : float, optional
            Learning rate for optimizer. Default is 0.1.
        optimizer : torch.optim.Optimizer, optional
            Optimizer to use. If None, uses Adam. Default is None.
        patience : int, optional
            Early stopping patience. Default is 30.
        verbose : int, optional
            Verbosity level. Default is 1.
        **kwargs
            Additional keyword arguments passed to _build_model.

        Returns
        -------
        Dict[str, list]
            Training history with 'train_loss' key (negative MLL).
        """

        # Reshape y if needed
        if y.ndim > 1:
            y = y.ravel()

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Build model if not already built
        if self.model is None:
            self.model = self._build_model(X.shape, **kwargs)
            # Initialize likelihood if not set
            if self.likelihood is None:
                self.likelihood = GaussianLikelihood().to(self.device)
            # Initialize MLL
            self.mll = self._build_mll(self.likelihood, self.model)

        # Always update training data (allows retraining with new data)
        # This is GP-specific: we need to explicitly set training data
        self._set_train_data(X_tensor, y_tensor)

        # Rebuild MLL after setting training data
        self.mll = self._build_mll(self.likelihood, self.model)

        # Setup optimizer
        if optimizer is None:
            optimizer = torch.optim.Adam(
                list(self.model.parameters()) + list(self.likelihood.parameters()),
                lr=learning_rate,
            )

        # Setup learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=10
        )

        history = {"train_loss": []}
        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        best_likelihood_state = None

        # Training loop
        use_progress_bar = verbose > 0
        epoch_range = range(epochs)
        pbar = None
        if use_progress_bar:
            pbar = tqdm(epoch_range, desc="Training GP", unit="epoch")
            epoch_range = pbar

        self.model.train()
        self.likelihood.train()

        for epoch in epoch_range:
            optimizer.zero_grad()

            # Forward pass: compute negative marginal log likelihood
            # This is the GP-specific loss function
            loss = self._compute_loss(X_tensor, y_tensor)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.likelihood.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            loss_value = loss.item()
            history["train_loss"].append(loss_value)
            scheduler.step(loss_value)

            # Early stopping
            if loss_value < best_loss - 1e-4:
                best_loss = loss_value
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                best_likelihood_state = self.likelihood.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose > 0:
                        if pbar is not None:
                            pbar.set_postfix_str(f"Early stopping at epoch {epoch + 1}")
                        self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            # Update progress bar
            if pbar is not None:
                pbar.set_postfix_str(f"Loss: {loss_value:.4f}")
            elif verbose > 0 and (epoch + 1) % max(1, epochs // 10) == 0:
                self.logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {loss_value:.4f}")

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.likelihood.load_state_dict(best_likelihood_state)

        self.is_fitted = True

        return history

    def predict(
        self,
        X: np.ndarray,
        batch_size: Optional[int] = None,
        return_std: bool = False,
        verbose: int = 1,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions with the Gaussian Process model.

        GP models return distributions, so predictions include uncertainty
        estimates (standard deviation) by default.

        Parameters
        ----------
        X : np.ndarray
            Input data with shape (n_samples, n_features).
        batch_size : int, optional
            Batch size for prediction. If None, processes all at once.
            Default is None.
        return_std : bool, optional
            If True, returns both mean and standard deviation.
            Default is False.
        verbose : int, optional
            Verbosity level. Default is 1.

        Returns
        -------
        np.ndarray or tuple
            If return_std=False: predictions (mean) with shape (n_samples,).
            If return_std=True: tuple of (mean, std) both with shape (n_samples,).

        Raises
        ------
        ValueError
            If model is not fitted.
        """

        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction.")

        self.model.eval()
        self.likelihood.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        # Process in batches if batch_size is specified
        if batch_size is None:
            batch_size = len(X)

        predictions = []
        stds = []

        n_batches = (len(X) + batch_size - 1) // batch_size
        batch_range = range(0, len(X), batch_size)

        if verbose > 0 and n_batches > 1:
            batch_range = tqdm(
                batch_range, desc="Predicting", unit="batch", total=n_batches
            )

        with (
            torch.no_grad(),
            gpytorch.settings.fast_pred_var(),
            gpytorch.settings.cholesky_jitter(1e-1),
        ):
            for i in batch_range:
                batch_X = X_tensor[i : i + batch_size]
                pred_dist = self._predict_batch(batch_X)
                predictions.append(pred_dist.mean.cpu().numpy())
                if return_std:
                    stds.append(pred_dist.stddev.cpu().numpy())

        mean_pred = np.concatenate(predictions, axis=0)

        if return_std:
            std_pred = np.concatenate(stds, axis=0)
            return mean_pred, std_pred
        else:
            return mean_pred

    def _set_train_data(self, X: torch.Tensor, y: torch.Tensor):
        """
        Set training data for the GP model.

        This is GP-specific: GP models need explicit training data setting.

        Parameters
        ----------
        X : torch.Tensor
            Training inputs.
        y : torch.Tensor
            Training targets.
        """

        if hasattr(self.model, "set_train_data"):
            self.model.set_train_data(X, y, strict=False)
        else:
            raise AttributeError(
                f"Model {type(self.model)} does not support set_train_data(). "
                "This is required for GP models."
            )

    def _build_mll(
        self,
        likelihood: gpytorch.likelihoods.Likelihood,
        model: gpytorch.models.GP,
    ) -> gpytorch.mlls.MarginalLogLikelihood:
        """
        Build the marginal log likelihood objective.

        Parameters
        ----------
        likelihood : gpytorch.likelihoods.Likelihood
            The likelihood module.
        model : gpytorch.models.GP
            The GP model.

        Returns
        -------
        gpytorch.mlls.MarginalLogLikelihood
            The MLL objective.
        """

        return ExactMarginalLogLikelihood(likelihood, model)

    def _compute_loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the training loss (negative MLL).

        Parameters
        ----------
        X : torch.Tensor
            Training inputs.
        y : torch.Tensor
            Training targets.

        Returns
        -------
        torch.Tensor
            Negative marginal log likelihood.
        """

        with gpytorch.settings.cholesky_jitter(1e-1):
            output = self.model(X)
            loss = -self.mll(output, y)

        return loss

    def _predict_batch(self, X: torch.Tensor) -> gpytorch.distributions.Distribution:
        """
        Make predictions for a batch of inputs.

        Parameters
        ----------
        X : torch.Tensor
            Input batch.

        Returns
        -------
        gpytorch.distributions.Distribution
            Predictive distribution.
        """

        return self.likelihood(self.model(X))

    def save_pytorch_model(self, model_path: str, **kwargs):
        """
        Save the GP model to a file.

        GP models require saving both the model and likelihood state dicts.

        Parameters
        ----------
        model_path : str
            Path to the file where the model will be saved.
        **kwargs
            Additional arguments for torch.save.
        """

        if self.model is None or self.likelihood is None:
            raise ValueError("Model must be built before saving.")

        # Get model-specific metadata
        metadata = self._get_model_metadata()

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "likelihood_state_dict": self.likelihood.state_dict(),
                "is_fitted": self.is_fitted,
                "model_class": self.__class__.__name__,
                **metadata,
            },
            model_path,
            **kwargs,
        )
        self.logger.info(f"GP model saved to {model_path}")

    def load_pytorch_model(self, model_path: str, **kwargs):
        """
        Load a GP model from a file.

        Parameters
        ----------
        model_path : str
            Path to the file where the model is saved.
        **kwargs
            Additional arguments for torch.load.
        """

        checkpoint = torch.load(model_path, **kwargs)

        # Restore model-specific attributes
        self._restore_model_metadata(checkpoint)

        # Build model first if needed
        if self.model is None:
            # Need input shape to build model - use dummy data
            # In practice, you should save/load the training data shape
            dummy_shape = (10, 10)  # Default, user should provide actual shape
            self.model = self._build_model(dummy_shape)
            # Initialize likelihood if not set (should be set by _build_model, but check anyway)
            if self.likelihood is None:
                self.likelihood = GaussianLikelihood().to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.likelihood.load_state_dict(checkpoint["likelihood_state_dict"])
        self.is_fitted = checkpoint.get("is_fitted", False)
        self.logger.info(f"GP model loaded from {model_path}")

    def _get_model_metadata(self) -> Dict:
        """
        Get model-specific metadata for saving.

        Override this method in subclasses to save additional metadata.

        Returns
        -------
        Dict
            Metadata dictionary.
        """

        return {}

    def _restore_model_metadata(self, checkpoint: Dict):
        """
        Restore model-specific metadata from checkpoint.

        Override this method in subclasses to restore additional metadata.

        Parameters
        ----------
        checkpoint : Dict
            Checkpoint dictionary.
        """

        pass


class ExactGPModel(BaseGPRModel):
    """
    Exact Gaussian Process Regression model using GPyTorch.

    This model implements exact GP inference, suitable for datasets up to
    several thousand samples. For larger datasets, consider using approximate
    GP methods.

    Parameters
    ----------
    kernel : str, optional
        Type of kernel to use. Options: 'rbf', 'matern', 'rbf+matern'.
        Default is 'rbf+matern'.
    ard_num_dims : int, optional
        Number of input dimensions for ARD (Automatic Relevance Determination).
        If None, will be inferred from data. Default is None.
    device : str or torch.device, optional
        Device to run the model on. Default is None (auto-detect).
    **kwargs
        Additional keyword arguments passed to BaseGPRModel.

    Examples
    --------
    >>> import numpy as np
    >>> from bluemath_tk.deeplearning import ExactGPModel
    >>>
    >>> # Generate sample data
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randn(100)
    >>>
    >>> # Create and fit model
    >>> gp = ExactGPModel(kernel='rbf+matern')
    >>> history = gp.fit(X, y, epochs=100, learning_rate=0.1)
    >>>
    >>> # Make predictions
    >>> X_test = np.random.randn(50, 5)
    >>> y_pred, y_std = gp.predict(X_test, return_std=True)
    """

    def __init__(
        self,
        kernel: str = "rbf+matern",
        ard_num_dims: Optional[int] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.kernel_type = kernel.lower()
        self.ard_num_dims = ard_num_dims

    def _build_kernel(self, input_dim: int) -> Kernel:
        """
        Build the covariance kernel.
        """

        if self.ard_num_dims is None:
            ard_num_dims = input_dim
        else:
            ard_num_dims = self.ard_num_dims

        if self.kernel_type == "rbf":
            base_kernel = RBFKernel(ard_num_dims=ard_num_dims)
        elif self.kernel_type == "matern":
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
        elif self.kernel_type == "rbf+matern":
            base_kernel = RBFKernel(ard_num_dims=ard_num_dims) + MaternKernel(
                nu=2.5, ard_num_dims=ard_num_dims
            )
        else:
            raise ValueError(
                f"Unknown kernel type: {self.kernel_type}. "
                "Options: 'rbf', 'matern', 'rbf+matern'"
            )

        return ScaleKernel(base_kernel)

    def _build_model(self, input_shape: Tuple, **kwargs) -> ExactGP:
        """
        Build the GPyTorch ExactGP model.
        """

        if len(input_shape) == 1:
            input_dim = input_shape[0]
        else:
            input_dim = input_shape[-1]

        kernel = self._build_kernel(input_dim)

        class GPModel(ExactGP):
            def __init__(self, train_x, train_y, likelihood, kernel):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = ConstantMean()
                self.covar_module = kernel

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        # Create dummy data for initialization
        dummy_x = torch.randn(10, input_dim).to(self.device)
        dummy_y = torch.randn(10).to(self.device)

        # Initialize likelihood and model
        if self.likelihood is None:
            self.likelihood = GaussianLikelihood().to(self.device)
        model = GPModel(dummy_x, dummy_y, self.likelihood, kernel.to(self.device))

        return model.to(self.device)

    def _get_model_metadata(self) -> Dict:
        """
        Get model-specific metadata for saving.
        """

        return {
            "kernel_type": self.kernel_type,
            "ard_num_dims": self.ard_num_dims,
        }

    def _restore_model_metadata(self, checkpoint: Dict):
        """
        Restore model-specific metadata from checkpoint.
        """

        self.kernel_type = checkpoint.get("kernel_type", "rbf+matern")
        self.ard_num_dims = checkpoint.get("ard_num_dims", None)
