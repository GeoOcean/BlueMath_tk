"""
Package: BlueMath_tk
Module: interpolation
File: gps.py
Author: GeoOcean Research Group, Universidad de Cantabria
Repository: https://github.com/GeoOcean/BlueMath_tk.git
Status: Under development (Working)

Gaussian Process interpolation models using GPyTorch.

This module provides Gaussian Process interpolation following the same
interface pattern as RBF interpolation, with support for:
- Multiple target variables
- Directional variables (wind direction, wave direction, etc.)
- Data normalization
- Uncertainty quantification

References
----------
1. Wang, Z., Leung, M., Mukhopadhyay, S., et al. (2024).
   "A hybrid statistical–dynamical framework for compound coastal flooding analysis."
   *Environmental Research Letters*, 20(1), 014005.
2. Wang, Z., Leung, M., Mukhopadhyay, S., et al. (2025).
   "Compound coastal flooding in San Francisco Bay under climate change."
   *npj Natural Hazards*, 2(1), 3.
3. GPyTorch Documentation: https://docs.gpytorch.ai/
"""

import gpytorch
import numpy as np
import pandas as pd
import torch
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from tqdm import tqdm

from ..core.decorators import validate_gp_data
from ._base_interpolation import BaseInterpolation


class GPError(Exception):
    """
    Custom exception for Gaussian Process interpolation model.
    """

    def __init__(self, message: str = "GP error occurred."):
        self.message = message
        super().__init__(self.message)


class ExactGPInterpolation(BaseInterpolation):
    """
    Exact Gaussian Process interpolation model using GPyTorch.

    This model implements exact GP inference for interpolation tasks,
    following the same interface pattern as RBF interpolation. Suitable
    for datasets up to several thousand samples.

    Examples
    --------
    .. jupyter-execute::

        import numpy as np
        import pandas as pd
        from bluemath_tk.interpolation.gps import ExactGPInterpolation

        dataset = pd.DataFrame({
            "Hs": np.random.rand(1000) * 7,
            "Tp": np.random.rand(1000) * 20,
            "Dir": np.random.rand(1000) * 360,
        })
        subset = dataset.sample(frac=0.25)
        target = pd.DataFrame({
            "HsPred": subset["Hs"] * 2 + subset["Tp"] * 3,
            "DirPred": -subset["Dir"],
        })

        gp = ExactGPInterpolation(kernel='rbf+matern')
        predictions = gp.fit_predict(
            subset_data=subset,
            subset_directional_variables=["Dir"],
            target_data=target,
            target_directional_variables=["DirPred"],
            normalize_target_data=True,
            dataset=dataset,
        )
        print(predictions.head())

    References
    ----------
    [1] https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html
    [2] Rasmussen, C. E., & Williams, C. K. I. (2006).
        Gaussian Processes for Machine Learning. MIT Press.
    """

    def __init__(
        self,
        kernel: str = "rbf+matern",
        ard_num_dims: int | None = None,
        device: str | torch.device | None = None,
        epochs: int = 200,
        learning_rate: float = 0.1,
        patience: int = 30,
    ):
        """
        Initialize Exact GP interpolation model.

        Parameters
        ----------
        kernel : str, optional
            Type of kernel to use. Options: 'rbf', 'matern', 'rbf+matern'.
            Default is 'rbf+matern'.
        ard_num_dims : int, optional
            Number of input dimensions for ARD. If None, inferred from data.
            Default is None.
        device : str or torch.device, optional
            Device to run the model on. Default is None (auto-detect).
        epochs : int, optional
            Maximum number of training epochs. Default is 200.
        learning_rate : float, optional
            Learning rate for optimizer. Default is 0.1.
        patience : int, optional
            Early stopping patience. Default is 30.
        """

        super().__init__()
        self.set_logger_name(name=self.__class__.__name__)

        # Validate and store parameters
        if kernel.lower() not in ["rbf", "matern", "rbf+matern"]:
            raise ValueError(
                f"kernel must be one of ['rbf', 'matern', 'rbf+matern'], got {kernel}"
            )
        self._kernel = kernel.lower()
        self._ard_num_dims = ard_num_dims
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._patience = patience

        # Device management
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # GP-specific attributes - store models per target variable
        self._models: dict[str, gpytorch.models.GP] = {}
        self._likelihoods: dict[str, gpytorch.likelihoods.Likelihood] = {}
        self._mlls: dict[str, gpytorch.mlls.MarginalLogLikelihood] = {}

        # Interpolation-specific attributes (similar to RBF)
        self.is_fitted: bool = False
        self.is_target_normalized: bool = False
        self._original_subset_data: pd.DataFrame = pd.DataFrame()
        self._subset_data: pd.DataFrame = pd.DataFrame()
        self._normalized_subset_data: pd.DataFrame = pd.DataFrame()
        self._target_data: pd.DataFrame = pd.DataFrame()
        self._normalized_target_data: pd.DataFrame = pd.DataFrame()
        self._subset_directional_variables: list[str] = []
        self._target_directional_variables: list[str] = []
        self._subset_processed_variables: list[str] = []
        self._target_processed_variables: list[str] = []
        self._subset_custom_scale_factor: dict = {}
        self._target_custom_scale_factor: dict = {}
        self._subset_scale_factor: dict = {}
        self._target_scale_factor: dict = {}
        self._hyperparameters: dict[str, dict] = {}  # Store hyperparams per target var

        # Exclude from pickling
        self._exclude_attributes = ["_models", "_likelihoods", "_mlls"]

        initial_msg = f"""
        ---------------------------------------------------------------------------------
        | Initializing Exact GP interpolation model with the following parameters:
        |    - kernel: {self._kernel}
        |    - ard_num_dims: {self._ard_num_dims}
        |    - device: {self.device}
        |    - epochs: {self._epochs}
        |    - learning_rate: {self._learning_rate}
        |    - patience: {self._patience}
        | For more information, please refer to the documentation.
        ---------------------------------------------------------------------------------
        """
        self.logger.info(initial_msg)

    @property
    def kernel(self) -> str:
        """Return the kernel name."""
        return self._kernel

    @property
    def ard_num_dims(self) -> int | None:
        """Return the ARD number of dimensions."""
        return self._ard_num_dims

    @property
    def subset_data(self) -> pd.DataFrame:
        """Return the subset data."""
        return self._subset_data

    @property
    def normalized_subset_data(self) -> pd.DataFrame:
        """Return the normalized subset data."""
        return self._normalized_subset_data

    @property
    def target_data(self) -> pd.DataFrame:
        """Return the target data."""
        return self._target_data

    @property
    def normalized_target_data(self) -> pd.DataFrame:
        """Return the normalized target data."""
        if self._normalized_target_data.empty:
            raise ValueError("Target data is not normalized.")
        return self._normalized_target_data

    @property
    def subset_directional_variables(self) -> list[str]:
        """Return the subset directional variables."""
        return self._subset_directional_variables

    @property
    def target_directional_variables(self) -> list[str]:
        """Return the target directional variables."""
        return self._target_directional_variables

    @property
    def subset_processed_variables(self) -> list[str]:
        """Return the subset processed variables."""
        return self._subset_processed_variables

    @property
    def target_processed_variables(self) -> list[str]:
        """Return the target processed variables."""
        return self._target_processed_variables

    @property
    def hyperparameters(self) -> dict[str, dict]:
        """
        Return the learned hyperparameters for each target variable.

        Returns
        -------
        dict
            Dictionary mapping target variable names to their hyperparameters.
            Each hyperparameter dict contains:
            - 'lengthscale': list of lengthscales (one per input dimension if ARD)
            - 'lengthscales': list of dicts for additive kernels (rbf+matern),
              each with 'kernel_0', 'kernel_1', etc. keys
            - 'outputscale': float, output scale from ScaleKernel
            - 'noise': float, noise variance from likelihood
            - 'mean_constant': float, mean constant from ConstantMean

        Notes
        -----
        Hyperparameters are extracted after training. For additive kernels
        (rbf+matern), lengthscales are stored per sub-kernel.
        """
        return self._hyperparameters

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
        if self._ard_num_dims is None:
            ard_num_dims = input_dim
        else:
            ard_num_dims = self._ard_num_dims

        if self._kernel == "rbf":
            base_kernel = RBFKernel(ard_num_dims=ard_num_dims)
        elif self._kernel == "matern":
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
        elif self._kernel == "rbf+matern":
            base_kernel = RBFKernel(ard_num_dims=ard_num_dims) + MaternKernel(
                nu=2.5, ard_num_dims=ard_num_dims
            )
        else:
            raise ValueError(f"Unknown kernel type: {self._kernel}")

        return ScaleKernel(base_kernel)

    def _build_model(
        self, input_dim: int, train_x: torch.Tensor, train_y: torch.Tensor
    ) -> tuple[ExactGP, GaussianLikelihood]:
        """
        Build the GPyTorch ExactGP model.

        Parameters
        ----------
        input_dim : int
            Number of input dimensions.
        train_x : torch.Tensor
            Training input data.
        train_y : torch.Tensor
            Training target data.

        Returns
        -------
        tuple
            (GP model, likelihood)
        """
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

        # Initialize likelihood with very small noise for exact interpolation
        # Use a small fixed value (1e-6) for numerical stability while
        # maintaining near-exact interpolation at training points (like RBF)
        noise_lower = np.finfo(float).eps
        noise_constraint = GreaterThan(lower_bound=noise_lower)
        likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(
            self.device
        )
        # Initialize noise to a very small value for exact interpolation
        # This ensures predictions at training points match observed values
        initial_noise = 1e-6
        with torch.no_grad():
            likelihood.noise = torch.tensor(
                initial_noise, device=self.device, dtype=torch.float32
            )
        model = GPModel(train_x, train_y, likelihood, kernel.to(self.device)).to(
            self.device
        )

        return model, likelihood

    def _preprocess_subset_data(
        self, subset_data: pd.DataFrame, is_fit: bool = True
    ) -> pd.DataFrame:
        """
        Preprocess the subset data (similar to RBF pattern).

        Parameters
        ----------
        subset_data : pd.DataFrame
            The subset data to preprocess (could be a dataset to predict).
        is_fit : bool, optional
            Whether the data is being fit or not. Default is True.

        Returns
        -------
        pd.DataFrame
            The preprocessed subset data.
        """

        subset_data = subset_data.copy()

        self.logger.info("Checking for NaNs in subset data")
        subset_data = self.check_nans(data=subset_data, raise_error=True)

        self.logger.info("Preprocessing subset data")
        for directional_variable in self._subset_directional_variables:
            var_u_component, var_v_component = self.get_uv_components(
                x_deg=subset_data[directional_variable].values
            )
            subset_data[f"{directional_variable}_u"] = var_u_component
            subset_data[f"{directional_variable}_v"] = var_v_component
            subset_data.drop(columns=[directional_variable], inplace=True)

        self.logger.info("Normalizing subset data")
        normalized_subset_data, subset_scale_factor = self.normalize(
            data=subset_data,
            custom_scale_factor=self._subset_custom_scale_factor
            if is_fit
            else self._subset_scale_factor,
        )

        if is_fit:
            self._subset_data = subset_data
            self._subset_processed_variables = list(subset_data.columns)
            self._normalized_subset_data = normalized_subset_data
            self._subset_scale_factor = subset_scale_factor
        else:
            normalized_subset_data = normalized_subset_data[
                self._subset_processed_variables
            ]

        self.logger.info("Subset data preprocessed successfully")

        return normalized_subset_data.copy()

    def _preprocess_target_data(
        self,
        target_data: pd.DataFrame,
        normalize_target_data: bool = True,
    ) -> pd.DataFrame:
        """
        Preprocess the target data (similar to RBF pattern).

        Parameters
        ----------
        target_data : pd.DataFrame
            The target data to preprocess.
        normalize_target_data : bool, optional
            Whether to normalize the target data. Default is True.

        Returns
        -------
        pd.DataFrame
            The preprocessed target data.
        """

        target_data = target_data.copy()

        self.logger.info("Checking for NaNs in target data")
        target_data = self.check_nans(data=target_data, raise_error=True)

        self.logger.info("Preprocessing target data")
        for directional_variable in self._target_directional_variables:
            var_u_component, var_v_component = self.get_uv_components(
                x_deg=target_data[directional_variable].values
            )
            target_data[f"{directional_variable}_u"] = var_u_component
            target_data[f"{directional_variable}_v"] = var_v_component
            target_data.drop(columns=[directional_variable], inplace=True)
        self._target_processed_variables = list(target_data.columns)

        if normalize_target_data:
            self.logger.info("Normalizing target data")
            normalized_target_data, target_scale_factor = self.normalize(
                data=target_data,
                custom_scale_factor=self._target_custom_scale_factor,
            )
            self.is_target_normalized = True
            self._target_data = target_data.copy()
            self._normalized_target_data = normalized_target_data.copy()
            self._target_scale_factor = target_scale_factor.copy()
            self.logger.info("Target data preprocessed successfully")
            return normalized_target_data.copy()
        else:
            self.is_target_normalized = False
            self._target_data = target_data.copy()
            self._normalized_target_data = pd.DataFrame()
            self._target_scale_factor = {}
            self.logger.info("Target data preprocessed successfully")
            return target_data.copy()

    @validate_gp_data
    def fit(
        self,
        subset_data: pd.DataFrame,
        target_data: pd.DataFrame,
        subset_directional_variables: list[str] = [],
        target_directional_variables: list[str] = [],
        subset_custom_scale_factor: dict = {},
        normalize_target_data: bool = True,
        target_custom_scale_factor: dict = {},
        verbose: int = 1,
    ) -> None:
        """
        Fit the GP model to the data.

        Parameters
        ----------
        subset_data : pd.DataFrame
            The subset data used to fit the model.
        target_data : pd.DataFrame
            The target data used to fit the model.
        subset_directional_variables : List[str], optional
            The subset directional variables. Default is [].
        target_directional_variables : List[str], optional
            The target directional variables. Default is [].
        subset_custom_scale_factor : dict, optional
            The custom scale factor for the subset data. Default is {}.
        normalize_target_data : bool, optional
            Whether to normalize the target data. Default is True.
        target_custom_scale_factor : dict, optional
            The custom scale factor for the target data. Default is {}.
        verbose : int, optional
            Verbosity level. Default is 1.
        """

        self._subset_directional_variables = subset_directional_variables
        self._target_directional_variables = target_directional_variables
        self._subset_custom_scale_factor = subset_custom_scale_factor
        self._target_custom_scale_factor = target_custom_scale_factor

        # Store original subset data before preprocessing
        # (for explain, plot_partial_dependence, etc.)
        self._original_subset_data = subset_data.copy()

        # Preprocess data
        normalized_subset = self._preprocess_subset_data(subset_data=subset_data)
        normalized_target = self._preprocess_target_data(
            target_data=target_data,
            normalize_target_data=normalize_target_data,
        )

        # Convert to tensors
        X_tensor = torch.FloatTensor(normalized_subset.values).to(self.device)

        # Fit GP model for each target variable
        self.logger.info("Fitting GP models for each target variable")

        for target_var in normalized_target.columns:
            self.logger.info(f"Fitting GP for target variable: {target_var}")
            y_tensor = torch.FloatTensor(normalized_target[target_var].values).to(
                self.device
            )

            # Build model
            input_dim = normalized_subset.shape[1]
            model, likelihood = self._build_model(input_dim, X_tensor, y_tensor)

            # Set training data
            model.set_train_data(X_tensor, y_tensor, strict=False)
            mll = ExactMarginalLogLikelihood(likelihood, model)

            # Setup optimizer
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(likelihood.parameters()),
                lr=self._learning_rate,
            )

            # Setup learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.8, patience=10
            )

            # Training loop
            model.train()
            likelihood.train()

            best_loss = float("inf")
            patience_counter = 0
            best_model_state = None
            best_likelihood_state = None

            use_progress_bar = verbose > 0
            epoch_range = range(self._epochs)
            if use_progress_bar:
                epoch_range = tqdm(
                    epoch_range,
                    desc=f"Training GP for {target_var}",
                    unit="epoch",
                )

            for epoch in epoch_range:
                optimizer.zero_grad()

                with gpytorch.settings.cholesky_jitter(1e-1):
                    output = model(X_tensor)
                    loss = -mll(output, y_tensor)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(likelihood.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()

                # Keep noise small for exact interpolation
                # Clip noise to stay within bounds for numerical stability
                with torch.no_grad():
                    if hasattr(likelihood, "noise"):
                        noise_value = likelihood.noise.item()
                        if noise_value > 1e-5:
                            likelihood.noise.data.clamp_(
                                min=np.finfo(float).eps, max=1e-5
                            )

                loss_value = loss.item()
                scheduler.step(loss_value)

                # Early stopping
                if loss_value < best_loss - 1e-4:
                    best_loss = loss_value
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                    best_likelihood_state = likelihood.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= self._patience:
                        if verbose > 0:
                            self.logger.info(
                                f"Early stopping at epoch {epoch + 1} for {target_var}"
                            )
                        break

                # Update progress bar
                if use_progress_bar and isinstance(epoch_range, tqdm):
                    epoch_range.set_postfix_str(f"Loss: {loss_value:.4f}")
                elif verbose > 0 and (epoch + 1) % max(1, self._epochs // 10) == 0:
                    self.logger.info(
                        f"Epoch {epoch + 1}/{self._epochs} - Loss: {loss_value:.4f}"
                    )

            # Restore best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                likelihood.load_state_dict(best_likelihood_state)

            # Store model, likelihood, and mll for this target variable
            self._models[target_var] = model
            self._likelihoods[target_var] = likelihood
            self._mlls[target_var] = mll

            # Extract and store hyperparameters
            hyperparams = {}

            # Get lengthscales from kernel
            covar_module = model.covar_module
            if hasattr(covar_module, "base_kernel"):
                base_kernel = covar_module.base_kernel
                # Handle additive kernels (rbf+matern)
                if hasattr(base_kernel, "kernels"):
                    # Additive kernel - extract from each sub-kernel
                    lengthscales = []
                    for i, sub_kernel in enumerate(base_kernel.kernels):
                        if hasattr(sub_kernel, "lengthscale"):
                            ls = sub_kernel.lengthscale.detach().cpu()
                            lengthscales.append({f"kernel_{i}": ls.flatten().tolist()})
                    hyperparams["lengthscales"] = lengthscales
                else:
                    # Single kernel
                    if hasattr(base_kernel, "lengthscale"):
                        lengthscale = base_kernel.lengthscale.detach().cpu()
                        hyperparams["lengthscale"] = lengthscale.flatten().tolist()

            # Get output scale
            if hasattr(covar_module, "outputscale"):
                outputscale = covar_module.outputscale.detach().cpu().item()
                hyperparams["outputscale"] = outputscale

            # Get noise variance
            if hasattr(likelihood, "noise"):
                noise = likelihood.noise.detach().cpu().item()
                hyperparams["noise"] = noise

            # Get mean constant
            if hasattr(model.mean_module, "constant"):
                mean_const = model.mean_module.constant.detach().cpu().item()
                hyperparams["mean_constant"] = mean_const

            self._hyperparameters[target_var] = hyperparams

        self.is_fitted = True
        self.logger.info("GP models fitted successfully")

    def predict(
        self,
        dataset: pd.DataFrame,
        return_std: bool = False,
        verbose: int = 1,
    ) -> pd.DataFrame:
        """
        Predict using the fitted GP model.

        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset to predict (must have same variables as subset).
        return_std : bool, optional
            If True, returns standard deviations. Default is False.
        verbose : int, optional
            Verbosity level. Default is 1.

        Returns
        -------
        pd.DataFrame
            The predicted dataset. If return_std=True, includes columns
            with '_std' suffix for standard deviations.

        Raises
        ------
        GPError
            If the model is not fitted.
        """

        if not self.is_fitted:
            raise GPError("GP model must be fitted before predicting.")

        self.logger.info("Preprocessing dataset for prediction")
        normalized_dataset = self._preprocess_subset_data(
            subset_data=dataset, is_fit=False
        )

        X_tensor = torch.FloatTensor(normalized_dataset.values).to(self.device)

        # Predict for each target variable
        predictions_dict = {}
        stds_dict = {}

        for target_var in self._target_processed_variables:
            if verbose > 0:
                self.logger.info(f"Predicting target variable: {target_var}")

            model = self._models[target_var]
            likelihood = self._likelihoods[target_var]

            model.eval()
            likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred_dist = likelihood(model(X_tensor))
                predictions_dict[target_var] = pred_dist.mean.cpu().numpy()
                if return_std:
                    # stds_dict[f"{target_var}_std"] = pred_dist.stddev.cpu().numpy()
                    # self._target_scale_factor[f"{target_var}_std"] = (
                    #     self._target_scale_factor[target_var]
                    # )
                    (
                        stds_dict[f"{target_var}_lower_ci"],
                        stds_dict[f"{target_var}_upper_ci"],
                    ) = (
                        pred_dist.confidence_region()[0].cpu().numpy(),
                        pred_dist.confidence_region()[1].cpu().numpy(),
                    )
                    self._target_scale_factor[f"{target_var}_lower_ci"] = (
                        self._target_scale_factor[target_var]
                    )
                    self._target_scale_factor[f"{target_var}_upper_ci"] = (
                        self._target_scale_factor[target_var]
                    )

        # Convert to DataFrame
        result = pd.DataFrame(predictions_dict)

        if return_std:
            std_df = pd.DataFrame(stds_dict)
            result = pd.concat([result, std_df], axis=1)

        # Denormalize if needed
        if self.is_target_normalized:
            self.logger.info("Denormalizing target data")
            result = self.denormalize(
                normalized_data=result,
                scale_factor=self._target_scale_factor,
            )

        # Reconstruct directional variables
        for directional_variable in self._target_directional_variables:
            self.logger.info(f"Calculating target degrees for {directional_variable}")
            result[directional_variable] = self.get_degrees_from_uv(
                xu=result[f"{directional_variable}_u"].values,
                xv=result[f"{directional_variable}_v"].values,
            )

        return result

    def fit_predict(
        self,
        subset_data: pd.DataFrame,
        target_data: pd.DataFrame,
        dataset: pd.DataFrame,
        subset_directional_variables: list[str] = [],
        target_directional_variables: list[str] = [],
        subset_custom_scale_factor: dict = {},
        normalize_target_data: bool = True,
        target_custom_scale_factor: dict = {},
        return_std: bool = False,
        verbose: int = 1,
    ) -> pd.DataFrame:
        """
        Fit the model and predict in one step.

        Parameters
        ----------
        subset_data : pd.DataFrame
            The subset data used to fit the model.
        target_data : pd.DataFrame
            The target data used to fit the model.
        dataset : pd.DataFrame
            The dataset to predict (must have same variables as subset).
        subset_directional_variables : List[str], optional
            The subset directional variables. Default is [].
        target_directional_variables : List[str], optional
            The target directional variables. Default is [].
        subset_custom_scale_factor : dict, optional
            The custom scale factor for the subset data. Default is {}.
        normalize_target_data : bool, optional
            Whether to normalize the target data. Default is True.
        target_custom_scale_factor : dict, optional
            The custom scale factor for the target data. Default is {}.
        return_std : bool, optional
            If True, returns standard deviations. Default is False.
        verbose : int, optional
            Verbosity level. Default is 1.

        Returns
        -------
        pd.DataFrame
            The interpolated dataset.
        """

        self.fit(
            subset_data=subset_data,
            target_data=target_data,
            subset_directional_variables=subset_directional_variables,
            target_directional_variables=target_directional_variables,
            subset_custom_scale_factor=subset_custom_scale_factor,
            normalize_target_data=normalize_target_data,
            target_custom_scale_factor=target_custom_scale_factor,
            verbose=verbose,
        )

        return self.predict(dataset=dataset, return_std=return_std, verbose=verbose)
