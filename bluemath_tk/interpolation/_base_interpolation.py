from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from ..core.models import BlueMathModel
from ..core.plotting.base_plotting import DefaultStaticPlotting


class BaseInterpolation(BlueMathModel):
    """
    Base class for all interpolation BlueMath models.
    This class provides the basic structure for all interpolation models.

    Methods
    -------
    fit(*args, **kwargs)
    predict(*args, **kwargs)
    fit_predict(*args, **kwargs)
    """

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Fits the model to the data.

        Parameters
        ----------
        *args : list
            Positional arguments.
        **kwargs : dict
            Keyword arguments.
        """

        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        Predicts the interpolated data given a dataset.

        Parameters
        ----------
        *args : list
            Positional arguments.
        **kwargs : dict
            Keyword arguments.
        """

        pass

    @abstractmethod
    def fit_predict(self, *args, **kwargs):
        """
        Fits the model to the subset and predicts the interpolated dataset.

        Parameters
        ----------
        *args : list
            Positional arguments.
        **kwargs : dict
            Keyword arguments.
        """

        pass

    def explain_with_mendezf(
        self,
        target_variable: str = None,
        dataset: pd.DataFrame | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        **kwargs,
    ) -> tuple[Figure, np.ndarray]:
        """
        Explain model predictions with scatter plots colored by target variable.

        Creates triangle scatter plots showing input feature relationships,
        with points colored by the predicted target variable values. This provides
        visual insight into how the target variable varies across the input space.

        Parameters
        ----------
        target_variable : str, optional
            The target variable to visualize. If None, uses the first target variable.
            Default is None.
        dataset : pd.DataFrame, optional
            Dataset to plot. If None, uses the training subset data.
            Default is None.
        vmin : float, optional
            Minimum value for color scale. If None, uses data minimum.
            Default is None.
        vmax : float, optional
            Maximum value for color scale. If None, uses data maximum.
            Default is None.
        **kwargs : dict, optional
            Additional keyword arguments for scatter plot (e.g., s, alpha, marker).

        Returns
        -------
        Tuple[Figure, np.ndarray]
            A tuple containing:
            - Figure object
            - 2D array of Axes objects

        Raises
        ------
        ValueError
            If the model is not fitted or target_variable is invalid.
        """

        if not self.is_fitted:
            raise ValueError("Model must be fitted before explaining.")

        # Select target variable
        if target_variable is None:
            target_variable = self.target_processed_variables[0]
        elif target_variable not in self.target_processed_variables:
            raise ValueError(
                f"target_variable '{target_variable}' not found in "
                f"target_processed_variables: {self.target_processed_variables}"
            )

        # Use provided dataset or training data
        if dataset is None:
            dataset = self._original_subset_data.copy()
        else:
            dataset = dataset.copy()

        # Ensure dataset has the same columns as training data
        if not all(
            col in dataset.columns for col in self._original_subset_data.columns
        ):
            raise ValueError(
                f"Dataset must contain the same columns as subset_data: "
                f"{self._original_subset_data.columns.tolist()}"
            )

        # Predict target variable for the dataset
        self.logger.info(f"Predicting {target_variable} for visualization dataset")
        try:
            predictions = self.predict(dataset=dataset, verbose=0)
        except TypeError:
            predictions = self.predict(dataset=dataset)

        target_values = predictions[target_variable].values

        # Get variable names from dataset
        variables_names = list(dataset.columns)
        num_variables = len(variables_names)

        if num_variables < 2:
            raise ValueError(
                "Dataset must have at least 2 variables for triangle plot."
            )

        # Create figure and axes in triangle arrangement
        default_static_plot = DefaultStaticPlotting()
        fig, axes = default_static_plot.get_subplots(
            nrows=num_variables - 1,
            ncols=num_variables - 1,
            sharex=False,
            sharey=False,
        )
        if isinstance(axes, Axes):
            axes = np.array([[axes]])
        elif axes.ndim == 1:
            axes = axes.reshape(-1, 1)

        # Set color scale limits
        if vmin is None:
            vmin = target_values.min()
        if vmax is None:
            vmax = target_values.max()

        # Create scatter plots in triangle arrangement
        # c1 indexes variables_names[1:] (x-axis variables)
        # c2 indexes variables_names[:-1] (y-axis variables)
        for c1, v1 in enumerate(variables_names[1:]):
            for c2, v2 in enumerate(variables_names[:-1]):
                if c1 == c2:
                    # Diagonal: set labels
                    default_static_plot.plot_scatter(
                        ax=axes[c2, c1],
                        x=dataset[v1],
                        y=dataset[v2],
                        c=target_values,
                        alpha=0.6,
                        cmap="bwr",
                        vmin=vmin,
                        vmax=vmax,
                        **kwargs,
                    )
                    axes[c2, c1].set_xlabel(variables_names[c1 + 1])
                    axes[c2, c1].set_ylabel(variables_names[c2])
                elif c1 > c2:
                    # Lower triangle: hide tick labels
                    default_static_plot.plot_scatter(
                        ax=axes[c2, c1],
                        x=dataset[v1],
                        y=dataset[v2],
                        c=target_values,
                        alpha=0.6,
                        cmap="bwr",
                        vmin=vmin,
                        vmax=vmax,
                        **kwargs,
                    )
                    axes[c2, c1].xaxis.set_ticklabels([])
                    axes[c2, c1].yaxis.set_ticklabels([])
                else:
                    # Upper triangle: remove axes
                    fig.delaxes(axes[c2, c1])

        fig.suptitle(f"Input Features Colored by {target_variable}", fontsize=14)

        # Create a custom axis for the colorbar at the bottom right
        cbar_ax = fig.add_axes([0.15, 0.05, 0.4, 0.02])
        # Create colorbar with proper normalization
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap="bwr", norm=norm)

        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(target_variable, fontsize=12, fontweight="bold")
        cbar.ax.tick_params(labelsize=10)

        plt.tight_layout()
        plt.show()

        return fig, axes

    def explain_with_shap(
        self,
        dataset: pd.DataFrame,
        target_variable: str = None,
        num_samples: int = 100,
        max_background_samples: int = 100,
    ) -> None:
        """
        Explain model predictions using SHAP (SHapley Additive exPlanations) values.

        This method provides comprehensive model interpretability by automatically
        generating interactive SHAP visualizations for each target variable. It uses
        the training subset data as background.

        Parameters
        ----------
        dataset : pd.DataFrame
            The test dataset to explain predictions for. Must have the same variables
            as the subset_data used for fitting.
        target_variable : str, optional
            The target variable to explain. If None, explains all target variables.
            Default is None.
        num_samples : int, optional
            Number of samples to use for SHAP approximation. Higher values give
            more accurate results but are slower. Default is 100.
            Recommended: 100-500 for good balance between speed and accuracy.
        max_background_samples : int, optional
            Maximum number of background samples to use. The subset data will be
            automatically summarized using k-means if it exceeds this value.
            Default is 100.

        Raises
        ------
        ImportError
            If SHAP is not installed.
        ValueError
            If the model is not fitted or target_variable is invalid.
        """

        try:
            import logging

            import shap

            # Suppress SHAP INFO logs (keep progress bars)
            shap_logger = logging.getLogger("shap")
            shap_logger.setLevel(logging.WARNING)

            shap.initjs()  # Initialize JavaScript for interactive plots
        except ImportError:
            raise ImportError(
                "SHAP is required for explain method. Install with: pip install shap"
            )

        if not self.is_fitted:
            raise ValueError("Model must be fitted before explaining.")

        # Determine which target variables to explain
        if target_variable is None:
            target_vars = self.target_processed_variables
        else:
            if target_variable not in self.target_processed_variables:
                raise ValueError(
                    f"target_variable '{target_variable}' not found in "
                    f"target_processed_variables: {self.target_processed_variables}"
                )
            target_vars = [target_variable]

        # Prepare background data from subset (raw data, not preprocessed)
        # SHAP will normalize it internally, and predict will handle preprocessing
        background = self._original_subset_data.copy()

        # Summarize background data for efficiency if it's too large
        if len(background) > max_background_samples:
            self.logger.info(
                f"Summarizing background data from {len(background)} "
                f"to {max_background_samples} samples using k-means"
            )
            n_clusters = min(max_background_samples, len(background))
            background_summary = shap.kmeans(background.values, n_clusters)
        else:
            n_clusters = len(background)
            background_summary = background.values

        for target_var in target_vars:
            self.logger.info(
                f"Explaining predictions for target variable: {target_var}"
            )

            # Create a prediction function for this specific target variable
            # SHAP normalizes the background internally, so X is normalized
            # We convert back to DataFrame with original column names (matching
            # subset_data), then predict handles preprocessing
            def predict_fn(X):
                """
                Predict the target variable for SHAP explanation.

                Parameters
                ----------
                X : np.ndarray
                    Input features normalized by SHAP (shape: n_samples, n_features)

                Returns
                -------
                np.ndarray
                    Predictions for the target variable (shape: n_samples,)
                """
                # Convert normalized array to DataFrame with original column names
                # (matching self._original_subset_data.columns, not processed columns)
                # SHAP normalizes based on background, so X is in normalized space
                # but we need original column structure for predict
                dataset_df = pd.DataFrame(X, columns=self._original_subset_data.columns)

                # Use predict to get all target variables, then extract the one we want
                # This handles preprocessing internally (Dir -> Dir_u/Dir_v, normalize)
                # and returns denormalized values
                # Try to call predict with verbose=0, fallback if not supported
                try:
                    predictions = self.predict(dataset=dataset_df, verbose=0)
                except TypeError:
                    # Some models might not support verbose parameter
                    predictions = self.predict(dataset=dataset_df)
                return predictions[target_var].values

            # Create SHAP explainer
            self.logger.info(
                f"Creating SHAP KernelExplainer with {n_clusters} "
                f"background samples and {num_samples} evaluation samples"
            )
            explainer = shap.KernelExplainer(predict_fn, background_summary)

            # Calculate SHAP values using original dataset
            # SHAP will normalize internally, but we use original for plotting
            self.logger.info(f"Calculating SHAP values for {len(dataset)} samples...")
            shap_values = explainer.shap_values(dataset.values, nsamples=num_samples)

            # Ensure shap_values is 2D (handle both single and multiple samples)
            shap_values = np.array(shap_values)
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)

            # Generate SHAP summary plot using original dataset (good magnitudes)
            self.logger.info(f"Generating SHAP summary plot for {target_var}")
            shap.summary_plot(shap_values, dataset, show=True)


class InterpolationComparator:
    """
    Class for comparing interpolation models.
    """

    def __init__(self, list_of_models: list[BaseInterpolation]) -> None:
        """
        Initialize the InterpolationComparator class.
        """

        self.list_of_models = list_of_models

    def fit(
        self,
        subset_data: pd.DataFrame,
        target_data: pd.DataFrame,
    ) -> None:
        """
        Fits the clustering models.
        """

        for model in self.list_of_models:
            model.fit(
                subset_data=subset_data,
                target_data=target_data,
            )
