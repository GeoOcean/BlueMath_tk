from typing import List
import pandas as pd
from scipy.interpolate import RBFInterpolator
from ._base_interpolation import BaseInterpolation
from ..core.decorators import validate_data_rbf


class RBFError(Exception):
    """
    Custom exception for RBF class.
    """

    def __init__(self, message: str = "RBF error occurred."):
        self.message = message
        super().__init__(self.message)


class RBF(BaseInterpolation):
    """
    Radial Basis Function (RBF) interpolation model.

    Here, scipy's RBFInterpolator is used to interpolate the data.

    class RBFInterpolator(y, d, neighbors=None, smoothing=0.0, kernel='thin_plate_spline', epsilon=None, degree=None)
    """

    def __init__(
        self,
        neighbors: int = None,
        smoothing: float = 0.0,
        kernel: str = "gaussian",
        epsilon: float = None,
        degree: int = None,
    ):
        """
        Initializes the RBF model.
        """

        super().__init__()
        self.set_logger_name(name=self.__class__.__name__)
        if not isinstance(neighbors, int) and neighbors is not None:
            raise ValueError("neighbors must be an integer.")
        self._neighbors = neighbors
        if not isinstance(smoothing, float):
            raise ValueError("smoothing must be a float.")
        self._smoothing = smoothing
        if not isinstance(kernel, str):
            raise ValueError("kernel must be a string.")
        self._kernel = kernel
        if not isinstance(epsilon, float) and epsilon is not None:
            raise ValueError("epsilon must be a float.")
        self._epsilon = epsilon
        if not isinstance(degree, int) and degree is not None:
            raise ValueError("degree must be an integer.")
        self._degree = degree
        self._rbf: RBFInterpolator = None
        # Below, we initialize the attributes that will be set in the fit method
        self.is_fitted: bool = False
        self.is_target_normalized: bool = False
        self._subset_data: pd.DataFrame = pd.DataFrame()
        self._normalized_subset_data: pd.DataFrame = pd.DataFrame()
        self._target_data: pd.DataFrame = pd.DataFrame()
        self._normalized_target_data: pd.DataFrame = pd.DataFrame()
        self._subset_directional_variables: List[str] = []
        self._target_directional_variables: List[str] = []
        self._subset_processed_variables: List[str] = []
        self._target_processed_variables: List[str] = []
        self._subset_custom_scale_factor: dict = {}
        self._target_custom_scale_factor: dict = {}
        self._subset_scale_factor: dict = {}
        self._target_scale_factor: dict = {}
        self._rbf_coeffs: pd.DataFrame = pd.DataFrame()
        self._opt_sigmas: dict = {}

    @property
    def neighbors(self) -> int:
        return self._neighbors

    @property
    def smoothing(self) -> float:
        return self._smoothing

    @property
    def kernel(self) -> str:
        return self._kernel

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def rbf(self) -> RBFInterpolator:
        return self._rbf

    @property
    def subset_data(self) -> pd.DataFrame:
        return self._subset_data

    @property
    def normalized_subset_data(self) -> pd.DataFrame:
        return self._normalized_subset_data

    @property
    def target_data(self) -> pd.DataFrame:
        return self._target_data

    @property
    def normalized_target_data(self) -> pd.DataFrame:
        if self._normalized_target_data.empty:
            raise ValueError("Target data is not normalized.")
        return self._normalized_target_data

    @property
    def subset_directional_variables(self) -> List[str]:
        return self._subset_directional_variables

    @property
    def target_directional_variables(self) -> List[str]:
        return self._target_directional_variables

    @property
    def subset_processed_variables(self) -> List[str]:
        return self._subset_processed_variables

    @property
    def target_processed_variables(self) -> List[str]:
        return self._target_processed_variables

    @property
    def subset_custom_scale_factor(self) -> dict:
        return self._subset_custom_scale_factor

    @property
    def target_custom_scale_factor(self) -> dict:
        return self._target_custom_scale_factor

    @property
    def subset_scale_factor(self) -> dict:
        return self._subset_scale_factor

    @property
    def target_scale_factor(self) -> dict:
        return self._target_scale_factor

    @property
    def rbf_coeffs(self) -> pd.DataFrame:
        return self._rbf_coeffs

    @property
    def opt_sigmas(self) -> dict:
        return self._opt_sigmas

    def _preprocess_subset_data(
        self, subset_data: pd.DataFrame, is_fit: bool = True
    ) -> pd.DataFrame:
        """
        This function preprocesses the subset data.

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

        Raises
        ------
        ValueError
            If the subset contains NaNs.

        Notes
        -----
        - This function preprocesses the subset data by:
            - Checking for NaNs.
            - Preprocessing directional variables.
            - Normalizing the data.
        """

        # Make copies to avoid modifying the original data
        subset_data = subset_data.copy()

        self.logger.info("Checking for NaNs in subset data")
        subset_data = self.check_nans(data=subset_data, raise_error=True)

        self.logger.info("Preprocessing subset data")
        for directional_variable in self.subset_directional_variables:
            var_u_component, var_y_component = self.get_uv_components(
                x_deg=subset_data[directional_variable].values
            )
            subset_data[f"{directional_variable}_u"] = var_u_component
            subset_data[f"{directional_variable}_v"] = var_y_component
            # Drop the original directional variable in subset_data
            subset_data.drop(columns=[directional_variable], inplace=True)
        self._subset_processed_variables = list(subset_data.columns)

        self.logger.info("Normalizing subset data")
        normalized_subset_data, subset_scale_factor = self.normalize(
            data=subset_data,
            custom_scale_factor=self.subset_custom_scale_factor
            if is_fit
            else self.subset_scale_factor,
        )

        self.logger.info("Subset data preprocessed successfully")

        if is_fit:
            self._subset_data = subset_data
            self._normalized_subset_data = normalized_subset_data
            self._subset_scale_factor = subset_scale_factor

        return normalized_subset_data.copy()

    def _preprocess_target_data(
        self,
        target_data: pd.DataFrame,
        normalize_target_data: bool = True,
    ) -> pd.DataFrame:
        """
        This function preprocesses the target data.

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

        Raises
        ------
        ValueError
            If the target contains NaNs.

        Notes
        -----
        - This function preprocesses the target data by:
            - Checking for NaNs.
            - Preprocessing directional variables.
            - Normalizing the data.
        """

        # Make copies to avoid modifying the original data
        target_data = target_data.copy()

        self.logger.info("Checking for NaNs in target data")
        target_data = self.check_nans(data=target_data, raise_error=True)

        self.logger.info("Preprocessing target data")
        for directional_variable in self.target_directional_variables:
            var_u_component, var_y_component = self.get_uv_components(
                x_deg=target_data[directional_variable].values
            )
            target_data[f"{directional_variable}_u"] = var_u_component
            target_data[f"{directional_variable}_v"] = var_y_component
            # Drop the original directional variable in target_data
            target_data.drop(columns=[directional_variable], inplace=True)
        self._target_processed_variables = list(target_data.columns)

        if normalize_target_data:
            self.logger.info("Normalizing target data")
            normalized_target_data, target_scale_factor = self.normalize(
                data=target_data,
                custom_scale_factor=self.target_custom_scale_factor,
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

    @validate_data_rbf
    def fit(
        self,
        subset_data: pd.DataFrame,
        target_data: pd.DataFrame,
        subset_directional_variables: List[str] = [],
        target_directional_variables: List[str] = [],
        subset_custom_scale_factor: dict = {},
        normalize_target_data: bool = True,
        target_custom_scale_factor: dict = {},
        num_threads: int = None,
    ) -> None:
        """
        Fits the model to the data.

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
        num_threads : int, optional
            The number of threads to use for the optimization. Default is None.

        Notes
        -----
        - This function fits the RBF model to the data by:
            1. Preprocessing the subset and target data.
            2. Calculating the optimal sigma for the target variables.
            3. Storing the RBF coefficients and optimal sigmas.
        - The number of threads to use for the optimization can be specified.
        """

        if num_threads is not None:
            self.set_num_processors_to_use(num_processors=num_threads)
            self.logger.info(f"Using {num_threads} threads for optimization.")

        self._subset_directional_variables = subset_directional_variables
        self._target_directional_variables = target_directional_variables
        self._subset_custom_scale_factor = subset_custom_scale_factor
        self._target_custom_scale_factor = target_custom_scale_factor
        subset_data = self._preprocess_subset_data(subset_data=subset_data)
        target_data = self._preprocess_target_data(
            target_data=target_data,
            normalize_target_data=normalize_target_data,
        )

        self.logger.info("Fitting RBF model to the data")
        # RBF fitting for all variables
        rbf_coeffs, opt_sigmas = {}, {}

        # Instantiate the RBFInterpolator
        self._rbf = RBFInterpolator(
            y=subset_data.values,
            d=target_data.values,
            neighbors=self.neighbors,
            smoothing=self.smoothing,
            kernel=self.kernel,
            epsilon=self.epsilon,
            degree=self.degree,
        )

        # Store the RBF coefficients and optimal sigmas
        self._rbf_coeffs = pd.DataFrame(rbf_coeffs)
        self._opt_sigmas = opt_sigmas

        # Set the is_fitted attribute to True
        self.is_fitted = True

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the data for the provided dataset.

        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset to predict (must have same variables than subset).

        Returns
        -------
        pd.DataFrame
            The interpolated dataset.

        Raises
        ------
        ValueError
            If the model is not fitted.

        Notes
        -----
        - This function predicts the data by:
            1. Reconstructing the data using the fitted coefficients.
            2. Denormalizing the target data if normalize_target_data is True.
            3. Calculating the degrees for the target directional variables.
        """

        if self.is_fitted is False:
            raise RBFError("RBF model must be fitted before predicting.")
        self.logger.info("Reconstructing data using fitted coefficients.")
        normalized_dataset = self._preprocess_subset_data(
            subset_data=dataset, is_fit=False
        )
        interpolated_target_array = self.rbf(normalized_dataset.values)
        interpolated_target = pd.DataFrame(
            data=interpolated_target_array, columns=self.target_processed_variables
        )
        if self.is_target_normalized:
            self.logger.info("Denormalizing target data")
            interpolated_target = self.denormalize(
                normalized_data=interpolated_target,
                scale_factor=self.target_scale_factor,
            )
        for directional_variable in self.target_directional_variables:
            self.logger.info(f"Calculating target degrees for {directional_variable}")
            interpolated_target[directional_variable] = self.get_degrees_from_uv(
                xu=interpolated_target[f"{directional_variable}_u"].values,
                xv=interpolated_target[f"{directional_variable}_v"].values,
            )
        return interpolated_target

    def fit_predict(
        self,
        subset_data: pd.DataFrame,
        target_data: pd.DataFrame,
        dataset: pd.DataFrame,
        subset_directional_variables: List[str] = [],
        target_directional_variables: List[str] = [],
        subset_custom_scale_factor: dict = {},
        normalize_target_data: bool = True,
        target_custom_scale_factor: dict = {},
        num_threads: int = None,
    ) -> pd.DataFrame:
        """
        Fits the model to the subset and predicts the interpolated dataset.

        Parameters
        ----------
        subset_data : pd.DataFrame
            The subset data used to fit the model.
        target_data : pd.DataFrame
            The target data used to fit the model.
        dataset : pd.DataFrame
            The dataset to predict (must have same variables than subset).
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
        num_threads : int, optional
            The number of threads to use for the optimization. Default is None.

        Returns
        -------
        pd.DataFrame
            The interpolated dataset.

        Notes
        -----
        - This function fits the model to the subset and predicts the interpolated dataset.
        """

        self.fit(
            subset_data=subset_data,
            target_data=target_data,
            subset_directional_variables=subset_directional_variables,
            target_directional_variables=target_directional_variables,
            subset_custom_scale_factor=subset_custom_scale_factor,
            normalize_target_data=normalize_target_data,
            target_custom_scale_factor=target_custom_scale_factor,
            num_threads=num_threads,
        )

        return self.predict(dataset=dataset)
