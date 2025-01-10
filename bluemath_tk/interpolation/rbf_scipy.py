from typing import List
import copy
import time
import numpy as np
import pandas as pd
from scipy.optimize import fmin, fminbound
from scipy.interpolate import RBFInterpolator
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
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
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html

    Warnings
    --------
    - This class is a Beta, results may not be accurate.

    See Also
    --------
    bluemath_tk.interpolation.RBF :
        The stable version for this model.

    Attributes
    ----------
    sigma_min : float
        The minimum value for the sigma parameter.
        This value might change in the optimization process.
    sigma_max : float
        The maximum value for the sigma parameter.
        This value might change in the optimization process.
    sigma_opt : float
        The optimal value for the sigma parameter.
    kernel : str
        Type of RBF. This should be one of

            - 'linear'               : ``-r``
            - 'thin_plate_spline'    : ``r**2 * log(r)``
            - 'cubic'                : ``r**3``
            - 'quintic'              : ``-r**5``
            - 'multiquadric'         : ``-sqrt(1 + r**2)``
            - 'inverse_multiquadric' : ``1/sqrt(1 + r**2)``
            - 'inverse_quadratic'    : ``1/(1 + r**2)``
            - 'gaussian'             : ``exp(-r**2)``

    smoothing : float or (npoints, ) array_like
        Smoothing parameter. The interpolant perfectly fits the data when this
        is set to 0. For large values, the interpolant approaches a least
        squares fit of a polynomial with the specified degree.
    degree : int
        Degree of the added polynomial. For some RBFs the interpolant may not
        be well-posed if the polynomial degree is too small. Those RBFs and
        their corresponding minimum degrees are

            - 'multiquadric'      : 0
            - 'linear'            : 0
            - 'thin_plate_spline' : 1
            - 'cubic'             : 1
            - 'quintic'           : 2

        The default value is the minimum degree for `kernel` or 0 if there is
        no minimum degree. Set this to -1 for no added polynomial.
    neighbors : int
        If specified, the value of the interpolant at each evaluation point
        will be computed using only this many nearest data points. All the data
        points are used by default.
    rbfs : dict
        Dict with RBFInterpolator instances.
    subset_data : pd.DataFrame
        The subset data used to fit the model.
    normalized_subset_data : pd.DataFrame
        The normalized subset data used to fit the model.
    target_data : pd.DataFrame
        The target data used to fit the model.
    normalized_target_data : pd.DataFrame
        The normalized target data used to fit the model.
        This attribute is only set if normalize_target_data is True in the fit method.
    subset_directional_variables : List[str]
        The subset directional variables.
    target_directional_variables : List[str]
        The target directional variables.
    subset_processed_variables : List[str]
        The subset processed variables.
    target_processed_variables : List[str]
        The target processed variables.
    subset_custom_scale_factor : dict
        The custom scale factor for the subset data.
    target_custom_scale_factor : dict
        The custom scale factor for the target data.
    subset_scale_factor : dict
        The scale factor for the subset data.
    target_scale_factor : dict
        The scale factor for the target data.
    rbf_coeffs : pd.DataFrame
        The RBF coefficients for the target variables.
    opt_sigmas : dict
        The optimal sigmas for the target variables.

    Methods
    -------
    fit(...) :
        Fits the model to the data.
    predict(...) :
        Predicts the data for the provided dataset.
    fit_predict(...) :
        Fits the model to the subset and predicts the interpolated dataset.

    References
    ----------
    .. [1] Fasshauer, G., 2007. Meshfree Approximation Methods with Matlab.
        World Scientific Publishing Co.

    .. [2] http://amadeus.math.iit.edu/~fass/603_ch3.pdf

    .. [3] Wahba, G., 1990. Spline Models for Observational Data. SIAM.

    .. [4] http://pages.stat.wisc.edu/~wahba/stat860public/lect/lect8/lect8.pdf

    Notes
    -----
    .. versionadded:: 1.0.3
    TODO: For the moment, this class only supports optimization for one parameter kernels.
          For this reason, we only have sigma as the parameter to optimize.
          This sigma refers to the sigma parameter in the Gaussian kernel (but is used for all kernels).
    """

    def __init__(
        self,
        sigma_min: float = 0.001,
        sigma_max: float = 1.0,
        sigma_opt: float = None,
        kernel: str = "thin_plate_spline",
        smoothing: float = 0.0,
        degree: int = None,
        neighbors: int = None,
    ):
        """
        Initializes the RBF model.

        Parameters
        ----------
        sigma_min : float, optional
            The minimum value for the sigma parameter. Default is 0.001.
        sigma_max : float, optional
            The maximum value for the sigma parameter. Default is 1.0.
        sigma_opt : float, optional
            The optimal value for the sigma parameter. Default is None.
        kernel : str, optional
            Type of RBF. Default is 'thin_plate_spline'.
        smoothing : float, optional
            Smoothing parameter. Default is 0.0.
        degree : int, optional
            Degree of the added polynomial. Default is None.
        neighbors : int, optional
            If specified, the value of the interpolant at each evaluation point will be
            computed using only this many nearest data points. Default is None.
        """

        super().__init__()
        self.set_logger_name(name=self.__class__.__name__)
        if not isinstance(sigma_min, float) or sigma_min < 0:
            raise ValueError("sigma_min must be a positive float.")
        self._sigma_min = sigma_min
        if not isinstance(sigma_max, float) or sigma_max < sigma_min:
            raise ValueError(
                "sigma_max must be a positive float greater than sigma_min."
            )
        self._sigma_max = sigma_max
        if sigma_opt is not None:
            if not isinstance(sigma_opt, float) or sigma_opt < 0:
                raise ValueError("sigma_opt must be a positive float.")
        self._sigma_opt = sigma_opt
        if not isinstance(kernel, str):
            raise ValueError("kernel must be a string.")
        self._kernel = kernel
        if not isinstance(smoothing, float):
            raise ValueError("smoothing must be a float.")
        self._smoothing = smoothing
        if not isinstance(degree, int) and degree is not None:
            raise ValueError("degree must be an integer.")
        self._degree = degree
        if not isinstance(neighbors, int) and neighbors is not None:
            raise ValueError("neighbors must be an integer.")
        self._neighbors = neighbors
        self._rbfs: dict = {}  # Dict with RBFInterpolator instances
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
    def sigma_min(self) -> float:
        return self._sigma_min

    @property
    def sigma_max(self) -> float:
        return self._sigma_max

    @property
    def sigma_opt(self) -> float:
        return self._sigma_opt

    @property
    def kernel(self) -> str:
        return self._kernel

    @property
    def smoothing(self) -> float:
        return self._smoothing

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def neighbors(self) -> int:
        return self._neighbors

    @property
    def rbfs(self) -> dict:
        return self._rbfs

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
        if not self._opt_sigmas:
            raise ValueError("Specified kernel does not require optimization.")
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
            Whether the data is to fit or not. Default is True.

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

    def _cost_sigma(
        self, sigma: float, x: np.ndarray, y: np.ndarray, k: int = 5
    ) -> float:
        """
        Calculate the cost for a given sigma using K-Fold cross-validation.

        Parameters
        ----------
        sigma : float
            The sigma parameter for the kernel.
        x : np.ndarray
            The input data.
        y : np.ndarray
            The target data.
        k : int, optional
            The number of folds for cross-validation. Default is 5.

        Returns
        -------
        float
            The total cost for the RBF interpolation.
        """

        kf = KFold(n_splits=k)
        total_cost = 0.0

        for train_index, val_index in kf.split(x):
            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Instantiate the RBFInterpolator
            rbf = RBFInterpolator(
                y=x_train,
                d=y_train,
                neighbors=self.neighbors,
                smoothing=self.smoothing,
                kernel=self.kernel,
                epsilon=sigma,
                degree=self.degree,
            )

            # Predict on the validation set
            predicted_y = rbf(x_val)

            # Calculate the cost (mean squared error)
            cost = mean_squared_error(y_val, predicted_y)
            total_cost += cost

        return total_cost / k

    def _calc_opt_sigma(
        self,
        target_variable: np.ndarray,
        subset_variables: np.ndarray,
        iteratively_update_sigma: bool = False,
    ) -> RBFInterpolator:
        """
        This function calculates the optimal sigma for the given target variable.

        Parameters
        ----------
        target_variable : np.ndarray
            The target variable to interpolate.
        subset_variables : np.ndarray
            The subset variables used to interpolate.
        iteratively_update_sigma : bool, optional
            Whether to iteratively update the sigma parameter. Default is False.

        Returns
        -------
        float
            The optimal sigma.
        """

        t0 = time.time()

        # Optimize sigma using fminbound or fmin
        if self.sigma_opt is not None:
            opt_sigma = fmin(
                func=self._cost_sigma,
                x0=self.sigma_opt,
                args=(subset_variables, target_variable),
                disp=0,
            )[-1]
            if iteratively_update_sigma:
                self._sigma_opt = opt_sigma
        else:
            opt_sigma = fminbound(
                func=self._cost_sigma,
                x1=self.sigma_min,
                x2=self.sigma_max,
                args=(subset_variables, target_variable),
                disp=0,
            )

        # Save the fitted RBF for the optimal sigma
        rbf = RBFInterpolator(
            y=subset_variables,
            d=target_variable,
            neighbors=self.neighbors,
            smoothing=self.smoothing,
            kernel=self.kernel,
            epsilon=opt_sigma,
            degree=self.degree,
        )

        # Calculate the time taken to optimize sigma
        t1 = time.time()
        self.logger.info(f"Optimal sigma: {opt_sigma} - Time: {t1 - t0:.2f} seconds")

        return rbf, opt_sigma

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
        iteratively_update_sigma: bool = False,
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
        iteratively_update_sigma : bool, optional
            Whether to iteratively update the sigma parameter. Default is False.

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

        # Optimize sigma for each target variable
        for target_var in target_data.columns:
            self.logger.info(f"Fitting RBF for variable {target_var}")
            target_var_values = target_data[target_var].values
            if (
                self.kernel == "linear"
                or self.kernel == "cubic"
                or self.kernel == "quintic"
                or self.kernel == "thin_plate_spline"
            ):
                rbf = RBFInterpolator(
                    y=subset_data.values,
                    d=target_var_values,
                    neighbors=self.neighbors,
                    smoothing=self.smoothing,
                    kernel=self.kernel,
                    degree=self.degree,
                )
                opt_sigma = None
            else:
                rbf, opt_sigma = self._calc_opt_sigma(
                    target_variable=target_var_values,
                    subset_variables=subset_data.values,
                    iteratively_update_sigma=iteratively_update_sigma,
                )
            self.rbfs[target_var] = copy.deepcopy(rbf)
            rbf_coeffs[target_var] = rbf._coeffs.flatten()
            opt_sigmas[target_var] = opt_sigma

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

        # Create an empty array to store the interpolated target data
        interpolated_target_array = np.zeros(
            (normalized_dataset.shape[0], len(self.target_processed_variables))
        )
        for target_var in self.target_processed_variables:
            self.logger.info(f"Predicting target variable {target_var}")
            rbf = self.rbfs[target_var]
            interpolated_target_array[
                :, self.target_processed_variables.index(target_var)
            ] = rbf(normalized_dataset.values)
        interpolated_target = pd.DataFrame(
            data=interpolated_target_array, columns=self.target_processed_variables
        )

        # Denormalize the target data if normalize_target_data is True
        if self.is_target_normalized:
            self.logger.info("Denormalizing target data")
            interpolated_target = self.denormalize(
                normalized_data=interpolated_target,
                scale_factor=self.target_scale_factor,
            )

        # Calculate the degrees for the target directional variables
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
        iteratively_update_sigma: bool = False,
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
        iteratively_update_sigma : bool, optional
            Whether to iteratively update the sigma parameter. Default is False.

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
            iteratively_update_sigma=iteratively_update_sigma,
        )

        return self.predict(dataset=dataset)
