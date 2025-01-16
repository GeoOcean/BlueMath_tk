import time
from typing import List, Tuple, Callable
import numpy as np
import pandas as pd
from scipy.optimize import fminbound, fmin
from ._base_interpolation import BaseInterpolation
from ..core.decorators import validate_data_rbf


def gaussian_kernel(r: float, const: float) -> float:
    """
    This function calculates the Gaussian kernel for the given distance and constant.

    Parameters
    ----------
    r : float
        The distance.
    const : float
        The constant (default name is usually sigma for gaussian kernel).

    Returns
    -------
    float
        The Gaussian kernel value.

    Notes
    -----
    - The Gaussian kernel is defined as:
      K(r) = exp(r^2 / 2 * const^2) (https://en.wikipedia.org/wiki/Gaussian_function)
    - Here, we are assuming the mean is 0.
    """

    return np.exp(-0.5 * r * r / (const * const))


def multiquadratic_kernel(r, const):
    return np.sqrt(1 + (r / const) ** 2)


def inverse_kernel(r, const):
    return 1 / np.sqrt(1 + (r / const) ** 2)


def cubic_kernel(r, const):
    return r**3


def thin_plate_kernel(r, const):
    return r**2 * np.log(r / const)


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

    Attributes
    ----------
    sigma_min : float
        The minimum value for the sigma parameter.
        This value might change in the optimization process.
    sigma_max : float
        The maximum value for the sigma parameter.
        This value might change in the optimization process.
    sigma_diff : float
        The difference between the optimal bounded sigma and the minimum and maximum sigma values.
        If the difference is less than this value, the optimization process continues.
    kernel : str
        The kernel to use for the RBF model.
        The available kernels are:

            - gaussian              : ``exp(-1/2 * (r / const)**2)``
            - multiquadratic        : ``sqrt(1 + (r / const)**2)``
            - inverse               : ``1 / sqrt(1 + (r / const)**2)``
            - cubic                 : ``r**3``
            - thin_plate            : ``r**2 * log(r / const)``

    kernel_func : function
        The kernel function to use for the RBF model.
    smooth : float
        The smoothness parameter.
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
    fit :
        Fits the model to the data.
    predict :
        Predicts the data for the provided dataset.
    fit_predict :
        Fits the model to the subset and predicts the interpolated dataset.

    Notes
    -----
    TODO: For the moment, this class only supports optimization for one parameter kernels.
          For this reason, we only have sigma as the parameter to optimize.
          This sigma refers to the sigma parameter in the Gaussian kernel (but is used for all kernels).

    Main reference for sigma optimization: https://link.springer.com/article/10.1023/A:1018975909870

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from bluemath_tk.interpolation.rbf import RBF
    >>> dataset = pd.DataFrame(
    ...     {
    ...         "Hs": np.random.rand(1000) * 7,
    ...         "Tp": np.random.rand(1000) * 20,
    ...         "Dir": np.random.rand(1000) * 360,
    ...     }
    ... )
    >>> subset = dataset.sample(frac=0.25)
    >>> target = pd.DataFrame(
    ...     {
    ...         "HsPred": self.subset["Hs"] * 2 + self.subset["Tp"] * 3,
    ...         "DirPred": - self.subset["Dir"],
    ...     }
    ... )
    >>> rbf = RBF()
    >>> predictions = rbf.fit_predict(
    ...     subset_data=subset,
    ...     subset_directional_variables=["Dir"],
    ...     target_data=target,
    ...     target_directional_variables=["DirPred"],
    ...     normalize_target_data=True,
    ...     dataset=dataset,
    ...     num_threads=4,
    ...     iteratively_update_sigma=True,
    ... )
    """

    rbf_kernels = {
        "gaussian": gaussian_kernel,
        "multiquadratic": multiquadratic_kernel,
        "inverse": inverse_kernel,
        "cubic": cubic_kernel,
        "thin_plate": thin_plate_kernel,
    }

    def __init__(
        self,
        sigma_min: float = 0.001,
        sigma_max: float = 0.1,
        sigma_diff: float = 0.0001,
        sigma_opt: float = None,
        kernel: str = "gaussian",
        smooth: float = 1e-5,
    ):
        """
        Raises
        ------
        ValueError
            If the sigma_min is not a positive float.
            If the sigma_max is not a positive float greater than sigma_min.
            If the sigma_diff is not a positive float.
            If the kernel is not a string and one of the available kernels.
            If the smooth is not a positive float.
        """

        initial_msg = f"""
        ---------------------------------------------------------------------------------
        | Initializing RBF interpolation model with the following parameters:
        |    - sigma_min: {sigma_min}
        |    - sigma_max: {sigma_max}
        |    - sigma_diff: {sigma_diff}
        |    - sigma_opt: {sigma_opt}
        |    - kernel: {kernel}
        |    - smooth: {smooth}
        | For more information, please refer to the documentation.
        | Recommended lecture: https://link.springer.com/article/10.1023/A:1018975909870
        ---------------------------------------------------------------------------------
        """
        print(initial_msg)

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
        if not isinstance(sigma_diff, float) or sigma_diff < 0:
            raise ValueError("sigma_diff must be a positive float.")
        self._sigma_diff = sigma_diff
        if not isinstance(kernel, str) or kernel not in self.rbf_kernels.keys():
            raise ValueError(
                f"kernel must be a string and one of {list(self.rbf_kernels.keys())}."
            )
        if sigma_opt is not None:
            if not isinstance(sigma_opt, float) or sigma_opt < 0:
                raise ValueError("sigma_opt must be a positive float.")
        self._sigma_opt = sigma_opt
        if not isinstance(kernel, str) or kernel not in self.rbf_kernels.keys():
            raise ValueError(
                f"kernel must be a string and one of {list(self.rbf_kernels.keys())}."
            )
        self._kernel = kernel
        self._kernel_func = self.rbf_kernels[self.kernel]
        if not isinstance(smooth, float) or smooth < 0:
            raise ValueError("smooth must be a positive float.")
        self._smooth = smooth
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
    def sigma_diff(self) -> float:
        return self._sigma_diff

    @property
    def sigma_opt(self) -> float:
        return self._sigma_opt

    @property
    def kernel(self) -> str:
        return self._kernel

    @property
    def kernel_func(self) -> Callable:
        return self._kernel_func

    @property
    def smooth(self) -> float:
        return self._smooth

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

    def _rbf_assemble(self, x, sigma):
        """
        Assemble the RBF matrix.

        Parameters
        ----------
        x : np.ndarray
            The data.
        sigma : float
            The sigma parameter for the kernel.

        Returns
        -------
        np.ndarray
            The data with all the calculated kernel values.
        """

        # Get the number of rows and columns in x
        dim, n = x.shape

        # Compute the pairwise distances
        dists = np.linalg.norm(x[:, :, np.newaxis] - x[:, np.newaxis, :], axis=0)

        # Apply the kernel function to the distances
        A = self.kernel_func(dists, sigma)

        # Subtract the smoothing parameter from the diagonal elements
        np.fill_diagonal(A, A.diagonal() - self.smooth)

        # Add the identity matrix to the matrix (polynomial term)
        P = np.hstack((np.ones((n, 1)), x.T))
        A = np.vstack(
            (np.hstack((A, P)), np.hstack((P.T, np.zeros((dim + 1, dim + 1)))))
        )

        return A

    def _calc_rbf_coeff(
        self, sigma: float, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function calculates the RBF coefficients for the given data.

        Parameters
        ----------
        sigma : float
            The sigma parameter for the kernel.
        x : np.ndarray
            The subset data used to interpolate.
        y : np.ndarray
            The target data to interpolate.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The RBF coefficients and the A matrix.
        """

        # Get the number of rows and columns in x
        m, n = x.shape

        # Assemble the A matrix
        A = self._rbf_assemble(x=x, sigma=sigma)

        # Concatenate y with zeros and reshape
        b = np.concatenate((y, np.zeros((m + 1,)))).reshape(-1, 1)

        # Calculate the RBF coefficients
        rbfcoeff, _, _, _ = np.linalg.lstsq(A, b, rcond=None)  # inverse

        return rbfcoeff, A

    def _cost_sigma(self, sigma: float, x: np.ndarray, y: np.ndarray) -> float:
        """
        This function is called by fminbound to minimize the cost function.

        Parameters
        ----------
        sigma : float
            The sigma parameter for the kernel.
        x : np.ndarray
            The subset data used to interpolate.
        y : np.ndarray
            The target data to interpolate.

        Returns
        -------
        float
            The cost value.
        """

        # Calculate RBF coefficients and A matrix
        rbf_coeff, A = self._calc_rbf_coeff(sigma=sigma, x=x, y=y)

        # Extract the top-left n x n submatrix from A
        m, n = x.shape
        A = A[:n, :n]

        # Compute the pseudo-inverse of the submatrix A
        invA = np.linalg.pinv(A)

        # Initialize residuals by subtracting the last m elements of rbf_coeff from y
        m1, n1 = rbf_coeff.shape
        kk = y - rbf_coeff[m1 - m - 1]

        # Adjust residuals by subtracting the product of rbf_coeff and x
        # DEPRECATED: The loop is replaced by the vectorized operation below
        # for i in range(m):
        #     kk = kk - rbf_coeff[m1 - m + i] * x[i, :]
        kk -= np.dot(rbf_coeff[m1 - m :].T, x).reshape(-1)

        # Calculate the cost by multiplying invA with kk and normalizing by the diagonal elements of invA
        ceps = np.dot(invA, kk) / np.diagonal(invA)

        # Return the norm of ceps, representing the cost
        yy = np.linalg.norm(ceps)

        return yy

    def _calc_opt_sigma(
        self,
        target_variable: np.ndarray,
        subset_variables: np.ndarray,
        iteratively_update_sigma: bool = False,
    ) -> float:
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
        # Initialize sigma_min, sigma_max, and d_sigma
        sigma_min, sigma_max, d_sigma = self.sigma_min, self.sigma_max, 0

        if self.sigma_opt is not None:
            # Optimize sigma using the specified sigma_opt
            opt_sigma = fmin(
                func=self._cost_sigma,
                x0=self.sigma_opt,
                args=(subset_variables, target_variable),
                disp=0,
            )
            if iteratively_update_sigma:
                self._sigma_opt = opt_sigma
        else:
            # Loop until sigma_diff is less than the specified sigma_diff
            while d_sigma < self.sigma_diff:
                opt_sigma = fminbound(
                    func=self._cost_sigma,
                    x1=sigma_min,
                    x2=sigma_max,
                    args=(subset_variables, target_variable),
                    disp=0,
                )
                lm_min = np.abs(opt_sigma - sigma_min)
                lm_max = np.abs(opt_sigma - sigma_max)
                if lm_min < self.sigma_diff:
                    sigma_min = sigma_min - sigma_min / 2
                elif lm_max < self.sigma_min:
                    sigma_max = sigma_max + sigma_max / 2
                d_sigma = np.nanmin([lm_min, lm_max])
            if iteratively_update_sigma:
                self._sigma_opt = opt_sigma

        # Calculate the time taken to optimize sigma
        t1 = time.time()
        self.logger.info(f"Optimal sigma: {opt_sigma} - Time: {t1 - t0:.2f} seconds")

        # Calculate the RBF coefficients for the optimal sigma
        rbf_coeff, _ = self._calc_rbf_coeff(
            sigma=opt_sigma, x=subset_variables, y=target_variable
        )

        return rbf_coeff, opt_sigma

    def _rbf_interpolate(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        This function interpolates the dataset.

        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset to interpolate (must have same variables as subset).

        Returns
        -------
        pd.DataFrame
            The interpolated dataset (with all target variables).
        """

        normalized_dataset = self._preprocess_subset_data(
            subset_data=dataset, is_fit=False
        )

        # Get the number of rows and columns in subset and dataset
        num_vars_subset, num_points_subset = self.normalized_subset_data.T.shape
        _, num_points_dataset = normalized_dataset.T.shape

        # Initialize the interpolated dataset
        interpolated_array = np.zeros(
            (num_points_dataset, len(self.target_processed_variables))
        )

        # Loop through the target variables
        for i_var, target_var in enumerate(self.target_processed_variables):
            self.logger.info(f"Interpolating target variable {target_var}")
            rbf_coeff = self._rbf_coeffs[target_var].values
            opt_sigma = self._opt_sigmas[target_var]
            r = np.linalg.norm(
                normalized_dataset.values[:, np.newaxis, :]
                - self.normalized_subset_data.values[np.newaxis, :, :],
                axis=2,
            )
            kernel_values = self.kernel_func(r, opt_sigma)
            linear_part = np.dot(
                normalized_dataset.values,
                rbf_coeff[
                    num_points_subset + 1 : num_points_subset + 1 + num_vars_subset
                ].T,
            )
            s = (
                rbf_coeff[num_points_subset]
                + np.dot(kernel_values, rbf_coeff[:num_points_subset])
                + linear_part
            )
            interpolated_array[:, i_var] = s

        return pd.DataFrame(interpolated_array, columns=self.target_processed_variables)

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

        for target_var in target_data.columns:
            self.logger.info(f"Fitting RBF for variable {target_var}")
            target_var_values = target_data[target_var].values
            rbf_coeff, opt_sigma = self._calc_opt_sigma(
                target_variable=target_var_values,
                subset_variables=subset_data.values.T,
                iteratively_update_sigma=iteratively_update_sigma,
            )
            rbf_coeffs[target_var] = rbf_coeff.flatten()
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
        interpolated_target = self._rbf_interpolate(dataset=dataset)
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
