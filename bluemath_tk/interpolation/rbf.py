"""
Package: BlueMath_tk
Module: interpolation
File: rbf.py
Author: GeoOcean Research Group, Universidad de Cantabria
Repository: https://github.com/GeoOcean/BlueMath_tk.git
Status: Under development (Working)
"""

import time
from collections.abc import Callable

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fmin, fminbound
from sklearn.model_selection import KFold

from ..core.decorators import validate_data_rbf
from ._base_interpolation import BaseInterpolation


def linear_kernel(r: float, const: float):
    """
    Calculate the linear kernel value.

    Parameters
    ----------
    r : float
        The distance between the data points.
    const : float
        The constant parameter (not used in linear kernel).

    Returns
    -------
    float
        The value of the linear kernel.
    """

    return -r


def cubic_kernel(r: float, const: float):
    """
    Calculate the cubic kernel value.

    Parameters
    ----------
    r : float
        The distance between the data points.
    const : float
        The constant parameter (not used in cubic kernel).

    Returns
    -------
    float
        The value of the cubic kernel.
    """

    return r**3


def quintic_kernel(r: float, const: float):
    """
    Calculate the quintic kernel value.

    Parameters
    ----------
    r : float
        The distance between the data points.
    const : float
        The constant parameter (not used in quintic kernel).

    Returns
    -------
    float
        The value of the quintic kernel.
    """

    return -(r**5)


def thin_plate_kernel(r: float, const: float):
    """
    Calculate the thin plate spline kernel value.

    Parameters
    ----------
    r : float
        The distance between the data points.
    const : float
        The constant parameter.

    Returns
    -------
    float
        The value of the thin plate spline kernel.
    """

    return r**2 * np.log(r / const)


def inverse_kernel(r: float, const: float):
    """
    Calculate the inverse multiquadratic kernel value.

    Parameters
    ----------
    r : float
        The distance between the data points.
    const : float
        The constant parameter.

    Returns
    -------
    float
        The value of the inverse multiquadratic kernel.
    """

    return 1 / np.sqrt(1 + (r / const) ** 2)


def inverse_quadratic_kernel(r: float, const: float):
    """
    Calculate the inverse quadratic kernel value.

    Parameters
    ----------
    r : float
        The distance between the data points.
    const : float
        The constant parameter.

    Returns
    -------
    float
        The value of the inverse quadratic kernel.
    """

    return 1 / (1 + (r / const) ** 2)


def multiquadratic_kernel(r: float, const: float):
    """
    Calculate the multiquadratic kernel value.

    Parameters
    ----------
    r : float
        The distance between the data points.
    const : float
        The constant parameter.

    Returns
    -------
    float
        The value of the multiquadratic kernel.
    """

    return np.sqrt(1 + (r / const) ** 2)


def gaussian_kernel(r: float, const: float) -> float:
    """
    Calculate the Gaussian kernel value for the given distance and constant.

    Parameters
    ----------
    r : float
        The distance between the data points.
    const : float
        The constant (usually called sigma for the Gaussian kernel).

    Returns
    -------
    float
        The value of the Gaussian kernel.

    Notes
    -----
    - The Gaussian kernel is defined as:
      K(r) = exp(-0.5 * (r / const)**2) (https://en.wikipedia.org/wiki/Gaussian_function)
    - Here, we are assuming the mean is 0.
    """

    return np.exp(-0.5 * r * r / (const * const))


class RBFError(Exception):
    """
    Custom exception for RBF interpolation model.
    """

    def __init__(self, message: str = "RBF error occurred."):
        self.message = message
        super().__init__(self.message)


class RBF(BaseInterpolation):
    """
    Radial Basis Function (RBF) interpolation model.

    Notes
    -----
    TODO: For the moment, this class only supports optimization for one
          parameter kernels. For this reason, we only have sigma as the
          parameter to optimize. This sigma refers to the sigma parameter
          in the Gaussian kernel (but is used for all kernels).

    Examples
    --------
    .. jupyter-execute::

        import numpy as np
        import pandas as pd
        from bluemath_tk.interpolation.rbf import RBF

        dataset = pd.DataFrame(
            {
                "Hs": np.random.rand(1000) * 7,
                "Tp": np.random.rand(1000) * 20,
                "Dir": np.random.rand(1000) * 360,
            }
        )
        subset = dataset.sample(frac=0.25)
        target = pd.DataFrame(
            {
                "HsPred": subset["Hs"] * 2 + subset["Tp"] * 3,
                "DirPred": - subset["Dir"],
            }
        )

        rbf = RBF()
        predictions = rbf.fit_predict(
            subset_data=subset,
            subset_directional_variables=["Dir"],
            target_data=target,
            target_directional_variables=["DirPred"],
            normalize_target_data=True,
            dataset=dataset,
            num_workers=4,
            iteratively_update_sigma=True,
        )
        print(predictions.head())
        rbf.explain(dataset=dataset, target_variable="HsPred")

    References
    ----------
    [1] https://link.springer.com/article/10.1023/A:1018975909870
    [2] https://en.wikipedia.org/wiki/Radial_basis_function
    [3] https://en.wikipedia.org/wiki/Gaussian_function
    """

    rbf_kernels = {
        "linear": linear_kernel,
        "cubic": cubic_kernel,
        "quintic": quintic_kernel,
        "thin_plate": thin_plate_kernel,
        "inverse": inverse_kernel,
        "inverse_quadratic": inverse_quadratic_kernel,
        "multiquadratic": multiquadratic_kernel,
        "gaussian": gaussian_kernel,
    }

    # Kernels that don't require sigma optimization
    _kernels_no_sigma_opt = {"linear", "cubic", "quintic", "thin_plate"}

    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 10.0,
        sigma_diff: float = 0.01,
        sigma_opt: float = None,
        kernel: str = "gaussian",
        smooth: float = 1e-5,
    ):
        """
        Initialize RBF interpolation model.

        Parameters
        ----------
        sigma_min : float, optional
            The minimum value for the sigma parameter. Default is 0.01.
        sigma_max : float, optional
            The maximum value for the sigma parameter. Default is 10.0.
        sigma_diff : float, optional
            The difference between the sigma parameters. Default is 0.01.
        sigma_opt : float, optional
            The optimal value for the sigma parameter. Default is None.
        kernel : str, optional
            The kernel to use for the interpolation. Default is "gaussian".
        smooth : float, optional
            The smoothness parameter. Default is 1e-5.
        """

        super().__init__()
        self.set_logger_name(name=self.__class__.__name__)

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
        self.logger.info(initial_msg)

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
        self._rbf_coeffs: pd.DataFrame = pd.DataFrame()
        self._opt_sigmas: dict = {}

        # Exclude attributes to .save_model() method
        self._exclude_attributes = []

        # Row chunks for parallel computation
        self.row_chunks: int = None

    @property
    def sigma_min(self) -> float:
        """Return the minimum sigma value."""
        return self._sigma_min

    @property
    def sigma_max(self) -> float:
        """Return the maximum sigma value."""
        return self._sigma_max

    @property
    def sigma_diff(self) -> float:
        """Return the sigma difference threshold."""
        return self._sigma_diff

    @property
    def sigma_opt(self) -> float:
        """Return the optimal sigma value."""
        return self._sigma_opt

    @property
    def kernel(self) -> str:
        """Return the kernel name."""
        return self._kernel

    @property
    def kernel_func(self) -> Callable:
        """Return the kernel function."""
        return self._kernel_func

    @property
    def smooth(self) -> float:
        """Return the smoothness parameter."""
        return self._smooth

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
    def subset_custom_scale_factor(self) -> dict:
        """Return the subset custom scale factor."""
        return self._subset_custom_scale_factor

    @property
    def target_custom_scale_factor(self) -> dict:
        """Return the target custom scale factor."""
        return self._target_custom_scale_factor

    @property
    def subset_scale_factor(self) -> dict:
        """Return the subset scale factor."""
        return self._subset_scale_factor

    @property
    def target_scale_factor(self) -> dict:
        """Return the target scale factor."""
        return self._target_scale_factor

    @property
    def rbf_coeffs(self) -> pd.DataFrame:
        """Return the RBF coefficients."""
        return self._rbf_coeffs

    @property
    def opt_sigmas(self) -> dict:
        """
        Return the optimal sigmas.

        Returns
        -------
        dict
            Dictionary mapping target variable names to their optimal sigma values.
            Values may be None for kernels that don't require sigma optimization
            (e.g., linear, cubic, quintic, thin_plate).
        """
        return self._opt_sigmas

    def check_fit_quality(self, verbose: bool = True) -> dict:
        """
        Check the quality of the RBF fit and return diagnostic information.

        Parameters
        ----------
        verbose : bool, optional
            If True, print a summary of the fit quality. Default is True.

        Returns
        -------
        dict
            Dictionary containing diagnostic information:
            - 'sigmas': dict of sigma values for each target variable
            - 'sigma_warnings': list of warnings about sigma values
            - 'matrix_condition': dict of condition numbers for each variable
            - 'matrix_rank': dict of matrix ranks for each variable
            - 'fit_status': overall fit status ('good', 'warning', 'poor')

        Raises
        ------
        RBFError
            If the model is not fitted.
        """
        if not self.is_fitted:
            raise RBFError("RBF model must be fitted before checking fit quality.")

        diagnostics = {
            "sigmas": {},
            "sigma_warnings": [],
            "matrix_condition": {},
            "matrix_rank": {},
            "fit_status": "good",
        }

        # Check each target variable
        for target_var in self.target_processed_variables:
            opt_sigma = self._opt_sigmas.get(target_var)

            if opt_sigma is not None:
                diagnostics["sigmas"][target_var] = opt_sigma

                # Check if sigma is reasonable
                if opt_sigma > self.sigma_max * 0.9:
                    diagnostics["sigma_warnings"].append(
                        f"{target_var}: sigma ({opt_sigma:.6f}) is near upper "
                        f"bound ({self.sigma_max})"
                    )
                    diagnostics["fit_status"] = "warning"
                elif opt_sigma < self.sigma_min * 1.1:
                    diagnostics["sigma_warnings"].append(
                        f"{target_var}: sigma ({opt_sigma:.6f}) is near lower "
                        f"bound ({self.sigma_min})"
                    )
                    diagnostics["fit_status"] = "warning"

                # Reconstruct matrix to check condition
                x = self.normalized_subset_data.values.T
                A = self._rbf_assemble(x=x, sigma=opt_sigma)

                cond = np.linalg.cond(A)
                rank = np.linalg.matrix_rank(A)
                expected_rank = A.shape[0]

                diagnostics["matrix_condition"][target_var] = cond
                diagnostics["matrix_rank"][target_var] = {
                    "actual": rank,
                    "expected": expected_rank,
                    "deficiency": expected_rank - rank,
                }

                if cond > 1e12:
                    diagnostics["fit_status"] = "poor"
                elif cond > 1e8:
                    if diagnostics["fit_status"] == "good":
                        diagnostics["fit_status"] = "warning"

                if rank < expected_rank:
                    if diagnostics["fit_status"] == "good":
                        diagnostics["fit_status"] = "warning"

        if verbose:
            self._print_fit_quality_summary(diagnostics)

        return diagnostics

    def _print_fit_quality_summary(self, diagnostics: dict) -> None:
        """Print a summary of fit quality diagnostics."""
        print("\n" + "=" * 60)
        print("RBF Fit Quality Summary")
        print("=" * 60)

        print(f"\nOverall Status: {diagnostics['fit_status'].upper()}")

        if diagnostics["sigmas"]:
            print("\nOptimal Sigma Values:")
            for var, sigma in diagnostics["sigmas"].items():
                print(f"  {var}: {sigma:.6f}")

        if diagnostics["sigma_warnings"]:
            print("\n⚠️  Sigma Warnings:")
            for warning in diagnostics["sigma_warnings"]:
                print(f"  - {warning}")

        if diagnostics["matrix_condition"]:
            print("\nMatrix Condition Numbers:")
            for var, cond in diagnostics["matrix_condition"].items():
                status = "⚠️" if cond > 1e8 else "✓"
                print(f"  {status} {var}: {cond:.2e}")

        if diagnostics["matrix_rank"]:
            print("\nMatrix Rank:")
            for var, rank_info in diagnostics["matrix_rank"].items():
                if rank_info["deficiency"] > 0:
                    print(
                        f"  ⚠️  {var}: {rank_info['actual']}/{rank_info['expected']} "
                        f"(deficiency: {rank_info['deficiency']})"
                    )
                else:
                    print(
                        f"  ✓ {var}: {rank_info['actual']}/{rank_info['expected']}"
                    )

        print("\n" + "=" * 60)

    def _preprocess_subset_data(
        self, subset_data: pd.DataFrame, is_fit: bool = True
    ) -> pd.DataFrame:
        """
        Preprocess the subset data.

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
        - Preprocesses the subset data by:
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
            self._subset_processed_variables = list(subset_data.columns)
            self._normalized_subset_data = normalized_subset_data
            self._subset_scale_factor = subset_scale_factor
        else:
            normalized_subset_data = normalized_subset_data[
                self.subset_processed_variables
            ]

        return normalized_subset_data.copy()

    def _preprocess_target_data(
        self,
        target_data: pd.DataFrame,
        normalize_target_data: bool = True,
    ) -> pd.DataFrame:
        """
        Preprocess the target data.

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
        - Preprocesses the target data by:
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
        # For exact interpolation (smooth=0), use machine epsilon for numerical
        # stability to handle near-singular matrices while maintaining exactness
        if self.smooth == 0.0:
            # Use machine epsilon for minimal numerical stability
            # This is small enough to maintain essentially exact interpolation
            numerical_stability = np.finfo(A.dtype).eps
        else:
            numerical_stability = self.smooth
        np.fill_diagonal(A, A.diagonal() - numerical_stability)

        # Add the identity matrix to the matrix (polynomial term)
        P = np.hstack((np.ones((n, 1)), x.T))
        A = np.vstack(
            (np.hstack((A, P)), np.hstack((P.T, np.zeros((dim + 1, dim + 1)))))
        )

        return A

    def _calc_rbf_coeff(
        self, sigma: float, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the RBF coefficients for the given data.

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
        tuple[np.ndarray, np.ndarray]
            The RBF coefficients and the A matrix.
        """

        # Get the number of rows and columns in x
        m, n = x.shape

        # Assemble the A matrix
        A = self._rbf_assemble(x=x, sigma=sigma)

        # Concatenate y with zeros and reshape
        b = np.concatenate((y, np.zeros((m + 1,)))).reshape(-1, 1)

        # Calculate the RBF coefficients
        rbfcoeff, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        return rbfcoeff, A

    def _cost_sigma(self, sigma: float, x: np.ndarray, y: np.ndarray) -> float:
        """
        Minimize the cost function (called by fminbound).

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

        # Calculate the cost by multiplying invA with kk and normalizing
        # by the diagonal elements of invA
        ceps = np.dot(invA, kk) / np.diagonal(invA)

        # Return the norm of ceps, representing the cost
        yy = np.linalg.norm(ceps)

        return yy

    def _needs_sigma_optimization(self) -> bool:
        """
        Check if the current kernel requires sigma optimization.

        Returns
        -------
        bool
            True if the kernel requires sigma optimization, False otherwise.
        """
        return self.kernel not in self._kernels_no_sigma_opt

    def _validate_sigma(
        self,
        opt_sigma: float,
        sigma_min: float,
        sigma_max: float,
        subset_variables: np.ndarray,
    ) -> None:
        """
        Validate that the optimized sigma value is reasonable.

        Parameters
        ----------
        opt_sigma : float
            The optimized sigma value.
        sigma_min : float
            The minimum sigma value used in optimization.
        sigma_max : float
            The maximum sigma value used in optimization.
        subset_variables : np.ndarray
            The subset variables used for interpolation.

        Notes
        -----
        Logs warnings if:
        - Sigma is at or near the boundaries (suggests optimization failed)
        - Sigma is very large relative to data scale (suggests poor fit)
        """
        # Check if sigma is at boundaries
        tolerance = 0.01  # 1% tolerance
        if opt_sigma <= sigma_min * (1 + tolerance):
            self.logger.warning(
                f"Optimal sigma ({opt_sigma:.6f}) is at or near the lower "
                f"boundary ({sigma_min:.6f}). This may indicate optimization "
                "failed or the data requires a smaller sigma. Consider "
                "decreasing sigma_min."
            )
        elif opt_sigma >= sigma_max * (1 - tolerance):
            self.logger.warning(
                f"Optimal sigma ({opt_sigma:.6f}) is at or near the upper "
                f"boundary ({sigma_max:.6f}). This may indicate optimization "
                "failed or the data requires a larger sigma. Consider "
                "increasing sigma_max."
            )

        # Check if sigma is very large relative to data scale
        # For normalized data, typical scale is ~1, so sigma > 10 is suspicious
        # For Gaussian kernel, sigma should be on the order of typical distances
        dim, n = subset_variables.shape
        if dim > 0 and n > 1:
            # Calculate typical distance between points
            sample_distances = []
            for i in range(min(10, n)):  # Sample a few points
                for j in range(i + 1, min(i + 5, n)):
                    dist = np.linalg.norm(
                        subset_variables[:, i] - subset_variables[:, j]
                    )
                    sample_distances.append(dist)
            if sample_distances:
                typical_distance = np.median(sample_distances)
                if opt_sigma > 10 * typical_distance:
                    self.logger.warning(
                        f"Optimal sigma ({opt_sigma:.6f}) is very large "
                        f"compared to typical data distances "
                        f"({typical_distance:.6f}). This may indicate a poor "
                        "fit or that the Gaussian kernel is not appropriate "
                        "for this data."
                    )
                elif opt_sigma < 0.01 * typical_distance:
                    self.logger.warning(
                        f"Optimal sigma ({opt_sigma:.6f}) is very small "
                        f"compared to typical data distances "
                        f"({typical_distance:.6f}). This may cause numerical "
                        "instability or overfitting."
                    )

    def _validate_fit_quality(
        self,
        A: np.ndarray,
        rbf_coeff: np.ndarray,
        target_variable: np.ndarray,
    ) -> None:
        """
        Validate the quality of the RBF fit.

        Parameters
        ----------
        A : np.ndarray
            The RBF matrix used for fitting.
        rbf_coeff : np.ndarray
            The RBF coefficients.
        target_variable : np.ndarray
            The target variable values.

        Notes
        -----
        Logs warnings if:
        - Matrix is ill-conditioned (high condition number)
        - Matrix is rank-deficient
        - Coefficients are very large (suggests instability)
        """
        # Check matrix condition number
        cond = np.linalg.cond(A)
        if cond > 1e12:
            self.logger.warning(
                f"RBF matrix is ill-conditioned (condition number: {cond:.2e}). "
                "This may cause numerical instability and poor predictions. "
                "Consider increasing the smooth parameter or checking data "
                "quality."
            )
        elif cond > 1e8:
            self.logger.info(
                f"RBF matrix condition number: {cond:.2e} (moderately high, "
                "but acceptable)"
            )

        # Check matrix rank
        rank = np.linalg.matrix_rank(A)
        expected_rank = A.shape[0]
        if rank < expected_rank:
            rank_deficiency = expected_rank - rank
            self.logger.warning(
                f"RBF matrix is rank-deficient (rank: {rank}/{expected_rank}, "
                f"deficiency: {rank_deficiency}). This may prevent exact "
                "interpolation at training points. Consider checking for "
                "collinear data points or redundant features."
            )

        # Check coefficient magnitudes
        max_coeff = np.max(np.abs(rbf_coeff))
        if max_coeff > 1e10:
            self.logger.warning(
                f"RBF coefficients are very large (max: {max_coeff:.2e}). "
                "This may indicate numerical instability or a poor fit. "
                "Consider adjusting sigma or increasing the smooth parameter."
            )

    def _calc_opt_sigma(
        self,
        target_variable: np.ndarray,
        subset_variables: np.ndarray,
        iteratively_update_sigma: bool = False,
    ) -> tuple[np.ndarray, float | None]:
        """
        Calculate the optimal sigma for the given target variable.

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
        tuple[np.ndarray, float | None]
            A tuple containing the RBF coefficients and the optimal sigma
            (None if kernel doesn't require optimization).
        """

        # Check if kernel needs sigma optimization
        if not self._needs_sigma_optimization():
            self.logger.info(
                f"Kernel '{self.kernel}' does not require sigma optimization. "
                "Fitting directly with dummy sigma value."
            )
            # Use a dummy sigma value (1.0) for kernels that don't use it
            dummy_sigma = 1.0
            rbf_coeff, _ = self._calc_rbf_coeff(
                sigma=dummy_sigma, x=subset_variables, y=target_variable
            )
            return rbf_coeff, None

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
            )[0]
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

        # Validate sigma value
        self._validate_sigma(
            opt_sigma=opt_sigma,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            subset_variables=subset_variables,
        )

        # Calculate the RBF coefficients for the optimal sigma
        rbf_coeff, A = self._calc_rbf_coeff(
            sigma=opt_sigma, x=subset_variables, y=target_variable
        )

        # Validate fit quality
        self._validate_fit_quality(
            A=A, rbf_coeff=rbf_coeff, target_variable=target_variable
        )

        return rbf_coeff, opt_sigma

    def _rbf_variable_interpolation(
        self,
        opt_sigma: float | None,
        rbf_coeff: np.ndarray,
        normalized_dataset: pd.DataFrame,
        num_points_subset: int,
        num_vars_subset: int,
    ) -> np.ndarray:
        """
        Interpolates the surface for a variable.

        Parameters
        ----------
        opt_sigma : float | None
            The optimal sigma calculated for variable (None if kernel doesn't need it).
        rbf_coeff : np.ndarray
            The fitted coefficients for variable.
        normalized_dataset : pd.DataFrame
            The normalized dataset.
        num_points_subset : int
            The number of points used in the fitting.
        num_vars_subset : int
            The number of variables used in the fitting.

        Returns
        -------
        np.ndarray
            The interpolated variable.
        """

        # Use dummy sigma if None (for kernels that don't need it)
        if opt_sigma is None:
            opt_sigma = 1.0

        # Calculate optimal chunk size based on memory
        norm_dataset = normalized_dataset.values
        norm_subset = self.normalized_subset_data.values

        if self.row_chunks is not None:
            chunks = (min(self.row_chunks, norm_dataset.shape[0]), -1)
            self.logger.info(f"Using row chunks of size {chunks[0]}")
        # elif self.num_workers > 1:
        #     chunks = (norm_dataset.shape[0] // self.num_workers, -1)
        else:
            chunks = (norm_dataset.shape[0], -1)

        # Convert to dask arrays for large operations
        d_dataset = da.from_array(norm_dataset, chunks=chunks)
        d_subset = da.from_array(norm_subset)

        # Split computation into chunks
        result = []
        for i in range(0, len(d_dataset), chunks[0]):
            chunk = d_dataset[i : i + chunks[0]]

            # Calculate r for this chunk
            r_chunk = da.linalg.norm(chunk[:, None, :] - d_subset[None, :, :], axis=2)

            # Apply kernel and dot product
            kernel_values = self.kernel_func(r_chunk, opt_sigma)

            # Compute this chunk's result
            # Get linear coefficients and ensure proper shape for dot product
            linear_coeffs = rbf_coeff[
                num_points_subset + 1 : num_points_subset + 1 + num_vars_subset
            ]
            # For single column case, linear_coeffs is 1D (num_vars_subset,)
            # For multiple columns, it's also 1D (num_vars_subset,)
            # We need to reshape to (num_vars_subset, 1) for matrix multiplication
            # but then squeeze to get (n_chunk,) instead of (n_chunk, 1)
            if linear_coeffs.ndim == 1:
                linear_coeffs_2d = linear_coeffs.reshape(-1, 1)
            else:
                linear_coeffs_2d = linear_coeffs.T

            # Compute linear term: chunk (n_chunk, num_vars_subset) @
            # linear_coeffs_2d (num_vars_subset, 1)
            # Result is (n_chunk, 1), squeeze to (n_chunk,)
            linear_term = da.dot(chunk, linear_coeffs_2d)
            if linear_term.ndim > 1 and linear_term.shape[1] == 1:
                linear_term = linear_term.squeeze(axis=1)

            chunk_result = (
                rbf_coeff[num_points_subset]
                + da.dot(kernel_values, rbf_coeff[:num_points_subset])
                + linear_term
            )

            # Compute and append
            result.append(chunk_result.compute())

        # Combine results
        return np.concatenate(result)

    def _rbf_interpolate(
        self,
        dataset: pd.DataFrame,
        num_workers: int = None,
        target_variable: str = None,
    ) -> pd.DataFrame | np.ndarray:
        """
        Interpolate the dataset.

        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset to interpolate (must have same variables as subset).
        num_workers : int, optional
            The number of workers to use for the interpolation. Default is None.
        target_variable : str, optional
            If provided, only interpolate this target variable and return a numpy array.
            Default is None (interpolate all variables).

        Returns
        -------
        pd.DataFrame | np.ndarray
            If target_variable is None, returns DataFrame with all target variables.
            If target_variable is provided, returns numpy array with predictions for
            that variable only.
        """

        normalized_dataset = self._preprocess_subset_data(
            subset_data=dataset, is_fit=False
        )

        # Get the number of rows and columns in subset and dataset
        num_vars_subset, num_points_subset = self.normalized_subset_data.T.shape
        _, num_points_dataset = normalized_dataset.T.shape

        # If only one target variable requested, return array
        if target_variable is not None:
            interpolated_var = self._rbf_variable_interpolation(
                normalized_dataset=normalized_dataset,
                opt_sigma=self._opt_sigmas[target_variable],
                rbf_coeff=self._rbf_coeffs[target_variable].values,
                num_points_subset=num_points_subset,
                num_vars_subset=num_vars_subset,
            )

            # Denormalize if needed
            if self.is_target_normalized:
                temp_df = pd.DataFrame(
                    {target_variable: interpolated_var}, index=dataset.index
                )
                scale_factor_single = {
                    target_variable: self.target_scale_factor[target_variable]
                }
                temp_df = self.denormalize(
                    normalized_data=temp_df, scale_factor=scale_factor_single
                )
                interpolated_var = temp_df[target_variable].values

            return interpolated_var

        # Initialize the interpolated dataset for all variables
        interpolated_array = np.zeros(
            (num_points_dataset, len(self.target_processed_variables))
        )

        # Loop through the target variables
        if num_workers > 1:
            self.logger.info(
                f"Interpolating target variables using parallel execution "
                f"and num_workers={num_workers}"
            )
            rbf_interpolated_vars = self.parallel_execute(
                func=self._rbf_variable_interpolation,
                items=zip(
                    [
                        self._opt_sigmas[target_var]
                        for target_var in self.target_processed_variables
                    ],
                    [
                        self._rbf_coeffs[target_var].values
                        for target_var in self.target_processed_variables
                    ],
                ),
                num_workers=num_workers,
                normalized_dataset=normalized_dataset,
                num_points_subset=num_points_subset,
                num_vars_subset=num_vars_subset,
            )
            for i_var, interpolated_var in rbf_interpolated_vars.items():
                interpolated_array[:, i_var] = interpolated_var
        else:
            for i_var, target_var in enumerate(self.target_processed_variables):
                self.logger.info(f"Interpolating target variable {target_var}")
                interpolated_var = self._rbf_variable_interpolation(
                    normalized_dataset=normalized_dataset,
                    opt_sigma=self._opt_sigmas[target_var],
                    rbf_coeff=self._rbf_coeffs[target_var].values,
                    num_points_subset=num_points_subset,
                    num_vars_subset=num_vars_subset,
                )
                interpolated_array[:, i_var] = interpolated_var

        return pd.DataFrame(interpolated_array, columns=self.target_processed_variables)

    @validate_data_rbf
    def fit(
        self,
        subset_data: pd.DataFrame,
        target_data: pd.DataFrame,
        subset_directional_variables: list[str] = [],
        target_directional_variables: list[str] = [],
        subset_custom_scale_factor: dict = {},
        normalize_target_data: bool = True,
        target_custom_scale_factor: dict = {},
        num_workers: int = None,
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
        subset_directional_variables : list[str], optional
            The subset directional variables. Default is [].
        target_directional_variables : list[str], optional
            The target directional variables. Default is [].
        subset_custom_scale_factor : dict, optional
            The custom scale factor for the subset data. Default is {}.
        normalize_target_data : bool, optional
            Whether to normalize the target data. Default is True.
        target_custom_scale_factor : dict, optional
            The custom scale factor for the target data. Default is {}.
        num_workers : int, optional
            The number of workers to use for the optimization. Default is None.
        iteratively_update_sigma : bool, optional
            Whether to iteratively update the sigma parameter. Default is False.

        Notes
        -----
        - This function fits the RBF model to the data by:
            1. Preprocessing the subset and target data.
            2. Calculating the optimal sigma for the target variables (skipped for
               kernels that don't require it: linear, cubic, quintic, thin_plate).
            3. Storing the RBF coefficients and optimal sigmas.
        - The number of threads to use for the optimization can be specified.
        - For kernels that don't require sigma optimization, the sigma value in
          opt_sigmas will be None.
        """

        self._subset_directional_variables = subset_directional_variables
        self._target_directional_variables = target_directional_variables
        self._subset_custom_scale_factor = subset_custom_scale_factor
        self._target_custom_scale_factor = target_custom_scale_factor
        # Store original subset_data before preprocessing
        self._original_subset_data = subset_data.copy()
        subset_data = self._preprocess_subset_data(subset_data=subset_data)
        target_data = self._preprocess_target_data(
            target_data=target_data,
            normalize_target_data=normalize_target_data,
        )

        if num_workers is None:
            num_workers = self.num_workers

        self.logger.info("Fitting RBF model to the data")
        # RBF fitting for all variables
        rbf_coeffs, opt_sigmas = {}, {}

        if num_workers > 1:
            self.logger.info(
                f"Fitting RBF model using parallel execution "
                f"and num_workers={num_workers}"
            )
            rbf_coeffs_and_sigmas = self.parallel_execute(
                func=self._calc_opt_sigma,
                items=[
                    target_data[target_var].values for target_var in target_data.columns
                ],
                num_workers=num_workers,
                subset_variables=subset_data.values.T,
                iteratively_update_sigma=iteratively_update_sigma,
            )
            for i_target_var, (rbf_coeff, opt_sigma) in rbf_coeffs_and_sigmas.items():
                target_var = target_data.columns[i_target_var]
                rbf_coeffs[target_var] = rbf_coeff.flatten()
                opt_sigmas[target_var] = opt_sigma
        else:
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

    def predict(self, dataset: pd.DataFrame, num_workers: int = None) -> pd.DataFrame:
        """
        Predicts the data for the provided dataset.

        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset to predict (must have same variables than subset).
        num_workers : int, optional
            The number of workers to use for the interpolation. Default is None.

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

        if num_workers is None:
            num_workers = self.num_workers

        self.logger.info("Reconstructing data using fitted coefficients.")
        interpolated_target = self._rbf_interpolate(
            dataset=dataset, num_workers=num_workers
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
        subset_directional_variables: list[str] = [],
        target_directional_variables: list[str] = [],
        subset_custom_scale_factor: dict = {},
        normalize_target_data: bool = True,
        target_custom_scale_factor: dict = {},
        num_workers: int = None,
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
        subset_directional_variables : list[str], optional
            The subset directional variables. Default is [].
        target_directional_variables : list[str], optional
            The target directional variables. Default is [].
        subset_custom_scale_factor : dict, optional
            The custom scale factor for the subset data. Default is {}.
        normalize_target_data : bool, optional
            Whether to normalize the target data. Default is True.
        target_custom_scale_factor : dict, optional
            The custom scale factor for the target data. Default is {}.
        num_workers : int, optional
            The number of workers to use for the optimization. Default is None.
        iteratively_update_sigma : bool, optional
            Whether to iteratively update the sigma parameter. Default is False.

        Returns
        -------
        pd.DataFrame
            The interpolated dataset.

        Notes
        -----
        - Fits the model to the subset and predicts the interpolated dataset.
        """

        if num_workers is None:
            num_workers = self.num_workers

        self.fit(
            subset_data=subset_data,
            target_data=target_data,
            subset_directional_variables=subset_directional_variables,
            target_directional_variables=target_directional_variables,
            subset_custom_scale_factor=subset_custom_scale_factor,
            normalize_target_data=normalize_target_data,
            target_custom_scale_factor=target_custom_scale_factor,
            num_workers=num_workers,
            iteratively_update_sigma=iteratively_update_sigma,
        )

        return self.predict(dataset=dataset, num_workers=num_workers)

    def explain(
        self,
        dataset: pd.DataFrame,
        target_variable: str = None,
        num_samples: int = 100,
        max_background_samples: int = 100,
    ) -> None:
        """
        Explain RBF predictions using SHAP (SHapley Additive exPlanations) values.

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
        RBFError
            If the model is not fitted.
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
            raise RBFError("RBF model must be fitted before explaining.")

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
        # SHAP will normalize it internally, and _rbf_interpolate will handle
        # preprocessing
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
            # subset_data), then _rbf_interpolate handles preprocessing
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
                # but we need original column structure for _rbf_interpolate
                dataset_df = pd.DataFrame(X, columns=self._original_subset_data.columns)

                # Use _rbf_interpolate with target_variable to get only that variable
                # This handles preprocessing internally (Dir -> Dir_u/Dir_v, normalize)
                # and returns denormalized values
                return self._rbf_interpolate(
                    dataset=dataset_df, target_variable=target_var
                )

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

    def plot_partial_dependence(
        self,
        feature_name: str,
        target_variable: str = None,
        n_points: int = 100,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot partial dependence of a target variable on a single input feature.

        This creates a plot showing how the predicted target variable changes
        as one input feature varies, while other features are held constant.
        The RBF interpolation curve will pass exactly through all training points
        (since RBF is an exact interpolator).

        Parameters
        ----------
        feature_name : str
            Name of the input feature from subset_data to vary.
        target_variable : str, optional
            Target variable to plot. If None, plots the first target variable.
            Default is None.
        n_points : int, optional
            Number of points to evaluate along the feature range. Default is 100.

        Returns
        -------
        tuple
            (fig, ax) matplotlib figure and axes objects.

        Raises
        ------
        RBFError
            If the model is not fitted.
        ValueError
            If feature_name is not in subset_data or target_variable is invalid.
        """

        if not self.is_fitted:
            raise RBFError("RBF model must be fitted before plotting.")

        # Validate feature name
        if feature_name not in self._original_subset_data.columns:
            raise ValueError(
                f"feature_name '{feature_name}' not found in subset_data. "
                f"Available features: {self._original_subset_data.columns.tolist()}"
            )

        # Select target variable
        if target_variable is None:
            target_variable = self.target_processed_variables[0]
        elif target_variable not in self.target_processed_variables:
            raise ValueError(
                f"target_variable '{target_variable}' not found in "
                f"target_processed_variables: {self.target_processed_variables}"
            )

        # Get predictions at the actual training points (using all original features)
        # This ensures the curve passes exactly through training points since RBF
        # is exact
        self.logger.info(
            f"Computing partial dependence for {feature_name} -> {target_variable}"
        )
        training_predictions = self._rbf_interpolate(
            dataset=self._original_subset_data, target_variable=target_variable
        )

        # Get feature values and predictions, sort by feature_name for plotting
        training_x = self._original_subset_data[feature_name].values
        training_y = training_predictions

        # Sort by feature_name for smooth line plotting
        sort_idx = np.argsort(training_x)
        training_x_sorted = training_x[sort_idx]
        training_y_sorted = training_y[sort_idx]

        # Create additional points for smoother visualization
        # Use training data range
        feature_min = training_x_sorted.min()
        feature_max = training_x_sorted.max()
        feature_range = np.linspace(feature_min, feature_max, n_points)

        # Create base DataFrame with median values for other features
        base_data = self._original_subset_data.copy()
        base_data.drop(columns=[feature_name], inplace=True)
        base_data_median = base_data.median()

        # Create smooth grid for visualization
        smooth_data = pd.DataFrame({feature_name: feature_range})
        for col in base_data_median.index:
            smooth_data[col] = base_data_median.loc[col]

        # Get predictions on smooth grid
        smooth_predictions = self._rbf_interpolate(
            dataset=smooth_data, target_variable=target_variable
        )

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the smooth interpolation curve (partial dependence with median values)
        ax.plot(
            feature_range,
            smooth_predictions,
            "b-",
            linewidth=3,
            label="RBF interpolation",
            zorder=2,
        )

        # Plot the training data points connected by line (exact RBF evaluation)
        # This shows the actual RBF interpolation at training points
        ax.plot(
            training_x_sorted,
            training_y_sorted,
            "b-",
            linewidth=2,
            alpha=0.5,
            linestyle="--",
            zorder=1,
        )

        # Plot the training data points (should lie exactly on their own curve)
        ax.scatter(
            training_x,
            training_y,
            c="black",
            marker="+",
            s=150,
            linewidths=2.5,
            label="Training data",
            zorder=3,
            clip_on=False,
        )

        ax.set_xlabel(f"input, {feature_name}", fontsize=12)
        ax.set_ylabel(f"output, {target_variable}", fontsize=12)
        ax.set_title(
            f"Partial Dependence: {feature_name} → {target_variable}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", framealpha=0.9)

        plt.tight_layout()
        plt.show()

        return fig, ax


def basic_rbf_metric(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> float:
    """
    Calculate the basic RBF metric.

    Parameters
    ----------
    df_true : pd.DataFrame
        The true data.
    df_pred : pd.DataFrame
        The predicted data.

    Returns
    -------
    float
        The basic RBF metric.
    """

    return ((df_true - df_pred) ** 2).mean()


def KFold_cross_validation_RBF(
    subset_data: pd.DataFrame,
    target_data: pd.DataFrame,
    subset_directional_variables: list[str] = [],
    target_directional_variables: list[str] = [],
    subset_custom_scale_factor: dict = {},
    normalize_target_data: bool = True,
    target_custom_scale_factor: dict = {},
    num_workers: int = None,
    iteratively_update_sigma: bool = False,
    rbf_model: RBF = None,
    n_splits: int = 5,
    metric: Callable = basic_rbf_metric,
):
    """
    Perform K-Fold cross-validation for the RBF model.

    Parameters
    ----------
    subset_data : pd.DataFrame
        The subset data used to fit the model.
    target_data : pd.DataFrame
        The target data used to fit the model.
    subset_directional_variables : list[str], optional
        The subset directional variables. Default is [].
    target_directional_variables : list[str], optional
        The target directional variables. Default is [].
    subset_custom_scale_factor : dict, optional
        The custom scale factor for the subset data. Default is {}.
    normalize_target_data : bool, optional
        Whether to normalize the target data. Default is True.
    target_custom_scale_factor : dict, optional
        The custom scale factor for the target data. Default is {}.
    num_workers : int, optional
        The number of workers to use for the optimization. Default is None.
    iteratively_update_sigma : bool, optional
        Whether to iteratively update the sigma parameter. Default is False.
    rbf_model : RBF, optional
        The RBF model to use for the cross-validation. Default is None.
    n_splits : int, optional
        The number of splits for the cross-validation. Default is 5.
    metric : Callable, optional
        The metric to use for the cross-validation. Default is basic_rbf_metric.

    Returns
    -------
    dict
        A dictionary containing the results of the cross-validation.
        The keys are the fold indices, and the values are dictionaries containing the
        train and test data, the predictions, and the metric.
    """

    if rbf_model is None:
        rbf_model = RBF()

    # Initialize the K-Fold cross-validation
    kf = KFold(n_splits=n_splits)

    # Loop through the folds
    kfold_results = {}
    for i_fold, (train_index, test_index) in enumerate(
        kf.split(subset_data, target_data)
    ):
        # Get the train and test data
        subset_data_train, subset_data_test = (
            subset_data.iloc[train_index],
            subset_data.iloc[test_index],
        )
        target_data_train, target_data_test = (
            target_data.iloc[train_index],
            target_data.iloc[test_index],
        )

        # Fit the RBF model
        rbf_model.fit(
            subset_data=subset_data_train,
            target_data=target_data_train,
            subset_directional_variables=subset_directional_variables,
            target_directional_variables=target_directional_variables,
            subset_custom_scale_factor=subset_custom_scale_factor,
            normalize_target_data=normalize_target_data,
            target_custom_scale_factor=target_custom_scale_factor,
            num_workers=num_workers,
            iteratively_update_sigma=iteratively_update_sigma,
        )

        # Predict the data
        predictions = rbf_model.predict(
            dataset=subset_data_test, num_workers=num_workers
        )
        predictions.index = target_data_test.index.copy()

        # Calculate directional variables for target data test
        for directional_variable in target_directional_variables:
            (
                target_data_test[f"{directional_variable}_u"],
                target_data_test[f"{directional_variable}_v"],
            ) = rbf_model.get_uv_components(
                x_deg=target_data_test[directional_variable]
            )

        # Store the results
        kfold_results[i_fold] = {
            "subset_data_train": subset_data_train,
            "subset_data_test": subset_data_test,
            "target_data_train": target_data_train,
            "target_data_test": target_data_test,
            "train_index": train_index,
            "test_index": test_index,
            "predictions": predictions,
            "metric": metric(
                target_data_test[rbf_model.target_processed_variables],
                predictions[rbf_model.target_processed_variables],
            ),
        }

    return kfold_results
