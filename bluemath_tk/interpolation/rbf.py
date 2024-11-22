import time
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import fminbound
from ._base_interpolation import BaseInterpolation


def gaussian_kernel(r: float, const: float) -> float:
    """
    This function calculates the Gaussian kernel for the given distance and constant.

    Parameters
    ----------
    r : float
        The distance.
    const : float
        The constant (default name is usually sigma).

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


class RBF(BaseInterpolation):
    """
    Radial Basis Function (RBF) interpolation model.
    This class provides the structure for RBF interpolation models.

    Methods
    -------
    fit(*args, **kwargs)
    predict(*args, **kwargs)
    fit_predict(*args, **kwargs)

    Notes
    -----
    TODO: For the moment, this class only supports optimization for one parameter kernels.
          For this reason, we only have sigma as the parameter to optimize.
          This sigma refers to the sigma parameter in the Gaussian kernel (but is used for all kernels if changed).
    """

    rbf_kernels = {
        "gaussian": gaussian_kernel,
        "multiquadric": multiquadratic_kernel,
        "inverse": inverse_kernel,
        "cubic": cubic_kernel,
        "thin_plate": thin_plate_kernel,
    }

    def __init__(
        self,
        sigma_min: float = 0.001,
        sigma_max: float = 0.1,
        sigma_diff: float = 0.0001,
        kernel: str = "gaussian",
        smooth: float = 1e-5,
    ):
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
        self._kernel = kernel
        self._kernel_func = self.rbf_kernels[self.kernel]
        if not isinstance(smooth, float) or smooth < 0:
            raise ValueError("smooth must be a positive float.")
        self._subset_data: pd.DataFrame = pd.DataFrame()
        self._normalized_subset_data: pd.DataFrame = pd.DataFrame()
        self._target_data: pd.DataFrame = pd.DataFrame()
        self._normalized_target_data: pd.DataFrame = pd.DataFrame()
        self._subset_directional_variables: List[str] = []
        self._target_directional_variables: List[str] = []
        self._custom_scale_factor: dict = {}
        self._scale_factor: dict = {}

    @property
    def sigma_min(self):
        return self._sigma_min

    @property
    def sigma_max(self):
        return self._sigma_max

    @property
    def sigma_diff(self):
        return self._sigma_diff

    @property
    def kernel(self):
        return self._kernel

    @property
    def kernel_func(self):
        return self._kernel_func

    @property
    def smooth(self):
        return self._smooth

    @property
    def subset_data(self):
        return self._subset_data

    @property
    def normalized_subset_data(self):
        return self._normalized_subset_data

    @property
    def target_data(self):
        return self._target_data

    @property
    def normalized_target_data(self):
        return self._normalized_target_data

    @property
    def subset_directional_variables(self):
        return self._subset_directional_variables

    @property
    def target_directional_variables(self):
        return self._target_directional_variables

    @property
    def custom_scale_factor(self):
        return self._custom_scale_factor

    @property
    def scale_factor(self):
        return self._scale_factor

    @staticmethod
    def _get_dir_components(self, x_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function calculates the x and y components for the given directional data.
        """
        
        # Convert degrees to radians and adjust by subtracting from π/2
        x_rad = np.pi / 2 - x_deg * np.pi / 180

        # Adjust angles less than -π by adding 2π
        pos = np.where(x_rad < -np.pi)[0]
        x_rad[pos] = x_rad[pos] + 2 * np.pi

        # Calculate x and y components using cosine and sine
        xx = np.cos(x_rad)
        xy = np.sin(x_rad)

        # Return the x and y components
        return xx, xy

    def _rbf_assemble(self, x: np.ndarray, sigma: float) -> np.ndarray:
        """
        This function assembles the RBF matrix for the given data.
        """

        # Get the number of rows and columns in x
        dim, n = x.shape

        # Fill the matrix with the kernel values
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1):
                r = np.linalg.norm(x[:, i] - x[:, j])
                temp = self.kernel_func(r, sigma)
                A[i, j] = temp
                A[j, i] = temp
            A[i, i] = A[i, i] - self.smooth

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
        """

        # Get the number of rows and columns in x
        m, n = x.shape

        # Assemble the A matrix
        A = self._rbf_assemble(x, sigma)

        # Concatenate y with zeros and reshape
        b = np.concatenate((y, np.zeros((m + 1,)))).reshape(-1, 1)

        # Calculate the RBF coefficients
        rbfcoeff, _, _, _ = np.linalg.lstsq(A, b, rcond=None)  # inverse

        return rbfcoeff, A

    def _cost_sigma(self, sigma: float, x: np.ndarray, y: np.ndarray) -> float:
        """
        This function is called by fminbound to minimize the cost function.
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
        for i in range(m):
            kk = kk - rbf_coeff[m1 - m + i] * x[i, :]

        # Calculate the cost by multiplying invA with kk and normalizing by the diagonal elements of invA
        ceps = np.dot(invA, kk) / np.diagonal(invA)

        # Return the norm of ceps, representing the cost
        yy = np.linalg.norm(ceps)

        return yy

    def _calc_opt_sigma(self, target_variable: np.ndarray) -> float:
        """
        This function calculates the optimal sigma for the given target variable.
        """

        t0 = time.time()
        # Initialize sigma_min, sigma_max, and sigma_diff
        sigma_min, sigma_max, sigma_diff = self.sigma_min, self.sigma_max, 0

        # Loop until sigma_diff is less than the specified sigma_diff
        while sigma_diff < self.sigma_diff:
            opt_sigma = fminbound(
                func=self._cost_eps,
                x1=sigma_min,
                x2=sigma_max,
                args=(self.normalized_subset_data.values.T, target_variable),
                disp=0,
            )
            lm_min = np.abs(opt_sigma - sigma_min)
            lm_max = np.abs(opt_sigma - sigma_max)
            if lm_min < self.sigma_diff:
                sigma_min = sigma_min - sigma_min / 2
            elif lm_max < self.sigma_min:
                sigma_max = sigma_max + sigma_max / 2
            sigma_diff = np.nanmin([lm_min, lm_max])

        # Calculate the time taken to optimize sigma
        t1 = time.time()
        self.logger.info(f"Optimal sigma: {opt_sigma} - Time: {t1 - t0:.2f} seconds")

        return opt_sigma

    # add custom decorator
    def fit(
        self,
        subset_data: pd.DataFrame,
        target_data: pd.DataFrame,
        subset_directional_variables: List[str] = [],
        target_directional_variables: List[str] = [],
        custom_scale_factor: dict = {},
    ):
        self._subset_data = subset_data
        self._target_data = target_data
        self._subset_directional_variables = subset_directional_variables
        self._target_directional_variables = target_directional_variables
        self._custom_scale_factor = custom_scale_factor
        self._normalized_subset_data, self._scale_factor = self.normalize(
            data=self.subset_data, custom_scale_factor=self.custom_scale_factor
        )
        self._normalized_target_data, _ = self.normalize(
            data=self.target_data, custom_scale_factor=self.scale_factor
        )

        # RBF scalar variables
        rbf_coeffs, opt_sigmas = [], []
        output_scalar, output_dir_x, output_dir_y = [], [], []

        for target_var in self.target_data.columns:
            if target_var in self.target_directional_variables:
                self.logger.info(
                    f"Fitting RBF for directional variable {target_var}. Skipping..."
                )
                continue
            else:
                self.logger.info(f"Fitting RBF for scalar variable {target_var}")
                target_variable = self.target_data[target_var]
                opt_sigma = self._calc_opt_sigma(target_variable=target_variable.values)
                rbf_coeff, _ = self._calc_rbf_coeff(
                    ep=opt_sigma, x=self.normalized_subset_data.T, y=v
                )
                output_scalar.append(np.reshape(rbf_coeff, -1))
                opt_sigmas.append(opt_sigma)

        # Fit the scalar variables
        for ix in ix_scalar_target:
            print(f"Calibrating scalar {ix}")
            v = target[:, ix]

            # minimize RBF cost function
            t0 = time.time()  # time counter

            # ensure that sigma opt is in the bounds of [sigma_min, sigma_max]
            # parameters
            sigma_min, sigma_max, d_sigma = 0.001, 0.1, 0

            print(
                "\rScalar {0}: Range sigma {1:.4f}-{2:.4f} - Opt sigma {3:.4f}".format(
                    ix, sigma_min, sigma_max, opt_sigma
                )
            )

            t1 = time.time()  # optimization time

            # calculate RBF coeff
            rbf_coeff, _ = calc_rbf_coeff(opt_sigma, subset_norm.T, v)
            output_scalar.append(np.reshape(rbf_coeff, -1))

            # rbf_coeffs.append(rbf_coeff)
            opt_sigmas.append(opt_sigma)

        # RBF directional variables
        opt_sigma_xs, opt_sigma_ys = [], []

        for ix in ix_directional_target:
            print(f"Calibrating directional {ix}")
            v = target[:, ix]

            # x and y directional variable components
            vdg = np.pi / 2 - v * np.pi / 180
            pos = np.where(vdg < -np.pi)[0]
            vdg[pos] = vdg[pos] + 2 * np.pi
            vdx = np.cos(vdg)
            vdy = np.sin(vdg)

            # minimize RBF cost function
            t0 = time.time()  # time counter

            # directional x

            sigma_min, sigma_max, d_sigma = 0.001, 0.1, 0
            while d_sigma < 0.0001:
                opt_sigma_x = fminbound(
                    cost_eps, sigma_min, sigma_max, args=(subset_norm.T, vdx)
                )
                lm_min = np.abs(opt_sigma_x - sigma_min)
                lm_max = np.abs(opt_sigma_x - sigma_max)

                if lm_min < 0.0001:
                    sigma_min = sigma_min - sigma_min / 2

                elif lm_max < 0.001:
                    sigma_max = sigma_max + sigma_max / 2

                d_sigma = np.nanmin([lm_min, lm_max])

            print(
                "\rDirectional x {0}: Range sigma {1:.4f}-{2:.4f} - Opt sigma {3:.4f}".format(
                    ix, sigma_min, sigma_max, opt_sigma_x
                )
            )

            # directional y
            sigma_min, sigma_max, d_sigma = 0.001, 0.1, 0
            while d_sigma < 0.0001:
                opt_sigma_y = fminbound(
                    cost_eps, sigma_min, sigma_max, args=(subset_norm.T, vdy)
                )
                lm_min = np.abs(opt_sigma_y - sigma_min)
                lm_max = np.abs(opt_sigma_y - sigma_max)

                if lm_min < 0.0001:
                    sigma_min = sigma_min - sigma_min / 2

                elif lm_max < 0.001:
                    sigma_max = sigma_max + sigma_max / 2

                d_sigma = np.nanmin([lm_min, lm_max])

            print(
                "\rDirectional x {0}: Range sigma {1:.4f}-{2:.4f} - Opt sigma {3:.4f}".format(
                    ix, sigma_min, sigma_max, opt_sigma_y
                )
            )

            t1 = time.time()  # optimization time

            # calculate RBF coeff
            rbf_coeff_x, _ = calc_rbf_coeff(opt_sigma_x, subset_norm.T, vdx)
            rbf_coeff_y, _ = calc_rbf_coeff(opt_sigma_y, subset_norm.T, vdy)
            output_dir_x.append(np.reshape(rbf_coeff_x, -1))
            output_dir_y.append(np.reshape(rbf_coeff_y, -1))

            opt_sigma_xs.append(opt_sigma_x)
            opt_sigma_ys.append(opt_sigma_y)

        if output_scalar != []:
            df_rbf_scalar = pd.DataFrame(np.transpose(output_scalar))
        else:
            df_rbf_scalar = pd.DataFrame()
            opt_sigmas = []

        if output_dir_x != []:
            df_rbf_dirx = pd.DataFrame(np.transpose(output_dir_x))
            df_rbf_diry = pd.DataFrame(np.transpose(output_dir_y))
        else:
            df_rbf_dirx = pd.DataFrame()
            df_rbf_diry = pd.DataFrame()
            opt_sigma_x, opt_sigma_y = [], []

        self.df_rbf_scalar = df_rbf_scalar
        self.opt_sigmas_scalar = opt_sigmas

        self.df_rbf_dir_x = df_rbf_dirx
        self.df_rbf_dir_y = df_rbf_diry
        self.opt_sigmas_dir_x = opt_sigma_xs
        self.opt_sigmas_dir_y = opt_sigma_ys

    def predict(self, *args, **kwargs):
        """
        Predicts the clusters for the provided data.

        Parameters
        ----------
        *args : list
            Positional arguments.
        **kwargs : dict
            Keyword arguments.
        """

        pass

    def fit_predict(self, *args, **kwargs):
        """
        Fits the model to the data and predicts the clusters.

        Parameters
        ----------
        *args : list
            Positional arguments.
        **kwargs : dict
            Keyword arguments.
        """

        pass
