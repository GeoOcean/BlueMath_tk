from typing import Dict, List

import numpy as np

from ._base_distributions import BaseDistribution, FitResult, fit_dist


class GPD(BaseDistribution):
    """
    Generalized Pareto Distribution (GPD) class.

    This class contains all the methods assocaited to the GPD distribution.

    Attributes
    ----------
    name : str
        The complete name of the distribution (GPD).
    nparams : int
        Number of GPD parameters.
    param_names : List[str]
        Names of the GPD parameters (threshold, scale, shape).

    Methods
    -------
    pdf(x, loc, scale, shape)
        Probability density function.
    cdf(x, loc, scale, shape)
        Cumulative distribution function
    qf(p, loc, scale, shape)
        Quantile function
    sf(x, loc, scale, shape)
        Survival function
    nll(data, loc, scale, shape)
        Negative Log-Likelihood function
    fit(data)
        Fit distribution to data (NOT IMPLEMENTED).
    random(size, loc, scale, shape)
        Generates random values from GPD distribution.
    mean(loc, scale, shape)
        Mean of GPD distribution.
    median(loc, scale, shape)
        Median of GPD distribution.
    variance(loc, scale, shape)
        Variance of GPD distribution.
    std(loc, scale, shape)
        Standard deviation of GPD distribution.
    stats(loc, scale, shape)
        Summary statistics of GPD distribution.

    Notes
    -----
    - This class is designed to obtain all the properties associated to the GPD distribution.

    Examples
    --------
    >>> from bluemath_tk.distributions.gpd import GPD
    >>> gpd_pdf = GPD.pdf(x, loc=0, scale=1, shape=0.1)
    >>> gpd_cdf = GPD.cdf(x, loc=0, scale=1, shape=0.1)
    >>> gpd_qf = GPD.qf(p, loc=0, scale=1, shape=0.1)
    """

    def __init__(self) -> None:
        """
        Initialize the GPD distribution class
        """
        super().__init__()

    @staticmethod
    def name() -> str:
        return "Generalized Pareto Distribution (GPD)"

    @staticmethod
    def nparams() -> int:
        """
        Number of parameters of GPD
        """
        return int(3)

    @staticmethod
    def param_names() -> List[str]:
        """
        Name of parameters of GPD
        """
        return ["loc", "scale", "shape"]

    @staticmethod
    def pdf(
        x: np.ndarray, loc: float = 0.0, scale: float = 1.0, shape: float = 0.0
    ) -> np.ndarray:
        """
        Probability density function

        Parameters
        ----------
        x : np.ndarray
            Values to compute the probability density value
        loc : float, default=0.0
            Location parameter
        scale : float, default = 1.0
            Scale parameter.
            Must be greater than 0.
        shape : float, default = 0.0
            Shape parameter.

        Returns
        ----------
        pdf : np.ndarray
            Probability density function values

        Raises
        ------
        ValueError
            If scale is not greater than 0.
        """

        if scale <= 0:
            raise ValueError("Scale parameter must be > 0")

        y = np.maximum(x - loc, 0) / scale

        # Gumbel case (shape = 0)
        if shape == 0.0:
            pdf = (1 / scale) * (np.exp(-y))

        # General case (Weibull and Frechet, shape != 0)
        else:
            pdf = np.full_like(x, 0, dtype=float)
            yy = 1 + shape * y
            yymask = yy > 0
            pdf[yymask] = (1 / scale) * (yy[yymask] ** (-1 - (1 / shape)))

        return pdf

    @staticmethod
    def cdf(
        x: np.ndarray, loc: float = 0.0, scale: float = 1.0, shape: float = 0.0
    ) -> np.ndarray:
        """
        Cumulative distribution function

        Parameters
        ----------
        x : np.ndarray
            Values to compute their probability
        loc : float, default=0.0
            Location parameter
        scale : float, default = 1.0
            Scale parameter.
            Must be greater than 0.
        shape : float, default = 0.0
            Shape parameter.

        Returns
        ----------
        p : np.ndarray
            Probability

        Raises
        ------
        ValueError
            If scale is not greater than 0.
        """

        if scale <= 0:
            raise ValueError("Scale parameter must be > 0")

        y = np.maximum(x - loc, 0) / scale

        # Gumbel case (shape = 0)
        if shape == 0.0:
            p = 1 - np.exp(-y)

        # General case (Weibull and Frechet, shape != 0)
        else:
            p = 1 - np.maximum(1 + shape * y, 0) ** (-1 / shape)

        return p

    @staticmethod
    def sf(
        x: np.ndarray, loc: float = 0.0, scale: float = 1.0, shape: float = 0.0
    ) -> np.ndarray:
        """
        Survival function (1-Cumulative Distribution Function)

        Parameters
        ----------
        x : np.ndarray
            Values to compute their survival function value
        loc : float, default=0.0
            Location parameter
        scale : float, default = 1.0
            Scale parameter.
            Must be greater than 0.
        shape : float, default = 0.0
            Shape parameter.

        Returns
        ----------
        sp : np.ndarray
            Survival function value

        Raises
        ------
        ValueError
            If scale is not greater than 0.
        """

        if scale <= 0:
            raise ValueError("Scale parameter must be > 0")

        sp = 1 - GPD.cdf(x, loc=loc, scale=scale, shape=shape)

        return sp

    @staticmethod
    def qf(
        p: np.ndarray, loc: float = 0.0, scale: float = 1.0, shape: float = 0.0
    ) -> np.ndarray:
        """
        Quantile function (Inverse of Cumulative Distribution Function)

        Parameters
        ----------
        p : np.ndarray
            Probabilities to compute their quantile
        loc : float, default=0.0
            Location parameter
        scale : float, default = 1.0
            Scale parameter.
            Must be greater than 0.
        shape : float, default = 0.0
            Shape parameter.

        Returns
        ----------
        q : np.ndarray
            Quantile value

        Raises
        ------
        ValueError
            If probabilities are not in the range (0, 1).

        ValueError
            If scale is not greater than 0.
        """

        if np.min(p) <= 0 or np.max(p) >= 1:
            raise ValueError("Probabilities must be in the range (0, 1)")

        if scale <= 0:
            raise ValueError("Scale parameter must be > 0")

        # Gumbel case (shape = 0)
        if shape == 0.0:
            q = loc - scale * np.log(1 - p)

        # General case (Weibull and Frechet, shape != 0)
        else:
            q = loc + scale * ((1 - p) ** (-shape) - 1) / shape

        return q

    @staticmethod
    def nll(
        data: np.ndarray, loc: float = 0.0, scale: float = 1.0, shape: float = 0.0
    ) -> float:
        """
        Negative Log-Likelihood function

        Parameters
        ----------
        data : np.ndarray
            Data to compute the Negative Log-Likelihood value
        loc : float, default=0.0
            Location parameter
        scale : float, default = 1.0
            Scale parameter.
            Must be greater than 0.
        shape : float, default = 0.0
            Shape parameter.

        Returns
        ----------
        nll : float
            Negative Log-Likelihood value
        """

        if scale <= 0:
            nll = np.inf  # Return a large value for invalid scale

        else:
            y = (data - loc) / scale

            # # Gumbel case (shape = 0)
            # if shape == 0.0:
            #     nll = data.shape[0] * np.log(scale) + np.sum(y)

            # General case (Weibull and Frechet, shape != 0)
            # else:
            shape = (
                np.maximum(shape, 1e-8) if shape > 0 else np.minimum(shape, -1e-8)
            )  # Avoid division by zero
            y = 1 + shape * y
            if np.min(y <= 0):
                nll = np.inf  # Return a large value for invalid y
            else:
                nll = data.shape[0] * np.log(scale) + (1 / shape + 1) * np.sum(
                    np.log(y)
                )

        return nll

    @staticmethod
    def fit(data: np.ndarray, **kwargs) -> FitResult:
        """
        Fit GEV distribution

        Parameters
        ----------
        data : np.ndarray
            Data to fit the GEV distribution
        **kwargs : dict, optional
            Additional keyword arguments for the fitting function.
            These can include options like method, bounds, etc.
            See fit_dist for more details.
            If not provided, default fitting options will be used.

        Returns
        ----------
        FitResult
            Result of the fit containing the parameters loc, scale, shape,
            success status, and negative log-likelihood value.
        """
        # Fit the GEV distribution to the data using the fit_dist function
        return fit_dist(GPD, data, **kwargs)

    @staticmethod
    def random(
        size: int,
        loc: float = 0.0,
        scale: float = 1.0,
        shape: float = 0.0,
        random_state: int = None,
    ) -> np.ndarray:
        """
        Generates random values from GPD distribution

        Parameters
        ----------
        size : int
            Number of random values to generate
        loc : float, default=0.0
            Location parameter
        scale : float, default = 1.0
            Scale parameter.
            Must be greater than 0.
        shape : float, default = 0.0
            Shape parameter.
        random_state : np.random.RandomState, optional
            Random state for reproducibility.
            If None, do not use random stat.

        Returns
        ----------
        x : np.ndarray
            Random values from GEV distribution

        Raises
        ------
        ValueError
            If scale is not greater than 0.
        """

        if scale <= 0:
            raise ValueError("Scale parameter must be > 0")

        # Set random state if provided
        if random_state is not None:
            np.random.seed(random_state)

        # Generate uniform random numbers
        u = np.random.uniform(0, 1, size)

        # Gumbel case (shape = 0)
        if shape == 0.0:
            x = loc - scale * np.log(u)

        # General case (Weibull and Frechet, shape != 0)
        else:
            x = loc + scale * (u ** (-shape) - 1) / shape

        return x

    @staticmethod
    def mean(loc: float = 0.0, scale: float = 1.0, shape: float = 0.0) -> float:
        """
        Mean

        Parameters
        ----------
        loc : float, default=0.0
            Location parameter
        scale : float, default = 1.0
            Scale parameter.
            Must be greater than 0.
        shape : float, default = 0.0
            Shape parameter.

        Returns
        ----------
        mean : np.ndarray
            Mean value of GEV with the given parameters

        Raises
        ------
        ValueError
            If scale is not greater than 0.

        Warning
            If shape is greater than or equal to 1, mean is not defined.
            In this case, it returns infinity.
        """

        if scale <= 0:
            raise ValueError("Scale parameter must be > 0")

        if shape >= 1:
            Warning("Shape parameter must be < 1 for mean to be defined")
            mean = np.inf

        # Shape < 1 case
        else:
            mean = scale / (1 - shape)

        return mean

    @staticmethod
    def median(loc: float = 0.0, scale: float = 1.0, shape: float = 0.0) -> float:
        """
        Median

        Parameters
        ----------
        loc : float, default=0.0
            Location parameter
        scale : float, default = 1.0
            Scale parameter.
            Must be greater than 0.
        shape : float, default = 0.0
            Shape parameter.

        Returns
        ----------
        median : np.ndarray
            Median value of GEV with the given parameters

        Raises
        ------
        ValueError
            If scale is not greater than 0.
        """

        if scale <= 0:
            raise ValueError("Scale parameter must be > 0")

        if shape == 0:
            median = np.inf
        else:
            median = loc + scale * (2**shape - 1) / shape

        return median

    @staticmethod
    def variance(loc: float = 0.0, scale: float = 1.0, shape: float = 0.0) -> float:
        """
        Variance

        Parameters
        ----------
        loc : float, default=0.0
            Location parameter
        scale : float, default = 1.0
            Scale parameter.
            Must be greater than 0.
        shape : float, default = 0.0
            Shape parameter.

        Returns
        ----------
        var : np.ndarray
            Variance of GEV with the given parameters

        Raises
        ------
        ValueError
            If scale is not greater than 0.
        Warning
            If shape is greater than or equal to 172, mean is not defined.
            In this case, it returns infinity.
        """

        if scale <= 0:
            raise ValueError("Scale parameter must be > 0")

        # Gumbel case (shape = 0)
        if shape >= 1 / 2:
            Warning("Shape parameter must be < 1/2 for variance to be defined")
            var = np.inf

        else:
            var = scale**2 / ((1 - shape) ** 2 * (1 - 2 * shape))

        return var

    @staticmethod
    def std(loc: float = 0.0, scale: float = 1.0, shape: float = 0.0) -> float:
        """
        Standard deviation

        Parameters
        ----------
        loc : float, default=0.0
            Location parameter
        scale : float, default = 1.0
            Scale parameter.
            Must be greater than 0.
        shape : float, default = 0.0
            Shape parameter.

        Returns
        ----------
        std : np.ndarray
            Standard Deviation of GEV with the given
            parameters

        Raises
        ------
        ValueError
            If scale is not greater than 0.
        """

        if scale <= 0:
            raise ValueError("Scale parameter must be > 0")

        std = np.sqrt(GPD.variance(loc, scale, shape))

        return std

    @staticmethod
    def stats(
        loc: float = 0.0, scale: float = 1.0, shape: float = 0.0
    ) -> Dict[str, float]:
        """
        Summary statistics

        Return summary statistics including mean, std, variance, etc.

        Parameters
        ----------
        loc : float, default=0.0
            Location parameter
        scale : float, default = 1.0
            Scale parameter.
            Must be greater than 0.
        shape : float, default = 0.0
            Shape parameter.

        Returns
        ----------
        stats : dict
            Summary statistics of GEV distribution with the given
            parameters

        Raises
        ------
        ValueError
            If scale is not greater than 0.
        """

        if scale <= 0:
            raise ValueError("Scale parameter must be > 0")

        stats = {
            "mean": float(GPD.mean(loc, scale, shape)),
            "median": float(GPD.median(loc, scale, shape)),
            "variance": float(GPD.variance(loc, scale, shape)),
            "std": float(GPD.std(loc, scale, shape)),
        }

        return stats
