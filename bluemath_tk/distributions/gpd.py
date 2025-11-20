from typing import Dict, List

import numpy as np
from scipy.stats import genpareto

from ._base_distributions import BaseDistribution, FitResult


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
    pdf(x, threshold, scale, shape)
        Probability density function.
    cdf(x, threshold, scale, shape)
        Cumulative distribution function
    qf(p, threshold, scale, shape)
        Quantile function
    sf(x, threshold, scale, shape)
        Survival function
    nll(data, threshold, scale, shape)
        Negative Log-Likelihood function
    fit(data)
        Fit distribution to data (NOT IMPLEMENTED).
    random(size, threshold, scale, shape)
        Generates random values from GPD distribution.
    mean(threshold, scale, shape)
        Mean of GPD distribution.
    median(threshold, scale, shape)
        Median of GPD distribution.
    variance(threshold, scale, shape)
        Variance of GPD distribution.
    std(threshold, scale, shape)
        Standard deviation of GPD distribution.
    stats(threshold, scale, shape)
        Summary statistics of GPD distribution.

    Notes
    -----
    - This class is designed to obtain all the properties associated to the GPD distribution.

    Examples
    --------
    >>> from bluemath_tk.distributions.gpd import GPD
    >>> gpd_pdf = GPD.pdf(x, threshold=0, scale=1, shape=0.1)
    >>> gpd_cdf = GPD.cdf(x, threshold=0, scale=1, shape=0.1)
    >>> gpd_qf = GPD.qf(p, threshold=0, scale=1, shape=0.1)
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
        return ["threshold", "scale", "shape"]

    @staticmethod
    def pdf(
        x: np.ndarray, threshold: float = 0.0, scale: float = 1.0, shape: float = 0.0
    ) -> np.ndarray:
        """
        Probability density function

        Parameters
        ----------
        x : np.ndarray
            Values to compute the probability density value
        threshold : float, default=0.0
            Threshold parameter
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

        return genpareto.pdf(x, c=shape, loc=threshold, scale=scale)

    @staticmethod
    def cdf(
        x: np.ndarray, threshold: float = 0.0, scale: float = 1.0, shape: float = 0.0
    ) -> np.ndarray:
        """
        Cumulative distribution function

        Parameters
        ----------
        x : np.ndarray
            Values to compute their probability
        threshold : float, default=0.0
            Threshold parameter
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

        return genpareto.cdf(x, c=shape, loc=threshold, scale=scale)

    @staticmethod
    def sf(
        x: np.ndarray, threshold: float = 0.0, scale: float = 1.0, shape: float = 0.0
    ) -> np.ndarray:
        """
        Survival function (1-Cumulative Distribution Function)

        Parameters
        ----------
        x : np.ndarray
            Values to compute their survival function value
        threshold : float, default=0.0
            Threshold parameter
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

        return genpareto.sf(x, c=shape, loc=threshold, scale=scale)

    @staticmethod
    def qf(
        p: np.ndarray, threshold: float = 0.0, scale: float = 1.0, shape: float = 0.0
    ) -> np.ndarray:
        """
        Quantile function (Inverse of Cumulative Distribution Function)

        Parameters
        ----------
        p : np.ndarray
            Probabilities to compute their quantile
        threshold : float, default=0.0
            Threshold parameter
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

        return genpareto.ppf(p, c=shape, loc=threshold, scale=scale)

    @staticmethod
    def nll(
        data: np.ndarray, threshold: float = 0.0, scale: float = 1.0, shape: float = 0.0
    ) -> float:
        """
        Negative Log-Likelihood function

        Parameters
        ----------
        data : np.ndarray
            Data to compute the Negative Log-Likelihood value
        threshold : float, default=0.0
            Threshold parameter
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
            return np.inf
        if np.any(data < threshold):
            return np.inf

        return -np.sum(
            genpareto.logpdf(data, c=shape, loc=threshold, scale=scale), axis=0
        )

    @staticmethod
    def fit(data: np.ndarray, threshold: float, **kwargs) -> FitResult:
        """
        Fit GEV distribution

        Parameters
        ----------
        data : np.ndarray
            Data to fit the GEV distribution
        threshold : float, default=0.0
            Threshold parameter
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
        exceedances = data[data > threshold]

        if len(exceedances) == 0:
            raise ValueError("No exceedances above threshold")

        # Adjust exceedances relative to threshold
        exceedances_adjusted = exceedances - threshold

        # Default optimization settings
        default_x0 = [np.std(exceedances_adjusted), 0.1]
        x0 = kwargs.get("x0", default_x0)
        method = kwargs.get("method", "Nelder-Mead")
        default_bounds = [(1e-6, None), (None, None)]
        bounds = kwargs.get("bounds", default_bounds)
        options = kwargs.get("options", {"disp": False})

        # Objective function that includes threshold
        def obj(params):
            scale, shape = params
            return GPD.nll(exceedances, threshold=threshold, scale=scale, shape=shape)

        # Perform optimization
        from scipy.optimize import OptimizeResult, minimize

        result = minimize(fun=obj, x0=x0, method=method, bounds=bounds, options=options)

        # Create modified result
        modified_result = OptimizeResult(
            x=[threshold, result.x[0], result.x[1]],
            success=result.success,
            fun=result.fun,
            message=result.message,
            hess_inv=result.hess_inv if hasattr(result, "hess_inv") else None,
        )

        # Return the fitting result
        return FitResult(GPD, exceedances, modified_result)

    @staticmethod
    def random(
        size: int,
        threshold: float = 0.0,
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
        threshold : float, default=0.0
            Threshold parameter
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

        return genpareto.rvs(
            c=shape, loc=threshold, scale=scale, size=size, random_state=random_state
        )

    @staticmethod
    def mean(threshold: float = 0.0, scale: float = 1.0, shape: float = 0.0) -> float:
        """
        Mean

        Parameters
        ----------
        threshold : float, default=0.0
            Threshold parameter
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

        return genpareto.mean(c=shape, loc=threshold, scale=scale)

    @staticmethod
    def median(threshold: float = 0.0, scale: float = 1.0, shape: float = 0.0) -> float:
        """
        Median

        Parameters
        ----------
        threshold : float, default=0.0
            Threshold parameter
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

        return genpareto.median(c=shape, loc=threshold, scale=scale)

    @staticmethod
    def variance(
        threshold: float = 0.0, scale: float = 1.0, shape: float = 0.0
    ) -> float:
        """
        Variance

        Parameters
        ----------
        threshold : float, default=0.0
            Threshold parameter
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

        return genpareto.var(c=shape, loc=threshold, scale=scale)

    @staticmethod
    def std(threshold: float = 0.0, scale: float = 1.0, shape: float = 0.0) -> float:
        """
        Standard deviation

        Parameters
        ----------
        threshold : float, default=0.0
            Threshold parameter
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

        return genpareto.std(c=shape, loc=threshold, scale=scale)

    @staticmethod
    def stats(
        threshold: float = 0.0, scale: float = 1.0, shape: float = 0.0
    ) -> Dict[str, float]:
        """
        Summary statistics

        Return summary statistics including mean, std, variance, etc.

        Parameters
        ----------
        threshold : float, default=0.0
            Threshold parameter
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
            "mean": float(GPD.mean(threshold, scale, shape)),
            "median": float(GPD.median(threshold, scale, shape)),
            "variance": float(GPD.variance(threshold, scale, shape)),
            "std": float(GPD.std(threshold, scale, shape)),
        }

        return stats
