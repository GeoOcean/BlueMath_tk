from typing import Tuple

import numpy as np
import pandas as pd

from ._base_distributions import BaseDistribution


class gev(BaseDistribution):
    """
    Generalized Extreme Value (GEV) distribution class.

    This class contains all the methods assocaited to the GEV distribution.

    Attributes
    ----------
    

    Methods
    -------
    generate(dimensions_names, lower_bounds, upper_bounds, num_samples)
        Generate LHS samples.

    Notes
    -----
    - This class is designed to obtain all the properties associated to the GEV distribution.

    Examples
    --------
    >>> from bluemath_tk.distributions.gev import gev
    >>> gev_pdf = gev.pdf(x, loc=0, scale=1, shape=0.1)
    >>> gev_cdf = gev.cdf(x, loc=0, scale=1, shape=0.1)
    >>> gev_qf = gev.qf(p, loc=0, scale=1, shape=0.1)
    """

    def __init__(
            self
    ) -> None:
        """
        Initialize the GEV distribution class
        """
        super().__init__()

    def name(
            self
    ) -> str:
        return "Generalized Extreme Value"
    
    @staticmethod
    def pdf(
        x: np.ndarray
    ) -> np.ndarray:
        """
        Probability density function
        """
        pass

    @staticmethod
    def cdf(
        x: np.ndarray
    ) -> np.ndarray:
        """
        Cumulative distribution function
        """
        pass

    @staticmethod
    def sf(
        x: np.ndarray
    ) -> np.ndarray:
        """
        Survival function (1 - cdf)
        """
        pass

    @staticmethod
    def qf(
        p: np.ndarray
    ) -> np.ndarray:
        """
        Quantile function
        """
        pass

    @staticmethod
    def nll(
        x: np.ndarray
    ) -> float:
        """
        Negative Log-Likelihood function
        """
        pass

    @staticmethod
    def fit(
        data: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Fit distribution
        """
        pass
    
    @staticmethod
    def random(
        data: np.ndarray,
        size: int
    ) -> np.ndarray:
        """
        Generate random values
        """
        pass

    @staticmethod
    def mean(
    ) -> float:
        """
        Mean
        """
        pass
    
    @staticmethod
    def median(
    ) -> float:
        """
        Median
        """
        pass
    
    @staticmethod
    def variance(
    ) -> float:
        """
        Variance 
        """
        pass
    
    @staticmethod
    def std(
    ) -> float:
        """
        Standard deviation
        """
        pass
    
    @staticmethod
    def stats(
    ) -> dict:
        """
        Return summary statistics including mean, std, variance, etc.
        """
        pass