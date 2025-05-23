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
    name : str
        The complete name of the distribution (GEV).
    nparams : int
        Number of GEV parameters.
    
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

        AÑADIR MAS METODOS

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

    @property
    def name(
            self
    ) -> str:
        return "Generalized Extreme Value"
    
    @property
    def nparams(self) -> int:
        """
        Number of parameters of GEV
        """
        return int(3)
    
    @staticmethod
    def pdf(
        x: np.ndarray,
        loc: float,
        scale: float,
        shape: float
    ) -> np.ndarray:
        """
        Probability density function
        
        Parameters
        ----------
        x : np.ndarray
            Values to compute the probability density
        loc : float, default=0.0
            Location parameter
        scale : float, default = 1.0
            Scale parameter. 
            Must be greater than 0.
        shape : float
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
        
        y = (x - loc)/scale

        # Gumbel case (shape = 0)
        if shape == 0.0:
            pdf = (1/scale) * (np.exp(-y) * np.exp(-np.exp(-y)))

        # General case (Weibull and Frechet, shape != 0)
        else: 
            pdf = np.full_like(x, 0, dtype=float)   # 0 
            yy = 1 + shape * y
            yymask = yy > 0
            pdf[yymask] = (1/scale) * (yy[yymask] ** (-1 - (1/shape)) * np.exp(-yy[yymask] ** (-1/shape)))

        return pdf

        

        

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
        data: np.ndarray,
        *args
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