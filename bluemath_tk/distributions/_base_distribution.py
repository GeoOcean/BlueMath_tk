from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..core.models import BlueMathModel

class BaseDistribution(BlueMathModel):
    """
    Base class for all extreme distributions.
    """
    
    @property
    @abstractmethod
    def name() -> str:
        return str

    @staticmethod
    @abstractmethod
    def pdf(
        x: np.ndarray
    ) -> np.ndarray:
        """
        Probability density function
        """
        return np.ndarray()

    @staticmethod
    @abstractmethod
    def cdf(
        x: np.ndarray
    ) -> np.ndarray:
        """
        Cumulative distribution function
        """
        return np.ndarray()

    @staticmethod
    @abstractmethod
    def sf(
        x: np.ndarray
    ) -> np.ndarray:
        """
        Survival function (1 - cdf)
        """
        return np.ndarray()

    @staticmethod
    @abstractmethod
    def qf(
        p: np.ndarray
    ) -> np.ndarray:
        """
        Quantile function
        """
        return np.ndarray()

    @staticmethod
    @abstractmethod
    def loglike(
        x: np.ndarray
    ) -> float:
        """
        Loglikelihood function
        """
        return np.ndarray()

    @staticmethod
    @abstractmethod
    def fit(
        data: np.ndarray
    ):
        """
        Fit distribution
        """
        return np.ndarray()
    
    @staticmethod
    @abstractmethod
    def random(
        data: np.ndarray,
        size: int
    ) -> np.ndarray:
        """
        Generate random values
        """
        return np.ndarray()

    @staticmethod
    @abstractmethod
    def mean(
    ) -> float:
        """
        Mean
        """
        return float
    
    @staticmethod
    @abstractmethod
    def median(
    ) -> float:
        """
        Median
        """
        return float
    
    @staticmethod
    @abstractmethod
    def variance(
    ) -> float:
        """
        Variance 
        """
        return float
    
    @staticmethod
    @abstractmethod
    def std(
    ) -> float:
        """
        Standard deviation
        """
        return float
    
    @staticmethod
    @abstractmethod
    def stats(
    ) -> float:
        """
        Stats
        """
        return float


    