from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

class BaseDistribution(ABC):
    """
    Base class for all extreme distributions.
    """
    
    @property
    @abstractmethod
    def name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def pdf(
        x: np.ndarray
    ) -> np.ndarray:
        """
        Probability density function
        """
        pass

    @staticmethod
    @abstractmethod
    def cdf(
        x: np.ndarray
    ) -> np.ndarray:
        """
        Cumulative distribution function
        """
        pass

    @staticmethod
    @abstractmethod
    def sf(
        x: np.ndarray
    ) -> np.ndarray:
        """
        Survival function (1 - cdf)
        """
        pass

    @staticmethod
    @abstractmethod
    def qf(
        p: np.ndarray
    ) -> np.ndarray:
        """
        Quantile function
        """
        pass

    @staticmethod
    @abstractmethod
    def loglike(
        x: np.ndarray
    ) -> float:
        """
        Loglikelihood function
        """
        pass

    @staticmethod
    @abstractmethod
    def fit(
        data: np.ndarray
    ):
        """
        Fit distribution
        """
        pass
    
    @staticmethod
    @abstractmethod
    def random(
        data: np.ndarray,
        size: int
    ) -> np.ndarray:
        """
        Generate random values
        """
        pass

    @staticmethod
    @abstractmethod
    def mean(
    ) -> float:
        """
        Mean
        """
        pass
    
    @staticmethod
    @abstractmethod
    def median(
    ) -> float:
        """
        Median
        """
        pass
    
    @staticmethod
    @abstractmethod
    def variance(
    ) -> float:
        """
        Variance 
        """
        pass
    
    @staticmethod
    @abstractmethod
    def std(
    ) -> float:
        """
        Standard deviation
        """
        pass
    
    @staticmethod
    @abstractmethod
    def stats(
    ) -> float:
        """
        Stats
        """
        pass


    