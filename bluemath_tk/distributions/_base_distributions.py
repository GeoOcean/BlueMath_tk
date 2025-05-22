from abc import abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..core.models import BlueMathModel


class BaseDistribution(BlueMathModel):
    """
    Base class for all extreme distributions.
    """

    @abstractmethod
    def __init__(
        self
    ) -> None:
        """
        Initialize the base distribution class
        """
        super().__init__()
    
    @property
    @abstractmethod
    def name(self) -> str:
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
    def nll(
        x: np.ndarray
    ) -> float:
        """
        Negative Log-Likelihood function
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
    ) -> dict:
        """
        Return summary statistics including mean, std, variance, etc.
        """
        pass

    def fit(
        self,
        data: np.ndarray,
        *args,
        **kwargs
    ) -> Tuple[float, float, float]:
        """
        Fit distribution
        """
        fitter = FitClass(self, data, *args, **kwargs)
        return fitter.run()




class FitResult(BlueMathModel):
    """
    Class used for the results of fitting a distribution
    """
    def __init__(self, params, success, message):
        self.params = params
        self.success = success
        self.message = message

    def summary(self):
        return {
            'parameters': self.params,
            'success': self.success,
            'message': self.message
        }


class FitClass(BlueMathModel):
    """
    Class used to fit the distributions
    """
    def __init__(self, dist, data, *args, **kwargs):
        super().__init__()

        self.dist = dist
        self.data = data
        self.args = args
        self.kwargs = kwargs

    def run(self):
        
        initial_guess = self.kwargs.get('initial_guess', [np.mean(self.data), np.std(self.data), 0.0])

        result = minimize(self.dist.nll, initial_guess)
        return FitResult(params=result.x, success=result.success, message=result.message)
