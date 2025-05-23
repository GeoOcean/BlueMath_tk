from abc import abstractmethod
from typing import Tuple

import numpy as np
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

    @property
    @abstractmethod
    def nparams(self) -> int:
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
        fit_result = fit_dist(self, data, *args, **kwargs)
        return fit_result




class FitResult(BlueMathModel):
    """
    Class used for the results of fitting a distribution
    """
    def __init__(self, dist, data, res):
        self.dist = dist
        self.data = data

        self.params = res.x
        self.success = res.success
        self.message = res.message
        self.nll = res.obj

    def summary(self):
        return {
            'parameters': self.params,
            'nll': self.nll,
            'success': self.success,
            'message': self.message
        }
    
    def plot(self, ax=None, plot_type="hist"):
        """
        Plots of fitting results
        """
        pass




def fit_dist(dist, data, *args, **kwargs) -> FitResult:
    """
    Function used to fit a distributions


    Parameters
    ----------
    dist : BaseDistribution
        Distribution used to fit
    data : array_like 
        Data to use in estimating the distribution parameters.
        
    **kwds : floats, optional
        - 'x0': initial guess of distribution parameters

        - method : The method to use. The default is "MLE" (Maximum
            Likelihood Estimate)
            AT THE MOMENT ONLY MLE IS AVAILABLE            
            "MM" (Method of Moments) is also available.
        
        - method : Method used in optimization step.
            Default 'Nelder-Mead'

        - bounds : Tuple
            Optimization parameter bounds.
        
        - options : dict
            Optimization options

    Returns
    -------
    result: FitResult
        The fitting results, see FitResult class.
    """

    nparams = dist.nparams

    method = kwargs.get('method', 'Nelder-Mead').lower()
    bounds = kwargs.get('bounds', ((None, None),(0,None)) + ((None,None),)*(nparams - 2))
    options = kwargs.get('opt_options', {'disp': False})
    
    


    # Default initial guess
    x0 = kwargs.get("x0", np.asarray([np.mean(data), np.std(data)] + [0.0]*(nparams - 2)))

    def obj(param):
        return dist.nll(data, *param)                          # Negative Log-likelihood (function to minimize)
    
    result = minimize(
        fun=obj,
        x0=x0,
        method=method,
        bounds=bounds,
        options=options
    )

    return FitResult(dist, data, result)



