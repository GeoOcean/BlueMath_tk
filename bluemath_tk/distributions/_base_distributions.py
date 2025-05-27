from abc import abstractmethod

import numpy as np
from scipy.optimize import minimize

from ..core.models import BlueMathModel


class FitResult(BlueMathModel):
    """
    Class used for the results of fitting a distribution

    Attributes  
    ----------
    dist : BaseDistribution
        The distribution that was fitted.
    data : np.ndarray
        The data used for fitting the distribution.
    params : np.ndarray
        Fitted parameters of the distribution.
    success : bool
        Indicates whether the fitting was successful.
    message : str
        Message from the optimization result.
    nll : float     
        Negative log-likelihood of the fitted distribution.
    res : OptimizeResult    
        The result of the optimization process, containing additional information.
 
    Methods
    ------- 
    summary() -> dict
        Returns a summary of the fitting results, including parameters, negative log-likelihood,
        success status, message, and the optimization result.
    plot(ax=None, plot_type="hist")
        Plots of fitting results (NOT IMPLEMENTED).     
    
    Notes   
    -------
    - This class is used to encapsulate the results of fitting a distribution to data.
    - It provides a method to summarize the fitting results and a placeholder for plotting the results.
    """

    def __init__(self, dist, data, res):
        super().__init__()
        self.dist = dist
        self.data = data

        self.params = res.x
        self.success = res.success
        self.message = res.message
        self.nll = res.fun
        self.res = res

    def summary(self):
        """
        Returns a summary of the fitting resultsº

        Returns
        ------- 
        dict
            A dictionary containing the fitting results, including parameters,
            negative log-likelihood, success status, message, and the optimization result.
        """
        return {
            "parameters": self.params,
            "nll": self.nll,
            "success": self.success,
            "message": self.message,
            "result": self.res
        }

    def plot(self, ax=None, plot_type="hist"):
        """
        Plots of fitting results
        """
        pass


def fit_dist(dist, data: np.ndarray, **kwargs) -> FitResult:
    """
    Fit a distribution to data using Maximum Likelihood Estimation (MLE).

    Parameters
    ----------
    dist : BaseDistribution
        Distribution to fit.
    data : np.ndarray
        Data to use for fitting the distribution.
    **kwargs : dict, optional
        Additional options for fitting:
        - 'x0': Initial guess for distribution parameters (default: [mean, std, 0.0]).
        - 'method': Optimization method (default: 'Nelder-Mead').
        - 'bounds': Bounds for optimization parameters (default: [(None, None), (0, None), ...]).
        - 'options': Options for the optimizer (default: {'disp': False}).

    Returns
    -------
    FitResult
        The fitting results, including parameters, success status, and negative log-likelihood.
    """
    nparams = dist().nparams

    # Default optimization settings
    x0 = kwargs.get(
        "x0", np.asarray([np.mean(data), np.std(data)] + [0.0] * (nparams - 2))
    )
    method = kwargs.get("method", "Nelder-Mead").lower()
    bounds = kwargs.get(
        "bounds", [(None, None), (0, None)] + [(None, None)] * (nparams - 2)
    )
    options = kwargs.get("options", {"disp": False})

    # Objective function: Negative Log-Likelihood
    def obj(params):
        return dist.nll(data, *params)

    # Perform optimization
    result = minimize(fun=obj, x0=x0, method=method, bounds=bounds, options=options)

    # Return the fitting result as a FitResult object
    return FitResult(dist, data, result)


class BaseDistribution(BlueMathModel):
    """
    Base class for all extreme distributions.
    """

    @abstractmethod
    def __init__(self) -> None:
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
    def pdf(x: np.ndarray) -> np.ndarray:
        """
        Probability density function
        """
        pass

    @staticmethod
    @abstractmethod
    def cdf(x: np.ndarray) -> np.ndarray:
        """
        Cumulative distribution function
        """
        pass

    @staticmethod
    @abstractmethod
    def sf(x: np.ndarray) -> np.ndarray:
        """
        Survival function (1 - cdf)
        """
        pass

    @staticmethod
    @abstractmethod
    def qf(p: np.ndarray) -> np.ndarray:
        """
        Quantile function
        """
        pass

    @staticmethod
    @abstractmethod
    def nll(x: np.ndarray) -> float:
        """
        Negative Log-Likelihood function
        """
        pass

    @staticmethod
    @abstractmethod
    def random(data: np.ndarray, size: int) -> np.ndarray:
        """
        Generate random values
        """
        pass

    @staticmethod
    @abstractmethod
    def mean() -> float:
        """
        Mean
        """
        pass

    @staticmethod
    @abstractmethod
    def median() -> float:
        """
        Median
        """
        pass

    @staticmethod
    @abstractmethod
    def variance() -> float:
        """
        Variance
        """
        pass

    @staticmethod
    @abstractmethod
    def std() -> float:
        """
        Standard deviation
        """
        pass

    @staticmethod
    @abstractmethod
    def stats() -> dict:
        """
        Return summary statistics including mean, std, variance, etc.
        """
        pass

    @abstractmethod
    def fit(dist, data: np.ndarray, **kwargs) -> FitResult:
        """
        Fit distribution
        """
        pass
