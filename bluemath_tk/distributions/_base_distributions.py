from abc import abstractmethod
from typing import Dict, List

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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

        # Auxiliar for diagnostics plots
        self.n = self.data.shape[0]
        self.ecdf = np.arange(1, self.n + 1) / (self.n + 1)

    def summary(self):
        """
        Print a summary of the fitting results
        """
        print(f"Fitting results for {self.dist().name}:")
        print("--------------------------------------")
        print("Parameters:")
        for i, param in enumerate(self.params):
            print(f"   - {self.dist().param_names[i]}: {param:.4f}")
        # print("\n")
        print(f"Negative Log-Likelihood value: {self.nll:.4f}")
        print(f"{self.message}")

    def plot(self, ax=None, plot_type="all"):
        """
        Plots of fitting results: PP-plot, QQ-plot, histogram with fitted distribution, and return period plot.
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes will be created.
        plot_type : str, optional
            Type of plot to create. Options are "hist" for histogram, "pp" for P-P plot,
            "qq" for Q-Q plot, "return_period" for return period plot, or "all" for all plots.
            Default is "all".

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object containing the plots. If `ax` is provided, returns None.
        
        Raises
        -------
        ValueError
            If `plot_type` is not one of the valid options ("hist", "pp", "qq", "return_period", "all").
        """
        if plot_type == "all":
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            self.hist(ax=axs[0, 0])
            self.pp(ax=axs[0, 1])
            self.qq(ax=axs[1, 0])
            self.return_period(ax=axs[1, 1])
            plt.tight_layout()
            return fig
        elif plot_type == "hist":
            return self.hist()
        elif plot_type == "pp":
            return self.pp()
        elif plot_type == "qq":
            return self.qq()
        elif plot_type == "return_period":
            return self.return_period()
        else:
            raise ValueError("Invalid plot type. Use 'hist', 'pp', 'qq', 'return_period', or 'all'.")

    def pp(self, ax=None):
        """
        Probability plot of the fitted distribution.
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes will be created.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = None

        probabilities = self.dist.cdf(np.sort(self.data), *self.params)
        ax.plot([0, 1], [0, 1], color="tab:red", linestyle="--")
        ax.plot(probabilities, self.ecdf, color="tab:blue", marker="o", linestyle="", alpha=0.7)
        ax.set_xlabel("Fitted Probability")
        ax.set_ylabel("Empirical Probability")
        ax.set_title(f"PP Plot of {self.dist().name}")
        ax.grid()

        return fig

    def qq(self, ax=None):
        """
        Quantile-Quantile plot of the fitted distribution.
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes will be created.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = None

        quantiles = self.dist.qf(self.ecdf, *self.params)
        ax.plot([np.min(self.data), np.max(self.data)], [np.min(self.data), np.max(self.data)], color="tab:red", linestyle="--")
        ax.plot(quantiles, np.sort(self.data), color="tab:blue", marker="o", linestyle="", alpha=0.7)
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.set_title(f"QQ Plot of {self.dist().name}")
        ax.grid()

        return fig

    def hist(self, ax=None):
        """
        Histogram of the data with the fitted distribution overlayed.
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes will be created.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = None

        ax.hist(self.data, bins=30, density=True, alpha=0.7, color='tab:blue', label='Data Histogram')
        x = np.linspace(np.min(self.data), np.max(self.data), 1000)
        ax.plot(x, self.dist.pdf(x, *self.params), color='tab:red', label='Fitted PDF')
        ax.set_xlabel("Data Values")
        ax.set_ylabel("Density")
        ax.set_title(f"Histogram and Fitted PDF of {self.dist().name}")
        ax.legend()
        ax.grid()

        return fig

    def return_period(self, ax=None):
        """
        Return period plot of the fitted distribution.
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes will be created.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = None


        sorted_data = np.sort(self.data)
        exceedance_prob = 1 - self.ecdf
        return_period = 1 / exceedance_prob

        ax.plot(return_period, self.dist.qf(self.ecdf, *self.params), color='tab:red', label='Fitted Distribution')
        ax.plot(return_period, sorted_data, marker="o", linestyle="", color="tab:blue", alpha=0.7, label='Empirical Data')
        ax.set_xscale("log")
        ax.set_xlabel("Return Period")
        ax.set_ylabel("Data Values")
        ax.set_title(f"Return Period Plot of {self.dist().name}")
        ax.legend()
        ax.grid()

        return fig


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

    @property
    @abstractmethod
    def param_names(self) -> List[str]:
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
    def stats() -> Dict[str, float]:
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
