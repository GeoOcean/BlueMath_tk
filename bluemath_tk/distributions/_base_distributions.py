from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeResult, minimize

from ..core.models import BlueMathModel


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

    @staticmethod
    @abstractmethod
    def name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def nparams() -> int:
        pass

    @staticmethod
    @abstractmethod
    def param_names() -> List[str]:
        pass

    @staticmethod
    @abstractmethod
    def pdf(x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Probability density function

        Parameters
        ----------
        x : np.ndarray
            Values to compute the probability density value
        **kwargs :
            Distribution specific parameters as keyword arguments.
            Common parameters include:
            - loc: Location parameter
            - scale: Scale parameter
            - shape: Shape parameter (for some distributions)

        Returns
        ----------
        pdf : np.ndarray
            Probability density function values
        """
        pass

    @staticmethod
    @abstractmethod
    def cdf(x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Cumulative distribution function

        Parameters
        ----------
        x : np.ndarray
            Values to compute their probability
        **kwargs :
            Distribution specific parameters as keyword arguments.
            Common parameters include:
            - loc: Location parameter
            - scale: Scale parameter
            - shape: Shape parameter (for some distributions)

        Returns
        ----------
        p : np.ndarray
            Probability
        """
        pass

    @staticmethod
    @abstractmethod
    def sf(x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Survival function (1-Cumulative Distribution Function)

        Parameters
        ----------
        x : np.ndarray
            Values to compute their survival function value
        **kwargs :
            Distribution specific parameters as keyword arguments.
            Common parameters include:
            - loc: Location parameter
            - scale: Scale parameter
            - shape: Shape parameter (for some distributions)

        Returns
        ----------
        sp : np.ndarray
            Survival function value
        """
        pass

    @staticmethod
    @abstractmethod
    def qf(p: np.ndarray, **kwargs) -> np.ndarray:
        """
        Quantile function

        Parameters
        ----------
        p : np.ndarray
            Probabilities to compute their quantile
        **kwargs :
            Distribution specific parameters as keyword arguments.
            Common parameters include:
            - loc: Location parameter
            - scale: Scale parameter
            - shape: Shape parameter (for some distributions)

        Returns
        ----------
        q : np.ndarray
            Quantile value
        """
        pass

    @staticmethod
    @abstractmethod
    def nll(data: np.ndarray, **kwargs) -> float:
        """
        Negative Log-Likelihood function

        Parameters
        ----------
        data : np.ndarray
            Data to compute the Negative Log-Likelihood value
        **kwargs :
            Distribution specific parameters as keyword arguments.
            Common parameters include:
            - loc: Location parameter
            - scale: Scale parameter
            - shape: Shape parameter (for some distributions)

        Returns
        ----------
        nll : float
            Negative Log-Likelihood value
        """
        pass

    @staticmethod
    @abstractmethod
    def random(size: int, random_state: int, **kwargs) -> np.ndarray:
        """
        Generate random values

        Parameters
        ----------
        size : int
            Number of random values to generate
        random_state : int
            Seed for the random number generator
        **kwargs :
            Distribution specific parameters as keyword arguments.
            Common parameters include:
            - loc: Location parameter
            - scale: Scale parameter
            - shape: Shape parameter (for some distributions)

        Returns
        ----------
        x : np.ndarray
            Random values from GEV distribution
        """
        pass

    @staticmethod
    @abstractmethod
    def mean(**kwargs) -> float:
        """
        Mean

        Parameters
        ----------
        **kwargs :
            Distribution specific parameters as keyword arguments.
            Common parameters include:
            - loc: Location parameter
            - scale: Scale parameter
            - shape: Shape parameter (for some distributions)

        Returns
        ----------
        mean : np.ndarray
            Mean value of GEV with the given parameters
        """
        pass

    @staticmethod
    @abstractmethod
    def median(**kwargs) -> float:
        """
        Median

        Parameters
        ----------
        **kwargs :
            Distribution specific parameters as keyword arguments.
            Common parameters include:
            - loc: Location parameter
            - scale: Scale parameter
            - shape: Shape parameter (for some distributions)

        Returns
        ----------
        median : np.ndarray
            Median value of GEV with the given parameters
        """
        pass

    @staticmethod
    @abstractmethod
    def variance(**kwargs) -> float:
        """
        Variance

        Parameters
        ----------
        **kwargs :
            Distribution specific parameters as keyword arguments.
            Common parameters include:
            - loc: Location parameter
            - scale: Scale parameter
            - shape: Shape parameter (for some distributions)

        Returns
        ----------
        var : np.ndarray
            Variance of GEV with the given parameters
        """
        pass

    @staticmethod
    @abstractmethod
    def std(**kwargs) -> float:
        """
        Standard deviation

        Parameters
        ----------
        **kwargs :
            Distribution specific parameters as keyword arguments.
            Common parameters include:
            - loc: Location parameter
            - scale: Scale parameter
            - shape: Shape parameter (for some distributions)

        Returns
        ----------
        std : np.ndarray
            Standard Deviation of GEV with the given
            parameters
        """
        pass

    @staticmethod
    @abstractmethod
    def stats(**kwargs) -> Dict[str, float]:
        """
        Summary statistics

        Return summary statistics including mean, std, variance, etc.

        Parameters
        ----------
        **kwargs :
            Distribution specific parameters as keyword arguments.
            Common parameters include:
            - loc: Location parameter
            - scale: Scale parameter
            - shape: Shape parameter (for some distributions)

        Returns
        ----------
        stats : dict
            Summary statistics of BaseDistribution with the given
            parameters
        """
        pass

    @staticmethod
    @abstractmethod
    def fit(dist, data: np.ndarray, **kwargs) -> FitResult:
        """
        Fit any distribution

        Parameters
        ----------
        dist : BaseDistribution
            Distribution to fit.
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
        pass


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

    def __init__(self, dist: BaseDistribution, data: np.ndarray, res: OptimizeResult):
        """
        Initialize the FitResult object.

        Parameters
        ----------
        dist : BaseDistribution
            The distribution that was fitted.
        data : np.ndarray
            The data used for fitting the distribution.
        res : OptimizeResult
            The result of the optimization process.
        """
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
        print(f"Fitting results for {self.dist.name()}:")
        print("--------------------------------------")
        print("Parameters:")
        for i, param in enumerate(self.params):
            print(f"   - {self.dist.param_names()[i]}: {param:.4f}")
        # print("\n")
        print(f"Negative Log-Likelihood value: {self.nll:.4f}")
        print(f"{self.message}")

    def plot(self, ax: plt.axes = None, plot_type="all") -> Tuple[plt.figure, plt.axes]:
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
            self.pp(ax=axs[0, 0])
            self.qq(ax=axs[0, 1])
            self.hist(ax=axs[1, 0])
            self.return_period(ax=axs[1, 1])
            plt.tight_layout()
            return fig, axs
        elif plot_type == "hist":
            return self.hist()
        elif plot_type == "pp":
            return self.pp()
        elif plot_type == "qq":
            return self.qq()
        elif plot_type == "return_period":
            return self.return_period()
        else:
            raise ValueError(
                "Invalid plot type. Use 'hist', 'pp', 'qq', 'return_period', or 'all'."
            )

    def pp(self, ax: plt.axes = None) -> Tuple[plt.figure, plt.axes]:
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
        ax.plot(
            probabilities,
            self.ecdf,
            color="tab:blue",
            marker="o",
            linestyle="",
            alpha=0.7,
        )
        ax.set_xlabel("Fitted Probability")
        ax.set_ylabel("Empirical Probability")
        ax.set_title("PP Plot")
        ax.grid()

        return fig, ax

    def qq(self, ax: plt.axes = None) -> Tuple[plt.figure, plt.axes]:
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
        ax.plot(
            [np.min(self.data), np.max(self.data)],
            [np.min(self.data), np.max(self.data)],
            color="tab:red",
            linestyle="--",
        )
        ax.plot(
            quantiles,
            np.sort(self.data),
            color="tab:blue",
            marker="o",
            linestyle="",
            alpha=0.7,
        )
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.set_title("QQ Plot")
        ax.grid()

        return fig, ax

    def hist(self, ax: plt.axes = None) -> Tuple[plt.figure, plt.axes]:
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

        ax.hist(
            self.data,
            bins=30,
            density=True,
            alpha=0.7,
            color="tab:blue",
            label="Data Histogram",
        )
        x = np.linspace(np.min(self.data), np.max(self.data), 1000)
        ax.plot(x, self.dist.pdf(x, *self.params), color="tab:red", label="Fitted PDF")
        ax.set_xlabel("Data Values")
        ax.set_ylabel("Density")
        ax.set_title("Histogram and Fitted PDF")
        ax.legend()
        ax.grid()

        return fig, ax

    def return_period(self, ax: plt.axes = None) -> Tuple[plt.figure, plt.axes]:
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

        return_years = np.asarray([1.001, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2, 3, 4, 5, 7.5, 10, 15, 20, 25, 50, 100, 250, 500, 1000])
        ecdf_fitted = 1 - 1/return_years
        sorted_data = np.sort(self.data)
        exceedance_prob = 1 - self.ecdf
        return_period = 1 / exceedance_prob

        ax.plot(
            return_years,
            self.dist.qf(ecdf_fitted, *self.params),
            color="tab:red",
            label="Fitted Distribution",
        )
        ax.plot(
            return_period,
            sorted_data,
            marker="o",
            linestyle="",
            color="tab:blue",
            alpha=0.7,
            label="Empirical Data",
        )
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 25, 50, 100, 250, 1000, 10000])
        ax.set_xticklabels([1, 2, 5, 10, 25, 50, 100, 500, 1000, 10000])
        # ax.set_xlim(right=np.max(return_period) * 1.2)
        ax.set_xlabel("Return Period")
        ax.set_ylabel("Data Values")
        ax.set_title("Return Period Plot")
        ax.legend()
        ax.grid()

        return fig, ax


def fit_dist(dist: BaseDistribution, data: np.ndarray, **kwargs) -> FitResult:
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
        - 'f0', 'floc': Fix location parameter (first parameter).
        - 'f1', 'fscale': Fix scale parameter (second parameter).
        - 'f2', 'fshape': Fix shape parameter (third parameter).

    Returns
    -------
    FitResult
        The fitting results, including parameters, success status, and negative log-likelihood.
    """
    nparams = dist.nparams()
    param_names = dist.param_names()

    # Handle fixed parameters
    fixed_params = {}
    for i, name in enumerate(param_names):
        # Check both numerical (f0, f1, f2) and named (floc, fscale, fshape) fixed parameters
        num_key = f"f{i}"
        if num_key in kwargs:
            fixed_params[i] = kwargs[num_key]

    # Create list of free parameters (those not fixed)
    free_params = [i for i in range(nparams) if i not in fixed_params]
    # n_free = len(free_params)

    # Default optimization settings for free parameters only
    default_x0 = np.asarray([np.mean(data), np.std(data)] + [0.1] * (nparams - 2))
    x0 = kwargs.get("x0", default_x0)
    x0_free = np.array([x0[i] for i in free_params])

    method = kwargs.get("method", "Nelder-Mead").lower()
    default_bounds = [(None, None), (1e-6, None)] + [(None, None)] * (nparams - 2)
    bounds = kwargs.get("bounds", default_bounds)
    bounds_free = [bounds[i] for i in free_params]
    options = kwargs.get("options", {"disp": False})

    # Objective function that handles fixed parameters
    def obj(free_values):
        # Reconstruct full parameter vector with fixed values
        full_params = np.zeros(nparams)
        free_idx = 0
        for i in range(nparams):
            if i in fixed_params:
                full_params[i] = fixed_params[i]
            else:
                full_params[i] = free_values[free_idx]
                free_idx += 1
        return dist.nll(data, *full_params)

    # Perform optimization only on free parameters
    result = minimize(
        fun=obj, x0=x0_free, method=method, bounds=bounds_free, options=options
    )

    # Reconstruct full parameter vector for final result
    full_params = np.zeros(nparams)
    free_idx = 0
    for i in range(nparams):
        if i in fixed_params:
            full_params[i] = fixed_params[i]
        else:
            full_params[i] = result.x[free_idx]
            free_idx += 1

    # Create modified result with full parameters
    # TODO: Modify the OptimizeResult to include all the attributes from the original result
    # but with the full parameter vector
    # TODO: Add hess info or hess_inv info to compute confidence intervals in FitResult
    modified_result = OptimizeResult(
        x=full_params,
        success=result.success,
        fun=result.fun,
        message=result.message,
        hess_inv=result.hess_inv if hasattr(result, "hess_inv") else None,
    )

    # Return the fitting result as a FitResult object
    return FitResult(dist, data, modified_result)
