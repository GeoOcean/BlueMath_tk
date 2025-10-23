import os
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.integrate import quad
from scipy.optimize import minimize, root_scalar
from scipy.stats import genextreme, norm

from ..core.models import BlueMathModel
from ..core.plotting.colors import default_colors

# @njit(fastmath=True)
# def search(times: np.ndarray, values: np.ndarray, xs) -> np.ndarray:
#     """
#     Function to search the nearest value of certain time to use in self._parametro function

#     Parameters
#     ----------
#     times : np.ndarray
#         Times when covariates are known
#     values : np.ndarray
#         Values of the covariates at those times
#     """
#     # n = times.shape[0]
#     # yin = np.zeros_like(xs)
#     # pos = 0
#     # for j in range(xs.size):
#     #     found = 0
#     #     while found == 0 and pos < n:
#     #         if xs[j] < times[pos]:
#     #             yin[j] = values[pos]
#     #             found = 1
#     #         else:
#     #             pos += 1

#     # return yin

#     idx = np.searchsorted(times, xs, side='right')
#     mask = idx < len(times)

#     return values[idx[mask]]


class NonStatGEV(BlueMathModel):
    """
    Non-Stationary Generalized Extreme Value Distribution

    Class to implement the Non-Stationary GEV including trends and/or covariates
    in the location, scale and shape parameters. This methodology selects the
    covariates and trends based on which of them minimize the Akaike Information Criteria (AIC).

    This class is based in the work of R. Mínguez et al. 2010. "Pseudooptimal parameter selection
    of non-stationary generalized extreme value models for environmental variables". Environ. Model. Softw. 25, 1592-1607.

    Parameters
    ----------
    xt : np.ndarray
        Data to fit Non Stationary GEV.
    t : np.ndarray, default=None.
        Time associated to the data.
    covariates: np.ndarray | pd.DataFrame, default=None.
        Covariates to include for location, scale and shape parameters.
    kt : np.ndarray, default=None.
        Frequency of block maxima, if None, it is assumed to be 1.
    trends: bool, defaul=False.
        Whether trends should be included, if so, t must be passed.
    quanval : float, default=0.95.
        Confidence interval value
    var_name : str, default="x"
        Name of the variable to be used in the model.
        Used for plotting purposes.

    Methods
    ----------
    fit:
        Fit the Non-Stationary GEV with desired Trends and Covariates.
    auto_adjust:
        Automatically selects the best covariates and trends based on AIC.


    Examples
    --------
    >>> from bluemath_tk.distributions.nonstat_gev import NonStatGEV
    >>> nonstat_gev = NonStatGEV(x, t, covariates, trends=True)
    >>> fit_result = nonstat_gev.auto_adjust()
    >>> fit_result = nonstat_gev.fit(nmu=2,npsi=2,ngamma=2,ntrend_loc=1,list_loc="all",ntrend_sc=1,list_sc="all",ntrend_sh=1,list_sh="all")
    """

    def __init__(
        self,
        xt: np.ndarray,
        t: np.ndarray,
        covariates: Optional[np.ndarray | pd.DataFrame] = None,
        harms: bool = True,
        trends: bool = False,
        kt: Optional[np.ndarray] = None,
        quanval: float = 0.95,
        var_name: str = "x",
    ):
        """
        Initiliaze the Non-Stationary GEV.
        """
        super().__init__()

        debug = 1
        self.set_logger_name(
            name=self.__class__.__name__, level="DEBUG" if debug else "INFO"
        )
        self.logger.debug("Initializing NonStatGEV")

        # Initialize arguments
        self.xt = xt
        self.t = t
        if covariates is None:
            self.include_covariates = False
            self.covariates = pd.DataFrame({"A": []})
        else:
            self.include_covariates = True
            self.covariates = covariates
        self.harms = harms
        self.trends = trends
        if kt is None:
            self.kt = np.ones_like(xt)
        else:
            self.kt = kt
        self.quanval = quanval
        self.var_name = var_name

        # Initialize parameters associated to the GEV
        # Location
        self.beta0 = np.empty(0)  # Location intercept
        self.beta = np.empty(0)  # Location harmonic
        self.betaT = np.empty(0)  # Location trend
        self.beta_cov = np.empty(0)  # Location covariates
        # Scale
        self.alpha0 = np.empty(0)  # Scale intercept
        self.alpha = np.empty(0)  # Scale harmonic
        self.alphaT = np.empty(0)  # Scale trend
        self.alpha_cov = np.empty(0)  # Scale covariates
        # Shape
        self.gamma0 = np.empty(0)  # Shape intercept
        self.gamma = np.empty(0)  # Shape harmonic
        self.gammaT = np.empty(0)  # Shape trend
        self.gamma_cov = np.empty(0)  # Shape covariates

        # Initilize the number of parameters used
        # Location
        self.nmu = 0  # Number of parameters of harmonic part of location
        self.nind_loc = 0  # Number of parameters of covariates part of location
        self.ntrend_loc = 0  # Number of parameters of trend part of location
        # Scale
        self.npsi = 0  # Number of parameters of harmonic part of scale
        self.nind_sc = 0  # Number of parameters of covariates part of scale
        self.ntrend_sc = 0  # Number of parameters of trend part of scale
        # Shape
        self.ngamma0 = 1  # 1 if shape parameter is included, defaul Weibull or Frechet
        self.ngamma = 0  # Number of parameters of harmonic part of shape
        self.nind_sh = 0  # Number of parameters of covariates part of shape
        self.ntrend_sh = 0  # Number of parameters of trend part of shape

        # Color palette
        self.colors = default_colors

    def auto_adjust(self, max_iter: int = 1000, plot: bool = False) -> dict:
        """
        This method automatically select and calculate the parameters which minimize the AIC related to
        Non-Stationary GEV distribution using the Maximum Likelihood method within an iterative scheme,
        including one parameter at a time based on a perturbation criteria.
        The process is repeated until no further improvement in the objective function is achieved.

        Parameters
        ----------
        max_iter : int, default=1000
            Number of iteration in optimization process.
        plot : bool, default=False
            If plot the adjusted distribution

        Return
        ----------
        fit_result : dict
            Dictionary with the optimal parameters and values related to the Non-Stationary GEV distribution.
            The keys of the dictionary are:
            - x: Optimal solution
            - beta0, beta, betaT, beta_cov: Location parameters (intercept, harmonic, trend, covariates)
            - alpha0, alpha, alphaT, alpha_cov: Scale parameters (intercept, harmonic, trend, covariates)
            - gamma0, gamma, gammaT, gamma_cov: Shape parameters (intercept, harmonic, trend, covariates)
            - negloglikelihood: Negative log-likelihood value at the optimal solution
            - loglikelihood: Log-likelihood value at the optimal solution
            - grad: Gradient of the log-likelihood function at the optimal solution
            - hessian: Hessian matrix of the log-likelihood function at the optimal solution
            - AIC: Akaike Information Criterion value at the optimal solution
            - invI0: Fisher information matrix at the optimal solution
            - std_param: Standard deviation of parameters at the optimal solution
        """

        self.max_iter = max_iter  # Set maximum number of iterations

        self.AIC_iter = np.zeros(
            self.max_iter
        )  # Initialize the values of AIC in each iteration
        self.loglike_iter = np.zeros(
            self.max_iter
        )  # Initialize the values of Loglikelihood in each iteration

        ### Step 1: Only stationary parameters
        nmu = 0  # Number of parameters of harmonic part of location
        npsi = 0  # Number of parameters of harmonic part of scale
        ngamma = 0  # Number of parameters of harmonic part of shape
        nind_loc = 0  # Number of parameters of covariates part of location
        ntrend_loc = 0  # Number of parameters of trend part of location
        nind_sc = 0  # Number of parameters of covariates part of scale
        ntrend_sc = 0  # Number of parameters of trend part of scale
        nind_sh = 0  # Number of parameters of covariates part of shape
        ntrend_sh = 0  # Number of parameters of trend part of shape

        ######### HARMONIC Iterative process #########
        if self.harms:
            print("Starting Harmonic iterative process")
            for iter in range(self.max_iter):
                ### Step 2: Fit for the selected parameters (initial step is stationary)
                fit_result = self._fit(nmu, npsi, ngamma)

                # Check if the model is Gumbel
                self.ngamma0 = 1
                if fit_result["gamma0"] is None:
                    self.ngamma0 = 0
                elif np.abs(fit_result["gamma0"]) <= 1e-8:
                    self.ngamma0 = 0

                # Compute AIC and Loglikelihood
                self.loglike_iter[iter] = -fit_result["negloglikelihood"]
                n_params = (
                    2
                    + self.ngamma0
                    + 2 * nmu
                    + 2 * npsi
                    + 2 * ngamma
                    + nind_loc
                    + ntrend_loc
                    + nind_sc
                    + ntrend_sc
                    + nind_sh
                    + ntrend_sh
                )
                self.AIC_iter[iter] = self._AIC(
                    -fit_result["negloglikelihood"], n_params
                )

                ### Step 4: Sensitivity of optimal loglikelihood respect to possible additional harmonics
                # for the location, scale and shape parameters.
                # Note that the new parameter values are set to zero since derivatives do not depend on them
                fit_result_aux = fit_result.copy()
                # Location
                if fit_result["beta"] is not None:
                    fit_result_aux["beta"] = np.concatenate(
                        (fit_result["beta"], [0, 0])
                    )
                else:
                    fit_result_aux["beta"] = np.array([0, 0])
                # Scale
                if fit_result["alpha"] is not None:
                    fit_result_aux["alpha"] = np.concatenate(
                        (fit_result["alpha"], [0, 0])
                    )
                else:
                    fit_result_aux["alpha"] = np.array([0, 0])
                # Shape
                if fit_result["gamma"] is not None:
                    fit_result_aux["gamma"] = np.concatenate(
                        (fit_result["gamma"], [0, 0])
                    )
                else:
                    fit_result_aux["gamma"] = np.array([0, 0])

                auxf, auxJx, auxHxx = self._loglikelihood(
                    beta0=fit_result_aux["beta0"],
                    beta=fit_result_aux["beta"],
                    alpha0=fit_result_aux["alpha0"],
                    alpha=fit_result_aux["alpha"],
                    gamma0=fit_result_aux["gamma0"],
                    gamma=fit_result_aux["gamma"],
                )

                # Inverse of the Information Matrix (auxHxx)
                auxI0 = np.linalg.inv(-auxHxx)

                # Updating the best model
                if iter > 0:
                    # TODO: Implement another criterias (Proflike)
                    if self.AIC_iter[iter] < self.AIC_iter[iter - 1]:
                        modelant = np.array([nmu, npsi, ngamma])
                else:
                    modelant = np.array([nmu, npsi, ngamma])

                ### Step 5: Compute maximum perturbation
                pos = 1
                # Perturbation for location
                max_val = np.abs(
                    auxJx[2 * nmu : 2 * nmu + 2].T
                    @ auxI0[2 * nmu : 2 * nmu + 2, 2 * nmu : 2 * nmu + 2]
                    @ auxJx[2 * nmu : 2 * nmu + 2]
                )
                # Perturbation for scale
                auxmax = abs(
                    auxJx[
                        2 + 2 * nmu + ntrend_loc + nind_loc + 2 * npsi + 2 : 2
                        + 2 * nmu
                        + ntrend_loc
                        + nind_loc
                        + 2 * npsi
                        + 2
                        + 2
                    ].T
                    @ auxI0[
                        2 + 2 * nmu + ntrend_loc + nind_loc + 2 * npsi + 2 : 2
                        + 2 * nmu
                        + ntrend_loc
                        + nind_loc
                        + 2 * npsi
                        + 2
                        + 2,
                        2 + 2 * nmu + ntrend_loc + nind_loc + 2 * npsi + 2 : 2
                        + 2 * nmu
                        + ntrend_loc
                        + nind_loc
                        + 2 * npsi
                        + 2
                        + 2,
                    ]
                    @ auxJx[
                        2 + 2 * nmu + ntrend_loc + nind_loc + 2 * npsi + 2 : 2
                        + 2 * nmu
                        + ntrend_loc
                        + nind_loc
                        + 2 * npsi
                        + 2
                        + 2
                    ]
                )
                if auxmax > max_val:
                    max_val = auxmax
                    pos = 2

                # Perturbation for shape
                auxmax = abs(
                    auxJx[
                        2
                        + self.ngamma0
                        + 2 * nmu
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + 2 * npsi
                        + 2 * ngamma
                        + 4 : 2
                        + self.ngamma0
                        + 2 * nmu
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + 2 * npsi
                        + 2 * ngamma
                        + 4
                        + 2
                    ].T
                    @ auxI0[
                        2
                        + self.ngamma0
                        + 2 * nmu
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + 2 * npsi
                        + 2 * ngamma
                        + 4 : 2
                        + self.ngamma0
                        + 2 * nmu
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + 2 * npsi
                        + 2 * ngamma
                        + 4
                        + 2,
                        2
                        + self.ngamma0
                        + 2 * nmu
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + 2 * npsi
                        + 2 * ngamma
                        + 4 : 2
                        + self.ngamma0
                        + 2 * nmu
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + 2 * npsi
                        + 2 * ngamma
                        + 4
                        + 2,
                    ]
                    @ auxJx[
                        2
                        + self.ngamma0
                        + 2 * nmu
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + 2 * npsi
                        + 2 * ngamma
                        + 4 : 2
                        + self.ngamma0
                        + 2 * nmu
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + 2 * npsi
                        + 2 * ngamma
                        + 4
                        + 2
                    ]
                )
                if auxmax > max_val:
                    max_val = auxmax
                    pos = 3

                # If maximum perturbation corresponds to location, include a new harmonic
                if pos == 1:
                    nmu += 1
                # If maximum perturbation corresponds to scale, include a new harmonic
                if pos == 2:
                    npsi += 1
                # If maximum perturbation corresponds to shape, include a new harmonic
                if pos == 3:
                    ngamma += 1

                print("Iteration: ", iter, "- AIC: ", self.AIC_iter[iter])
                if iter > 0:
                    if self.AIC_iter[iter] > self.AIC_iter[iter - 1]:
                        model = modelant
                        self.AICini = self.AIC_iter[iter - 1]
                        # loglikeobjITini = self.loglike_iter[iter - 1]
                        break
                    else:
                        model = np.array([nmu, npsi, ngamma])

            self.niter_harm = iter
            self.nit = iter
            print("End of the Harmonic iterative process")

            ######### End of the Harmonic iterative process
            # Obtaining the MLE for the best model
            nmu = model[0]
            npsi = model[1]
            ngamma = model[2]

            # CHECKING THE SHAPE PARAMETER
            self.ngamma0 = (
                0  # Force the elimination of the constant shape parameter (gamma0)
            )
            fit_result = self._fit(nmu, npsi, ngamma)

            n_params = (
                2
                + self.ngamma0
                + 2 * nmu
                + 2 * npsi
                + 2 * ngamma
                + nind_loc
                + ntrend_loc
                + nind_sc
                + ntrend_sc
                + nind_sh
                + ntrend_sh
            )
            self.AIC_iter[self.niter_harm + 1] = self._AIC(
                -fit_result["negloglikelihood"], n_params
            )
            self.loglike_iter[self.niter_harm + 1] = -fit_result["negloglikelihood"]

            if self.AICini < self.AIC_iter[self.niter_harm + 1]:
                # The constant shape parameter (gamma0) is significative
                self.ngamma0 = 1
                fit_result = self._fit(nmu, npsi, ngamma)

            print("Harmonic AIC:", self.AICini, "\n")

            self._update_params(**fit_result)
            self.nmu = nmu
            self.npsi = npsi
            self.ngamma = ngamma
        else:
            # Set 0 the number of harmonics parameters
            self.nmu = 0
            self.npsi = 0
            self.ngamma = 0

            fit_result = self._fit(self.nmu, self.npsi, self.ngamma)

            iter = 0
            self.niter_harm = iter
            # Check if the model is Gumbel
            self.ngamma0 = 1
            if fit_result["gamma0"] is None:
                self.ngamma0 = 0
            elif np.abs(fit_result["gamma0"]) <= 1e-8:
                self.ngamma0 = 0

            # Compute AIC and Loglikelihood
            self.loglike_iter[iter] = -fit_result["negloglikelihood"]
            n_params = (
                2
                + self.ngamma0
                + 2 * self.nmu
                + 2 * self.npsi
                + 2 * self.ngamma
                + nind_loc
                + ntrend_loc
                + nind_sc
                + ntrend_sc
                + nind_sh
                + ntrend_sh
            )
            self.AIC_iter[iter] = self._AIC(-fit_result["negloglikelihood"], n_params)

        ######### COVARIATES Iterative process #########
        nrows, nind_cov = self.covariates.shape  # Number of covariates

        # Auxiliar variables related to location parameter
        beta_cov = np.asarray([])
        list_loc = []  # List of covariates for location
        # auxcov_loc = self.covariates.iloc[:, list_loc].values
        # Auxiliar variables related to scale parameter
        alpha_cov = np.asarray([])
        list_sc = []
        # auxcov_sc = self.covariates.iloc[:, list_sc].values
        # Auxiliar variables related to shape parameter
        gamma_cov = np.asarray([])
        list_sh = []
        # auxcov_sh = self.covariates.iloc[:, list_sh].values

        auxlist_cov = list(np.arange(nind_cov))

        if self.include_covariates:
            print("Starting Covariates iterative process")
            for iter in range(self.niter_harm + 1, self.max_iter):
                self.ngamma0 = 1

                auxbeta_cov = np.zeros(nind_cov)
                if len(list_loc) > 0:
                    auxbeta_cov[list_loc] = beta_cov
                auxalpha_cov = np.zeros(nind_cov)
                if len(list_sc) > 0:
                    auxalpha_cov[list_sc] = alpha_cov
                auxgamma_cov = np.zeros(nind_cov)
                if len(list_sh) > 0:
                    auxgamma_cov[list_sh] = gamma_cov

                ### Step 9: Calculate the sensitivities of the optimal log-likelihood objective function with respect to possible
                # additional covariates for the location and  scale parameters
                auxf, auxJx, auxHxx = self._loglikelihood(
                    beta0=fit_result["beta0"],
                    beta=fit_result["beta"],
                    beta_cov=auxbeta_cov,
                    alpha0=fit_result["alpha0"],
                    alpha=fit_result["alpha"],
                    alpha_cov=auxalpha_cov,
                    gamma0=fit_result["gamma0"],
                    gamma=fit_result["gamma"],
                    gamma_cov=auxgamma_cov,
                    list_loc=auxlist_cov,
                    list_sc=auxlist_cov,
                    list_sh=auxlist_cov,
                )

                # Step 10: Include in the parameter vector the corresponding covariate
                auxI0 = np.linalg.inv(-auxHxx)
                # Perturbation of location
                values1 = np.abs(
                    auxJx[
                        1 + 2 * nmu + ntrend_loc : 1 + 2 * nmu + ntrend_loc + nind_cov
                    ]
                    ** 2
                    / np.diag(
                        auxI0[
                            1 + 2 * nmu + ntrend_loc : 1
                            + 2 * nmu
                            + ntrend_loc
                            + nind_cov,
                            1 + 2 * nmu + ntrend_loc : 1
                            + 2 * nmu
                            + ntrend_loc
                            + nind_cov,
                        ]
                    )
                )
                maximo_loc, pos_loc = np.max(values1), np.argmax(values1)
                # Perturbation of scale
                values2 = np.abs(
                    auxJx[
                        2 + 2 * nmu + ntrend_loc + nind_cov + 2 * npsi + ntrend_sc : 2
                        + 2 * nmu
                        + ntrend_loc
                        + nind_cov
                        + 2 * npsi
                        + ntrend_sc
                        + nind_cov
                    ]
                    ** 2
                    / np.diag(
                        auxI0[
                            2
                            + 2 * nmu
                            + ntrend_loc
                            + nind_cov
                            + 2 * npsi
                            + ntrend_sc : 2
                            + 2 * nmu
                            + ntrend_loc
                            + nind_cov
                            + 2 * npsi
                            + ntrend_sc
                            + nind_cov,
                            2
                            + 2 * nmu
                            + ntrend_loc
                            + nind_cov
                            + 2 * npsi
                            + ntrend_sc : 2
                            + 2 * nmu
                            + ntrend_loc
                            + nind_cov
                            + 2 * npsi
                            + ntrend_sc
                            + nind_cov,
                        ]
                    )
                )
                maximo_sc, pos_sc = np.max(values2), np.argmax(values2)
                # Perturbation of shape
                values3 = np.abs(
                    auxJx[
                        2
                        + self.ngamma0
                        + 2 * nmu
                        + ntrend_loc
                        + nind_cov
                        + 2 * npsi
                        + ntrend_sc
                        + nind_cov
                        + 2 * ngamma
                        + ntrend_sh : 2
                        + self.ngamma0
                        + 2 * nmu
                        + ntrend_loc
                        + nind_cov
                        + 2 * npsi
                        + ntrend_sc
                        + nind_cov
                        + 2 * ngamma
                        + ntrend_sh
                        + nind_cov
                    ]
                    ** 2
                    / np.diag(
                        auxI0[
                            2
                            + self.ngamma0
                            + 2 * nmu
                            + ntrend_loc
                            + nind_cov
                            + 2 * npsi
                            + ntrend_sc
                            + nind_cov
                            + 2 * ngamma : 2
                            + self.ngamma0
                            + 2 * nmu
                            + ntrend_loc
                            + nind_cov
                            + 2 * npsi
                            + ntrend_sc
                            + nind_cov
                            + 2 * ngamma
                            + ntrend_sh
                            + nind_cov,
                            2
                            + self.ngamma0
                            + 2 * nmu
                            + ntrend_loc
                            + nind_cov
                            + 2 * npsi
                            + ntrend_sc
                            + nind_cov
                            + 2 * ngamma : 2
                            + self.ngamma0
                            + 2 * nmu
                            + ntrend_loc
                            + nind_cov
                            + 2 * npsi
                            + ntrend_sc
                            + nind_cov
                            + 2 * ngamma
                            + ntrend_sh
                            + nind_cov,
                        ]
                    )
                )
                maximo_sh, pos_sh = np.max(values3), np.argmax(values3)

                # Select the maximum perturbation
                posmaxparam = np.argmax([maximo_loc, maximo_sc, maximo_sh])

                # Initialize auxiliar covariates variables
                if beta_cov.size > 0:
                    beta_cov_init = beta_cov.copy()
                else:
                    beta_cov_init = np.asarray([])
                if alpha_cov.size > 0:
                    alpha_cov_init = alpha_cov.copy()
                else:
                    alpha_cov_init = np.asarray([])
                if gamma_cov.size > 0:
                    gamma_cov_init = gamma_cov.copy()
                else:
                    gamma_cov_init = np.asarray([])

                if posmaxparam == 0:
                    # Add covariate to location
                    nind_loc += 1
                    list_loc.append(int(pos_loc))
                    beta_cov_init = np.append(
                        beta_cov_init, [0]
                    )  # Initialize the new covariate as zero
                elif posmaxparam == 1:
                    # Add covariate to scale
                    nind_sc += 1
                    list_sc.append(int(pos_sc))
                    alpha_cov_init = np.append(
                        alpha_cov_init, [0]
                    )  # Initialize the new covariate as zero
                elif posmaxparam == 2:
                    # Add covariate to shape
                    nind_sh += 1
                    list_sh.append(int(pos_sh))
                    gamma_cov_init = np.append(
                        gamma_cov_init, [0]
                    )  # Initialize the new covariate as zero

                # Update auxiliar covariates
                # auxcov_loc = self.covariates.iloc[:, list_loc].values
                # auxcov_sc = self.covariates.iloc[:, list_sc].values
                # auxcov_sh = self.covariates.iloc[:, list_sh].values

                ### Step 11: Obtain the maximum-likelihood estimators for the selected parameters and
                # calculate the Akaike Information criterion objective function AIC
                concatvalues = [
                    fit_result["x"][0 : 1 + 2 * nmu],
                    beta_cov_init,
                    fit_result["x"][1 + 2 * nmu : 2 + 2 * nmu + 2 * npsi],
                    alpha_cov_init,
                    0.1 * np.ones(self.ngamma0),
                    0.01 * np.zeros(2 * ngamma),
                    gamma_cov_init,
                ]
                xini = np.concatenate(
                    [np.asarray(v) for v in concatvalues if v is not None]
                )
                fit_result = self._fit(
                    self.nmu,
                    self.npsi,
                    self.ngamma,
                    list_loc,
                    ntrend_loc,
                    list_sc,
                    ntrend_sc,
                    list_sh,
                    ntrend_sh,
                    xini,
                )

                # Check if model is Gumbel
                # self.ngamma0 =
                n_params = (
                    2
                    + self.ngamma0
                    + 2 * nmu
                    + 2 * npsi
                    + 2 * ngamma
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc
                    + ntrend_sh
                    + nind_sh
                )
                self.AIC_iter[iter] = self._AIC(
                    -fit_result["negloglikelihood"], n_params
                )
                self.loglike_iter[iter] = -fit_result["negloglikelihood"]

                print("Iteration: ", iter, "- AIC: ", self.AIC_iter[iter])

                if self.AIC_iter[iter] <= self.AIC_iter[iter - 1]:
                    # Update the parameters
                    final_fit_result = fit_result
                    self.AICini = self.AIC_iter[iter]
                    self._update_params(**fit_result)
                    beta_cov = fit_result.get("beta_cov")
                    alpha_cov = fit_result.get("alpha_cov")
                    gamma_cov = fit_result.get("gamma_cov")
                    self.list_loc = list_loc
                    self.nind_loc = nind_loc
                    self.list_sc = list_sc
                    self.nind_sc = nind_sc
                    self.list_sh = list_sh
                    self.nind_sh = nind_sh
                else:
                    if posmaxparam == 0:
                        list_loc = list_loc[:-1]
                        beta_cov = beta_cov[:-1]
                        nind_loc -= 1
                    elif posmaxparam == 1:
                        list_sc = list_sc[:-1]
                        alpha_cov = alpha_cov[:-1]
                        nind_sc -= 1
                    else:
                        list_sh = list_sh[:-1]
                        gamma_cov = gamma_cov[:-1]
                        nind_sh -= 1

                    fit_result = final_fit_result
                    self.niter_cov = iter - self.niter_harm
                    self.nit = iter

                    self.list_loc = list_loc
                    self.nind_loc = nind_loc
                    self.list_sc = list_sc
                    self.nind_sc = nind_sc
                    self.list_sh = list_sh
                    self.nind_sh = nind_sh
                    break

            print("End of Covariates iterative process")
            print("Covariates AIC:", self.AICini, "\n")
        else:
            print("No covariates provided, skipping Covariates iterative process", "\n")
            self.list_loc = list_loc
            self.nind_loc = nind_loc
            self.list_sc = list_sc
            self.nind_sc = nind_sc
            self.list_sh = list_sh
            self.nind_sh = nind_sh

        ######### TRENDS Iterative process #########
        if self.trends:
            print("Starting Trends process")
            # Location trends
            ntrend_loc = 1

            # Initial parameter
            concatvalues = [
                fit_result["x"][0 : 1 + 2 * nmu],
                np.zeros(ntrend_loc),
                fit_result["x"][
                    1 + 2 * nmu : 1 + 2 * nmu + nind_loc
                ],  # Location initial parameter beta0, beta, betaT, beta_cov
                fit_result["x"][
                    1 + 2 * nmu + nind_loc : 2 + 2 * nmu + nind_loc + 2 * npsi + nind_sc
                ],  # Scale initial parameter alpha0, alpha, alpha_cov
                fit_result["x"][2 + 2 * nmu + nind_loc + 2 * npsi + nind_sc]
                * np.ones(self.ngamma0),
                fit_result["x"][
                    2 + 2 * nmu + nind_loc + 2 * npsi + nind_sc + self.ngamma0 : 2
                    + 2 * nmu
                    + nind_loc
                    + 2 * npsi
                    + nind_sc
                    + self.ngamma0
                    + 2 * ngamma
                    + nind_sh
                ]
                * np.ones(
                    2 * ngamma + nind_sh
                ),  # Shape initial parameter gamma0, gamma, gamma_cov
            ]
            xini = np.concatenate(
                [np.asarray(v) for v in concatvalues if v is not None]
            )
            fit_result_aux = self._fit(
                self.nmu,
                self.npsi,
                self.ngamma,
                self.list_loc,
                ntrend_loc,
                self.list_sc,
                ntrend_sc,
                self.list_sh,
                ntrend_sh,
                xini,
            )

            n_params = (
                2
                + self.ngamma0
                + 2 * nmu
                + 2 * npsi
                + 2 * ngamma
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc
                + ntrend_sh
                + nind_sh
            )
            self.AIC_iter[self.nit + 1] = self._AIC(
                -fit_result_aux["negloglikelihood"], n_params
            )
            self.loglike_iter[self.nit + 1] = -fit_result_aux["negloglikelihood"]

            if self.AIC_iter[self.nit + 1] < self.AICini:
                self.AICini = self.AIC_iter[self.nit + 1]
                print("Location trend is significative")
                print("Location trend AIC: ", self.AICini)
                # Update the parameters
                self._update_params(**fit_result_aux)
                self.ntrend_loc = ntrend_loc
            else:
                print("Location trend is NOT significative")
                self.ntrend_loc = 0

            # Scale trend
            ntrend_sc = 1

            concatvalues = [
                fit_result["x"][0 : 1 + 2 * nmu],
                np.zeros(self.ntrend_loc),
                fit_result["x"][
                    1 + 2 * nmu : 1 + 2 * nmu + nind_loc
                ],  # Location initial parameter beta0, beta, betaT, beta_cov
                fit_result["x"][
                    1 + 2 * nmu + nind_loc : 2 + 2 * nmu + nind_loc + 2 * npsi
                ],
                np.zeros(ntrend_sc),
                fit_result["x"][
                    2 + 2 * nmu + nind_loc + 2 * npsi : 2
                    + 2 * nmu
                    + nind_loc
                    + 2 * npsi
                    + nind_sc
                ],  # Scale initial parameter alpha0, alpha, alphaT, alpha_cov
                fit_result["x"][2 + 2 * nmu + nind_loc + 2 * npsi + nind_sc]
                * np.ones(self.ngamma0),
                fit_result["x"][
                    2 + 2 * nmu + nind_loc + 2 * npsi + nind_sc + self.ngamma0 : 2
                    + 2 * nmu
                    + nind_loc
                    + 2 * npsi
                    + nind_sc
                    + self.ngamma0
                    + 2 * ngamma
                    + nind_sh
                ]
                * np.ones(
                    2 * ngamma + nind_sh
                ),  # Shape initial parameter gamma0, gamma, gamma_cov
            ]
            xini = np.concatenate(
                [np.asarray(v) for v in concatvalues if v is not None]
            )

            fit_result = self._fit(
                self.nmu,
                self.npsi,
                self.ngamma,
                self.list_loc,
                self.ntrend_loc,
                self.list_sc,
                ntrend_sc,
                self.list_sh,
                ntrend_sh,
                xini,
            )

            n_params = (
                2
                + self.ngamma0
                + 2 * nmu
                + 2 * npsi
                + 2 * ngamma
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc
                + ntrend_sh
                + nind_sh
            )
            self.AIC_iter[self.nit + 2] = self._AIC(
                -fit_result["negloglikelihood"], n_params
            )
            self.loglike_iter[self.nit + 2] = -fit_result["negloglikelihood"]

            if self.AIC_iter[self.nit + 2] < self.AIC_iter[self.nit]:
                self.AICini = self.AIC_iter[self.nit + 2]
                print("Scale trend is significative")
                print("Scale trend AIC: ", self.AICini)
                # Update the parameters
                self._update_params(**fit_result)
                self.ntrend_sc = ntrend_sc
            else:
                print("Scale trend is NOT significative")
                self.ntrend_sc = 0

            # Shape trends
            ntrend_sh = 1

            concatvalues = [
                fit_result["x"][0 : 1 + 2 * nmu],
                np.zeros(self.ntrend_loc),
                fit_result["x"][
                    1 + 2 * nmu : 1 + 2 * nmu + nind_loc
                ],  # Location initial parameter beta0, beta, betaT, beta_cov
                fit_result["x"][
                    1 + 2 * nmu + nind_loc : 2 + 2 * nmu + nind_loc + 2 * npsi
                ],
                np.zeros(self.ntrend_sc),
                fit_result["x"][
                    2 + 2 * nmu + nind_loc + 2 * npsi : 2
                    + 2 * nmu
                    + nind_loc
                    + 2 * npsi
                    + nind_sc
                ],  # Scale initial parameter alpha0, alpha, alphaT, alpha_cov
                fit_result["x"][2 + 2 * nmu + nind_loc + 2 * npsi + nind_sc]
                * np.ones(self.ngamma0),
                fit_result["x"][
                    2 + 2 * nmu + nind_loc + 2 * npsi + nind_sc + self.ngamma0 : 2
                    + 2 * nmu
                    + nind_loc
                    + 2 * npsi
                    + nind_sc
                    + self.ngamma0
                    + 2 * ngamma
                ]
                * np.ones(2 * ngamma),
                0.01 * np.ones(ntrend_sh),
                fit_result["x"][
                    2
                    + 2 * nmu
                    + nind_loc
                    + 2 * npsi
                    + nind_sc
                    + self.ngamma0
                    + 2 * ngamma : 2
                    + 2 * nmu
                    + nind_loc
                    + 2 * npsi
                    + nind_sc
                    + self.ngamma0
                    + 2 * ngamma
                    + nind_sh
                ]
                * np.ones(
                    nind_sh
                ),  # Shape initial parameter gamma0, gamma, gammaT, gamma_cov
            ]
            xini = np.concatenate(
                [np.asarray(v) for v in concatvalues if v is not None]
            )
            fit_result = self._fit(
                self.nmu,
                self.npsi,
                self.ngamma,
                self.list_loc,
                self.ntrend_loc,
                self.list_sc,
                self.ntrend_sc,
                self.list_sh,
                ntrend_sh,
                xini,
            )

            n_params = (
                2
                + self.ngamma0
                + 2 * self.nmu
                + 2 * self.npsi
                + 2 * self.ngamma
                + self.ntrend_loc
                + self.nind_loc
                + ntrend_sc
                + self.nind_sc
                + ntrend_sh
                + self.nind_sh
            )
            self.AIC_iter[self.nit + 3] = self._AIC(
                -fit_result["negloglikelihood"], n_params
            )
            self.loglike_iter[self.nit + 3] = -fit_result["negloglikelihood"]

            if self.AIC_iter[self.nit + 3] < self.AIC_iter[self.nit]:
                self.AICini = self.AIC_iter[self.nit + 3]
                print("Shape trend is significative")
                print("Shape trend AIC: ", self.AICini)
                # Update the parameters
                self._update_params(**fit_result)
                self.ntrend_sh = ntrend_sh
            else:
                print("Shape trend is NOT significative")
                self.ntrend_sh = 0

        # Final parameters values
        concatvalues = [
            fit_result["x"][0 : 1 + 2 * nmu],
            np.zeros(self.ntrend_loc),
            fit_result["x"][
                1 + 2 * nmu : 1 + 2 * nmu + nind_loc
            ],  # Location initial parameter beta0, beta, betaT, beta_cov
            fit_result["x"][1 + 2 * nmu + nind_loc : 2 + 2 * nmu + nind_loc + 2 * npsi],
            np.zeros(self.ntrend_sc),
            fit_result["x"][
                2 + 2 * nmu + nind_loc + 2 * npsi : 2
                + 2 * nmu
                + nind_loc
                + 2 * npsi
                + nind_sc
            ],  # Scale initial parameter alpha0, alpha, alphaT, alpha_cov
            fit_result["x"][2 + 2 * nmu + nind_loc + 2 * npsi + nind_sc]
            * np.ones(self.ngamma0),
            fit_result["x"][
                2 + 2 * nmu + nind_loc + 2 * npsi + nind_sc + self.ngamma0 : 2
                + 2 * nmu
                + nind_loc
                + 2 * npsi
                + nind_sc
                + self.ngamma0
                + 2 * ngamma
            ]
            * np.ones(2 * ngamma),
            0.01 * np.ones(self.ntrend_sh),
            fit_result["x"][
                2
                + 2 * nmu
                + nind_loc
                + 2 * npsi
                + nind_sc
                + self.ngamma0
                + 2 * ngamma : 2
                + 2 * nmu
                + nind_loc
                + 2 * npsi
                + nind_sc
                + self.ngamma0
                + 2 * ngamma
                + nind_sh
            ]
            * np.ones(
                nind_sh
            ),  # Shape initial parameter gamma0, gamma, gammaT, gamma_cov
        ]
        xini = np.concatenate([np.asarray(v) for v in concatvalues if v is not None])
        fit_result = self._fit(
            self.nmu,
            self.npsi,
            self.ngamma,
            self.list_loc,
            self.ntrend_loc,
            self.list_sc,
            self.ntrend_sc,
            self.list_sh,
            self.ntrend_sh,
            xini,
        )
        self._update_params(**fit_result)

        n_params = (
            2
            + self.ngamma0
            + 2 * self.nmu
            + 2 * self.npsi
            + 2 * self.ngamma
            + self.ntrend_loc
            + self.nind_loc
            + self.ntrend_sc
            + self.nind_sc
            + self.ntrend_sh
            + self.nind_sh
        )

        # Compute the final loglikelihood and the information matrix
        f, Jx, Hxx = self._loglikelihood(
            beta0=self.beta0,
            beta=self.beta,
            betaT=self.betaT,
            beta_cov=self.beta_cov,
            alpha0=self.alpha0,
            alpha=self.alpha,
            alphaT=self.alphaT,
            alpha_cov=self.alpha_cov,
            gamma0=self.gamma0,
            gamma=self.gamma,
            gammaT=self.gammaT,
            gamma_cov=self.gamma_cov,
            list_loc=self.list_loc,
            list_sc=self.list_sc,
            list_sh=self.list_sh,
        )
        fit_result["loglikelihood"] = f
        fit_result["grad"] = Jx
        fit_result["hessian"] = Hxx

        self.invI0 = np.linalg.inv(-Hxx)
        fit_result["invI0"] = self.invI0

        std_params = np.sqrt(np.diag(self.invI0))
        self.std_params = std_params
        fit_result["std_params"] = std_params

        if plot:
            self.plot()

        return fit_result

    def _fit(
        self,
        nmu: int = 0,
        npsi: int = 0,
        ngamma: int = 0,
        list_loc: list = [],
        ntrend_loc: int = 0,
        list_sc: list = [],
        ntrend_sc: int = 0,
        list_sh: list = [],
        ntrend_sh: int = 0,
        xini: Optional[np.ndarray] = None,
        options: dict = None,
    ) -> dict:
        """
        Auxiliar function to determine the optimal parameters of given Non-Stationary GEV

        Parameters
        ----------
        nmu : int, default=0
            Number of parameters of harmonic part of location.
        npsi : int, default=0
            Number of parameters of harmonic part of scale.
        ngamma : int, default=0
            Number of parameters of harmonic part of shape.
        list_loc : list, default=[]
            List of indices of covariates to be included in the location parameter.
        ntrend_loc : int, default=0
            If trends in location are included.
        list_sc : list, default=[]
            List of indices of covariates to be included in the scale parameter.
        ntrend_sc : int, default=0
            If trends in scale are included.
        list_sh : list, default=[]
            List of indices of covariates to be included in the shape parameter.
        ntrend_sh : int, default=0
            If trends in shape are included.
        xini : Optional[np.ndarray], default = None
            Initial parameter if given.
        options : dict, default={
                "gtol": 1e-5,
                "xtol": 1e-5,
                "barrier_tol": 1e-4,
                "maxiter": 200,
            }
            Dictionary with the optimization options, see scipy.minimize method "trust-constr" options
        """
        # Fitting options
        if options is None:
            options = {
                "gtol": 1e-6,
                "xtol": 1e-10,
                "barrier_tol": 1e-6,
                "maxiter": 1000,
            }

        # Total number of parameters to be estimated
        nmu = 2 * nmu
        npsi = 2 * npsi
        ngamma = 2 * ngamma
        nind_loc = len(list_loc)
        nind_sc = len(list_sc)
        nind_sh = len(list_sh)

        # Initialize the parameters to be fitted
        n_params = (
            2
            + self.ngamma0
            + nmu
            + npsi
            + ngamma
            + ntrend_loc
            + nind_loc
            + ntrend_sc
            + nind_sc
            + ntrend_sh
            + nind_sh
        )
        if xini is not None:
            if xini.size != n_params:
                raise ValueError(
                    "Check the initial guess of fitting step (_fit function)"
                )
            x_ini = xini.copy()
        else:
            x_ini = np.zeros(n_params)
            x_ini[0] = np.mean(self.xt)  # Initial value for intercept location
            x_ini[1 + nmu + nind_loc + ntrend_loc] = np.log(
                np.std(self.xt)
            )  # Initial value for intercept scale
            if self.ngamma0 == 1:
                x_ini[2 + nmu + npsi + nind_loc + ntrend_loc + ntrend_sc + nind_sc] = (
                    0.1  # Initial value for intercept shape
                )
            if ngamma > 0:
                x_ini[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + nind_loc
                    + ntrend_loc
                    + ntrend_sc
                    + nind_sc : 2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + nind_loc
                    + ntrend_loc
                    + ntrend_sc
                    + nind_sc
                    + ngamma
                ] = 0.01
            # If trend in shape is included
            if ntrend_sh > 0:
                x_ini[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + nind_loc
                    + ntrend_loc
                    + ntrend_sc
                    + nind_sc
                    + ngamma
                ] = 0.01
            # If covariates in shape are included
            if nind_sh > 0:
                x_ini[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + nind_loc
                    + ntrend_loc
                    + ntrend_sc
                    + nind_sc
                    + ngamma
                    + ntrend_sh : 2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + nind_loc
                    + ntrend_loc
                    + ntrend_sc
                    + nind_sc
                    + ngamma
                    + ntrend_sh
                    + nind_sh
                ] = 0.01

        # Set bounds for all the parameters
        lb = -np.inf * np.ones(n_params)
        ub = np.inf * np.ones(n_params)

        # Initial bounds for the parameters related to the shape, gamma0 and gamma
        if self.ngamma0 == 1:
            lb[2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc] = -0.25
            ub[2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc] = 0.25

        if ngamma > 0:
            lb[
                2
                + self.ngamma0
                + nmu
                + npsi
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc : 2
                + self.ngamma0
                + nmu
                + npsi
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc
                + ngamma
            ] = -0.15
            ub[
                2
                + self.ngamma0
                + nmu
                + npsi
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc : 2
                + self.ngamma0
                + nmu
                + npsi
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc
                + ngamma
            ] = 0.15

        if ntrend_sh > 0:
            lb[
                2
                + self.ngamma0
                + nmu
                + npsi
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc
                + ngamma
            ] = -0.15
            ub[
                2
                + self.ngamma0
                + nmu
                + npsi
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc
                + ngamma
            ] = 0.15

        if nind_sh > 0:
            lb[
                2
                + self.ngamma0
                + nmu
                + npsi
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc
                + ngamma
                + ntrend_sh : 2
                + self.ngamma0
                + nmu
                + npsi
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc
                + ngamma
                + ntrend_sh
                + nind_sh
            ] = -0.15
            ub[
                2
                + self.ngamma0
                + nmu
                + npsi
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc
                + ngamma
                + ntrend_sh : 2
                + self.ngamma0
                + nmu
                + npsi
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc
                + ngamma
                + ntrend_sh
                + nind_sh
            ] = 0.15

        # Initialize the return dictionary
        fit_result = {}

        # If an inital value for the parameters vector is provided, it is used
        # if pini is not None and len(pini) > 0:
        #     x_ini = np.minimum(pini, ub)
        #     x_ini = np.maximum(x, lb)

        # Set the bounds properly for scipy.optimize.minimize
        bounds = [(lb_i, up_i) for lb_i, up_i in zip(lb, ub)]
        result = minimize(
            fun=self._auxmin_loglikelihood,
            jac=self._auxmin_loglikelihood_grad,  # Gradient information
            hess=self._auxmin_loglikelihood_hess,  # Hessian information, if applicable
            x0=x_ini,
            bounds=bounds,
            args=(
                nmu,
                npsi,
                ngamma,
                ntrend_loc,
                list_loc,
                ntrend_sc,
                list_sc,
                ntrend_sh,
                list_sh,
            ),
            options=options,  # Options
            method="trust-constr",
        )

        fit_result["x"] = result.x  # Optimal parameters vector
        fit_result["negloglikelihood"] = result.fun  # Optimal loglikelihood
        fit_result["success"] = result.success
        fit_result["message"] = result.message
        fit_result["grad"] = result.grad
        fit_result["hess_inv"] = (
            result.hess_inv if "hess_inv" in result else None
        )  # 'hess_inv' is only available if 'hess' is provided

        # Check if any of the bounds related to shape parameters become active, if active increase or decrease the bound and call the optimization routine again
        lambdas = result.v
        auxlb = []
        auxub = []
        for i, v in enumerate(lambdas[0]):
            if np.abs(fit_result["x"][i] - lb[i]) <= 1e-6 or v < -1e-6:
                lb[i] -= 0.05
                auxlb.append(i)
            if np.abs(fit_result["x"][i] - ub[i]) <= 1e-6 or v > 1e-6:
                ub[i] += 0.05
                auxub.append(i)

        it = 0
        while (len(auxlb) > 0 or len(auxub) > 0) and it < 10:
            it += 1
            result = minimize(
                fun=self._auxmin_loglikelihood,
                jac=self._auxmin_loglikelihood_grad,  # Gradient information
                hess=self._auxmin_loglikelihood_hess,  # Hessian information, if applicable
                x0=fit_result["x"],
                bounds=bounds,
                args=(
                    nmu,
                    npsi,
                    ngamma,
                    ntrend_loc,
                    list_loc,
                    ntrend_sc,
                    list_sc,
                    ntrend_sh,
                    list_sh,
                ),
                options=options,  # Options
                method="trust-constr",
            )

            fit_result["x"] = result.x  # Optimal parameters vector
            fit_result["negloglikelihood"] = result.fun  # Optimal loglikelihood
            fit_result["AIC"] = self._AIC(-fit_result["negloglikelihood"], n_params)
            fit_result["n_params"] = n_params
            fit_result["success"] = result.success
            fit_result["message"] = result.message
            fit_result["grad"] = result.grad
            fit_result["hess_inv"] = (
                result.hess_inv if "hess_inv" in result else None
            )  # 'hess_inv' is only available if 'hess' is provided
            fit_result["lambdas"] = result.v
            auxlb = []
            auxub = []
            for i, v in enumerate(lambdas[0]):
                if np.abs(fit_result["x"][i] - lb[i]) <= 1e-6 or v < -1e-6:
                    lb[i] -= 0.05
                    auxlb.append(i)
                if np.abs(fit_result["x"][i] - ub[i]) <= 1e-6 or v > 1e-6:
                    ub[i] += 0.05
                    auxub.append(i)

        # Location parameter
        fit_result["beta0"] = fit_result["x"][0]
        if nmu > 0:
            fit_result["beta"] = fit_result["x"][1 : 1 + nmu]
        else:
            fit_result["beta"] = np.empty(0)
        if ntrend_loc > 0:
            fit_result["betaT"] = fit_result["x"][1 + nmu]
        else:
            fit_result["betaT"] = np.empty(0)
        if nind_loc > 0:
            fit_result["beta_cov"] = fit_result["x"][
                1 + nmu + ntrend_loc : 1 + nmu + ntrend_loc + nind_loc
            ]
        else:
            fit_result["beta_cov"] = np.empty(0)

        # Scale parameter
        fit_result["alpha0"] = fit_result["x"][1 + nmu + ntrend_loc + nind_loc]
        if npsi > 0:
            fit_result["alpha"] = fit_result["x"][
                2 + nmu + ntrend_loc + nind_loc : 2 + nmu + ntrend_loc + nind_loc + npsi
            ]
        else:
            fit_result["alpha"] = np.empty(0)
        if ntrend_sc > 0:
            fit_result["alphaT"] = fit_result["x"][
                2 + nmu + ntrend_loc + nind_loc + npsi
            ]
        else:
            fit_result["alphaT"] = np.empty(0)
        if nind_sc > 0:
            fit_result["alpha_cov"] = fit_result["x"][
                2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc : 2
                + nmu
                + ntrend_loc
                + nind_loc
                + npsi
                + ntrend_sc
                + nind_sc
            ]
        else:
            fit_result["alpha_cov"] = np.empty(0)

        # Shape parameter
        if self.ngamma0 == 1:
            fit_result["gamma0"] = fit_result["x"][
                2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + nind_sc
            ]
        else:
            fit_result["gamma0"] = np.empty(0)
        if ngamma > 0:
            fit_result["gamma"] = fit_result["x"][
                2
                + self.ngamma0
                + nmu
                + ntrend_loc
                + nind_loc
                + npsi
                + ntrend_sc
                + nind_sc : 2
                + self.ngamma0
                + nmu
                + ntrend_loc
                + nind_loc
                + npsi
                + ntrend_sc
                + nind_sc
                + ngamma
            ]
        else:
            fit_result["gamma"] = np.empty(0)
        if ntrend_sh > 0:
            fit_result["gammaT"] = fit_result["x"][
                2
                + self.ngamma0
                + nmu
                + ntrend_loc
                + nind_loc
                + npsi
                + ntrend_sc
                + nind_sc
                + ngamma
            ]
        else:
            fit_result["gammaT"] = np.empty(0)
        if nind_sh > 0:
            fit_result["gamma_cov"] = fit_result["x"][
                2
                + self.ngamma0
                + nmu
                + ntrend_loc
                + nind_loc
                + npsi
                + ntrend_sc
                + nind_sc
                + ngamma : 2
                + self.ngamma0
                + nmu
                + ntrend_loc
                + nind_loc
                + npsi
                + ntrend_sc
                + nind_sc
                + ngamma
                + nind_sh
            ]
        else:
            fit_result["gamma_cov"] = np.empty(0)

        self.fit_result = fit_result

        return fit_result

    def _auxmin_loglikelihood(
        self,
        x: np.ndarray,
        nmu: int = 0,
        npsi: int = 0,
        ngamma: int = 0,
        ntrend_loc: int = 0,
        list_loc: list = [],
        ntrend_sc: int = 0,
        list_sc: list = [],
        ntrend_sh: int = 0,
        list_sh: list = [],
    ) -> float:
        """
        Function used for minimizing in the 'self._fit' where the Negative loglikelihood of the GEV will be minimized

        Parameters
        ----------
        x : np.ndarray
            Parameter vector to optimize
        nmu : int, default=0
            Number of harmonics included in location
        npsi : int, default=0
            Number of harmonics included in scale
        ngamma : int, default=0
            Number of harmonics included in shape
        ntrend_loc : int, default=0
            Whether to include trends in location
        list_loc : list, default=[]
            List of covariates indices to include in location
        ntrend_sc : int, default=0
            Whether to include trends in scale
        list_sc : list, default=[]
            List of covariates indices to include in scale
        ntrend_sh : int, default=0
            Whether to include trends in shape
        list_sh : list, default=[]
            List of covariates indices to include in shape

        Return
        ------
        f : float
            Negative loglikelihood value of the Non-stationary GEV
        """
        # Cheking the inputs
        covariates_loc = self.covariates.iloc[:, list_loc].values
        covariates_sc = self.covariates.iloc[:, list_sc].values
        covariates_sh = self.covariates.iloc[:, list_sh].values

        # Check consistency of the data
        na1, nind_loc = covariates_loc.shape
        if nind_loc > 0 and na1 > 0:
            if na1 != len(self.xt) or na1 != len(self.t) or len(self.xt) != len(self.t):
                ValueError("Check data x, t, indices: funcion aux loglikelihood")

        na2, nind_sc = covariates_sc.shape
        if nind_sc > 0 and na2 > 0:
            if na2 != len(self.xt) or na2 != len(self.t) or len(self.xt) != len(self.t):
                ValueError("Check data x, t, indices: funcion aux loglikelihood")

        na3, nind_sh = covariates_sh.shape
        if nind_sh > 0 and na3 > 0:
            if na3 != len(self.xt) or na3 != len(self.t) or len(self.xt) != len(self.t):
                ValueError("Check data x, t, indices: funcion aux loglikelihood")

        # Evaluate the location parameter at each time t as a function of the actual values of the parameters given by x
        if ntrend_loc == 0 and nind_loc == 0:
            mut1 = self._parametro(x[0], x[1 : 1 + nmu])  # beta0, beta
        elif ntrend_loc == 0 and nind_loc != 0:
            mut1 = self._parametro(
                x[0],
                x[1 : 1 + nmu],
                None,
                x[1 + nmu + ntrend_loc : 1 + nmu + ntrend_loc + nind_loc],
                covariates_loc,
            )  # beta0, beta, beta_cov
        elif ntrend_loc != 0 and nind_loc == 0:
            mut1 = self._parametro(
                x[0], x[1 : 1 + nmu], x[1 + nmu]
            )  # beta0, beta, betaT
        else:
            mut1 = self._parametro(
                x[0],
                x[1 : 1 + nmu],
                x[1 + nmu],
                x[1 + nmu + ntrend_loc : 1 + nmu + ntrend_loc + nind_loc],
                covariates_loc,
            )  # beta0, beta, betaT, beta_cov

        # Evaluate the scale parameter at each time t as a function of the actual values of the parameters given by x
        if ntrend_sc == 0 and nind_sc == 0:
            psit1 = np.exp(
                self._parametro(
                    x[1 + nmu + ntrend_loc + nind_loc],
                    x[
                        2 + nmu + ntrend_loc + nind_loc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                    ],
                )
            )  # alpha0, alpha
        elif ntrend_sc == 0 and nind_sc != 0:
            psit1 = np.exp(
                self._parametro(
                    x[1 + nmu + ntrend_loc + nind_loc],
                    x[
                        2 + nmu + ntrend_loc + nind_loc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                    ],
                    None,
                    x[
                        2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                    ],
                    covariates_sc,
                )
            )  # alpha0, alpha, alpha_cov
        elif ntrend_sc != 0 and nind_sc == 0:
            psit1 = np.exp(
                self._parametro(
                    x[1 + nmu + ntrend_loc + nind_loc],
                    x[
                        2 + nmu + ntrend_loc + nind_loc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                    ],
                    x[2 + nmu + ntrend_loc + nind_loc + npsi],
                )
            )  # alpha0, alpha, alphaT
        else:
            psit1 = np.exp(
                self._parametro(
                    x[1 + nmu + ntrend_loc + nind_loc],
                    x[
                        2 + nmu + ntrend_loc + nind_loc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                    ],
                    x[2 + nmu + ntrend_loc + nind_loc + npsi],
                    x[
                        2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                    ],
                    covariates_sc,
                )
            )  # alpha0, alpha, alphaT, alpha_cov

        # Evaluate the shape parameter at each time t as a function of the actual values of the parameters given by x
        if self.ngamma0 != 0:
            if ntrend_sh == 0 and nind_sh == 0:
                epst = self._parametro(
                    x[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + nind_sc],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                )  # gamma0, gamma
            elif ntrend_sh == 0 and nind_sh != 0:
                epst = self._parametro(
                    x[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + nind_sc],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    None,
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                        + nind_sh
                    ],
                    covariates_sh,
                )  # gamma0, gamma, gamma_cov
            elif ntrend_sh != 0 and nind_sh == 0:
                epst = self._parametro(
                    x[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + nind_sc],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                )  # gamma0, gamma, gammaT
            else:
                epst = self._parametro(
                    x[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + nind_sc],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                        + nind_sh
                    ],
                    covariates_sh,
                )  # gamma0, gamma, gammaT, gamma_cov
        # If intercept in shape is not included
        else:
            if ntrend_sh == 0 and nind_sh == 0:
                epst = self._parametro(
                    None,
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                )  # gamma
            elif ntrend_sh == 0 and nind_sh != 0:
                epst = self._parametro(
                    None,
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    None,
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                        + nind_sh
                    ],
                    covariates_sh,
                )  # gamma, gamma_cov
            elif ntrend_sh != 0 and nind_sh == 0:
                epst = self._parametro(
                    None,
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                )  # gamma, gammaT
            else:
                epst = self._parametro(
                    None,
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                        + nind_sh
                    ],
                    covariates_sh,
                )  # gamma, gammaT, gamma_cov

        # The values whose shape parameter is almost 0 correspond to the Gumbel distribution
        posG = list(np.where(np.abs(epst) <= 1e-8)[0])
        # The remaining values correspond to Weibull or Frechet
        pos = list(np.where(np.abs(epst) > 1e-8)[0])

        # The corresponding Gumbel values are set to 1 to avoid numerical problems, note that in this case, the Gumbel expressions are used
        epst[posG] = 1

        # Modify the parameters to include the length of the data
        mut = mut1
        psit = psit1
        mut[pos] = mut1[pos] + psit1[pos] * (self.kt[pos] ** epst[pos] - 1) / epst[pos]
        psit[pos] = psit1[pos] * self.kt[pos] ** epst[pos]
        # Modify the parameters to include the length of the data in Gumbel
        mut[posG] = mut[posG] + psit[posG] * np.log(self.kt[posG])

        # Evaluate auxiliarly variables
        xn = (self.xt - mut) / psit
        z = 1 + epst * xn

        # Since z-values must be greater than 0 in order to avoid numerical problems, their values are set to be greater than 1e-8
        z = np.maximum(1e-8, z)
        zn = z ** (-1 / epst)

        # Evaluate the loglikelihood function with the sign changed, not that the general and Gumbel expressions are used
        f = np.sum(
            -np.log(self.kt[pos])
            + np.log(psit[pos])
            + (1 + 1 / epst[pos]) * np.log(z[pos])
            + self.kt[pos] * zn[pos]
        ) + np.sum(
            -np.log(self.kt[posG])
            + np.log(psit[posG])
            + xn[posG]
            + self.kt[posG] * np.exp(-xn[posG])
        )

        return f

    def _auxmin_loglikelihood_grad(
        self,
        x,
        nmu,
        npsi,
        ngamma,
        ntrend_loc=0,
        list_loc=[],
        ntrend_sc=0,
        list_sc=[],
        ntrend_sh=0,
        list_sh=[],
    ) -> np.ndarray:
        """
        Function used for minimizing in the 'self._optimize_parameters' where the Negative loglikelihood of the GEV will be minimized

        Parameters
        ----------
        x : np.ndarray
            Parameter vector to optimize
        nmu : int, default=0
            Number of harmonics included in location
        npsi : int, default=0
            Number of harmonics included in scale
        ngamma : int, default=0
            Number of harmonics included in shape
        ntrend_loc : int, default=0
            Whether to include trends in location
        list_loc : list, default=[]
            List of covariates indices to include in location
        ntrend_sc : int, default=0
            Whether to include trends in scale
        list_sc : list, default=[]
            List of covariates indices to include in scale
        ntrend_sh : int, default=0
            Whether to include trends in shape
        list_sh : list, default=[]
            List of covariates indices to include in shape

        Return
        ------
        Jx : np.ndarray
            Gradient of negative loglikelihood value of the Non-stationary GEV
        """
        # Cheking the inputs
        covariates_loc = self.covariates.iloc[:, list_loc].values
        covariates_sc = self.covariates.iloc[:, list_sc].values
        covariates_sh = self.covariates.iloc[:, list_sh].values

        # Check consistency of the data
        na1, nind_loc = covariates_loc.shape
        if nind_loc > 0 and na1 > 0:
            if na1 != len(self.xt) or na1 != len(self.t) or len(self.xt) != len(self.t):
                ValueError("Check data x, t, indices: funcion aux loglikelihood")

        na2, nind_sc = covariates_sc.shape
        if nind_sc > 0 and na2 > 0:
            if na2 != len(self.xt) or na2 != len(self.t) or len(self.xt) != len(self.t):
                ValueError("Check data x, t, indices: funcion aux loglikelihood")

        na3, nind_sh = covariates_sh.shape
        if nind_sh > 0 and na3 > 0:
            if na3 != len(self.xt) or na3 != len(self.t) or len(self.xt) != len(self.t):
                ValueError("Check data x, t, indices: funcion aux loglikelihood")

        # Evaluate the location parameter at each time t as a function of the actual values of the parameters given by x
        if ntrend_loc == 0 and nind_loc == 0:
            mut1 = self._parametro(x[0], x[1 : 1 + nmu])  # beta0, beta
        elif ntrend_loc == 0 and nind_loc != 0:
            mut1 = self._parametro(
                x[0],
                x[1 : 1 + nmu],
                None,
                x[1 + nmu + ntrend_loc : 1 + nmu + ntrend_loc + nind_loc],
                covariates_loc,
            )  # beta0, beta, beta_cov
        elif ntrend_loc != 0 and nind_loc == 0:
            mut1 = self._parametro(
                x[0], x[1 : 1 + nmu], x[1 + nmu]
            )  # beta0, beta, betaT
        else:
            mut1 = self._parametro(
                x[0],
                x[1 : 1 + nmu],
                x[1 + nmu],
                x[1 + nmu + ntrend_loc : 1 + nmu + ntrend_loc + nind_loc],
                covariates_loc,
            )  # beta0, beta, betaT, beta_cov

        # Evaluate the scale parameter at each time t as a function of the actual values of the parameters given by x
        if ntrend_sc == 0 and nind_sc == 0:
            psit1 = np.exp(
                self._parametro(
                    x[1 + nmu + ntrend_loc + nind_loc],
                    x[
                        2 + nmu + ntrend_loc + nind_loc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                    ],
                )
            )  # alpha0, alpha
        elif ntrend_sc == 0 and nind_sc != 0:
            psit1 = np.exp(
                self._parametro(
                    x[1 + nmu + ntrend_loc + nind_loc],
                    x[
                        2 + nmu + ntrend_loc + nind_loc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                    ],
                    None,
                    x[
                        2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                    ],
                    covariates_sc,
                )
            )  # alpha0, alpha, alpha_cov
        elif ntrend_sc != 0 and nind_sc == 0:
            psit1 = np.exp(
                self._parametro(
                    x[1 + nmu + ntrend_loc + nind_loc],
                    x[
                        2 + nmu + ntrend_loc + nind_loc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                    ],
                    x[2 + nmu + ntrend_loc + nind_loc + npsi],
                )
            )  # alpha0, alpha, alphaT
        else:
            psit1 = np.exp(
                self._parametro(
                    x[1 + nmu + ntrend_loc + nind_loc],
                    x[
                        2 + nmu + ntrend_loc + nind_loc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                    ],
                    x[2 + nmu + ntrend_loc + nind_loc + npsi],
                    x[
                        2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                    ],
                    covariates_sc,
                )
            )  # alpha0, alpha, alphaT, alpha_cov

        # Evaluate the shape parameter at each time t as a function of the actual values of the parameters given by x
        if self.ngamma0 != 0:
            if ntrend_sh == 0 and nind_sh == 0:
                epst = self._parametro(
                    x[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + nind_sc],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                )  # gamma0, gamma
            elif ntrend_sh == 0 and nind_sh != 0:
                epst = self._parametro(
                    x[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + nind_sc],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    None,
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                        + nind_sh
                    ],
                    covariates_sh,
                )  # gamma0, gamma, gamma_cov
            elif ntrend_sh != 0 and nind_sh == 0:
                epst = self._parametro(
                    x[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + nind_sc],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                )  # gamma0, gamma, gammaT
            else:
                epst = self._parametro(
                    x[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + nind_sc],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                        + nind_sh
                    ],
                    covariates_sh,
                )  # gamma0, gamma, gammaT, gamma_cov
        # If intercept in shape is not included
        else:
            if ntrend_sh == 0 and nind_sh == 0:
                epst = self._parametro(
                    None,
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                )  # gamma
            elif ntrend_sh == 0 and nind_sh != 0:
                epst = self._parametro(
                    None,
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    None,
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                        + nind_sh
                    ],
                    covariates_sh,
                )  # gamma, gamma_cov
            elif ntrend_sh != 0 and nind_sh == 0:
                epst = self._parametro(
                    None,
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                )  # gamma, gammaT
            else:
                epst = self._parametro(
                    None,
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                        + nind_sh
                    ],
                    covariates_sh,
                )  # gamma, gammaT, gamma_cov

        # The values whose shape parameter is almost 0 correspond to the Gumbel distribution
        posG = np.where(np.abs(epst) <= 1e-8)[0]
        # The remaining values correspond to Weibull or Frechet
        pos = np.where(np.abs(epst) > 1e-8)[0]

        # The corresponding Gumbel values are set to 1 to avoid numerical problems, note that in this case, the Gumbel expressions are used
        epst[posG] = 1

        # Modify the parameters to include the length of the data
        mut = mut1
        psit = psit1
        mut[pos] = mut1[pos] + psit1[pos] * (self.kt[pos] ** epst[pos] - 1) / epst[pos]
        psit[pos] = psit1[pos] * self.kt[pos] ** epst[pos]
        # Modify the parameters to include the length of the data in Gumbel
        mut[posG] = mut[posG] + psit[posG] * np.log(self.kt[posG])

        # Evaluate auxiliarly variables
        xn = (self.xt - mut) / psit
        z = 1 + epst * xn

        # Since z-values must be greater than 0 in order to avoid numerical problems, their values are set to be greater than 1e-8
        z = np.maximum(1e-8, z)
        zn = z ** (-1 / epst)

        ### Gradient of the loglikelihood
        # Derivatives given by equations (A.1)-(A.3) in the paper
        Dmut = (1 + epst - self.kt * zn) / (psit * z)
        Dpsit = -(1 - xn * (1 - self.kt * zn)) / (psit * z)
        Depst = (
            zn
            * (
                xn * (self.kt - (1 + epst) / zn)
                + z * (-self.kt + 1 / zn) * np.log(z) / epst
            )
            / (epst * z)
        )

        # Gumbel derivatives given by equations (A.4)-(A.5) in the paper
        Dmut[posG] = (1 - self.kt[posG] * np.exp(-xn[posG])) / psit[posG]
        Dpsit[posG] = (xn[posG] - 1 - self.kt[posG] * xn[posG] * np.exp(-xn[posG])) / (
            psit[posG]
        )
        Depst[posG] = 0

        ## New Derivatives
        Dmutastmut = np.ones_like(self.kt)
        Dmutastpsit = (-1 + self.kt**epst) / epst
        Dmutastepst = (
            psit1 * (1 + (self.kt**epst) * (epst * np.log(self.kt) - 1)) / (epst**2)
        )

        Dpsitastpsit = self.kt**epst
        Dpsitastepst = np.log(self.kt) * psit1 * (self.kt**epst)

        Dmutastpsit[posG] = np.log(self.kt[posG])
        Dmutastepst[posG] = 0

        Dpsitastpsit[posG] = 1
        Dpsitastepst[posG] = 0

        # Set the Jacobian to zero
        Jx = np.zeros(
            2
            + self.ngamma0
            + nmu
            + npsi
            + ngamma
            + ntrend_loc
            + nind_loc
            + ntrend_sc
            + nind_sc
            + ntrend_sh
            + nind_sh
        )
        # Jacobian elements related to the location parameters beta0 and beta, equation (A.6) in the paper
        Jx[0] = np.dot(Dmut, Dmutastmut)

        # If location harmonics are included
        if nmu > 0:
            for i in range(nmu):
                aux = 0
                for k in range(len(self.t)):
                    aux += Dmut[k] * Dmutastmut[k] * self._Dparam(self.t[k], i + 1)
                Jx[i + 1] = aux

        # Jacobian elements related to the location parameters betaT, beta_cov (equation A.9)
        if ntrend_loc > 0:
            Jx[1 + nmu] = np.sum(Dmut * self.t * Dmutastmut)  # betaT
        if nind_loc > 0:
            for i in range(nind_loc):
                Jx[1 + nmu + ntrend_loc + i] = np.sum(
                    Dmut * covariates_loc[:, i] * Dmutastmut
                )  # beta_cov_i

        # Jacobian elements related to the scale parameters alpha0, alpha (equation A.7)
        Jx[1 + nmu + ntrend_loc + nind_loc] = np.sum(
            psit1 * (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit)
        )  # alpha0
        # If scale harmonic are included
        if npsi > 0:
            for i in range(npsi):
                aux = 0
                for k in range(len(self.t)):
                    aux += (
                        self._Dparam(self.t[k], i + 1)
                        * psit1[k]
                        * (Dpsit[k] * Dpsitastpsit[k] + Dmut[k] * Dmutastpsit[k])
                    )
                Jx[2 + nmu + ntrend_loc + nind_loc + i] = aux  # alpha
        # Jacobian elements related to the scale parameters alphaT and beta_cov (equation A.10)
        if ntrend_sc > 0:
            Jx[2 + nmu + ntrend_loc + nind_loc + npsi] = np.sum(
                (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit) * self.t * psit1
            )  # alphaT
        if nind_sc > 0:
            for i in range(nind_sc):
                Jx[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i] = np.sum(
                    (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit)
                    * covariates_sc[:, i]
                    * psit1
                )  # alpha_cov

        # Jacobian elements related to the shape parameters gamma0 and gamma (equation A.10)
        if self.ngamma0 == 1:
            Jx[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + nind_sc] = np.sum(
                Depst + Dpsit * Dpsitastepst + Dmut * Dmutastepst
            )
        # If shape harmonics are included
        if ngamma > 0:
            for i in range(ngamma):
                aux = 0
                for k in range(len(self.t)):
                    aux += (
                        Depst[k] + Dpsit[k] * Dpsitastepst[k] + Dmut[k] * Dmutastepst[k]
                    ) * self._Dparam(self.t[k], i + 1)
                Jx[
                    2
                    + self.ngamma0
                    + nmu
                    + ntrend_loc
                    + nind_loc
                    + npsi
                    + ntrend_sc
                    + nind_sc
                    + i
                ] = aux

        # Jacobian elements related to the shape parameter gamma (defined by Victor)
        if ntrend_sh > 0:
            Jx[
                2
                + self.ngamma0
                + nmu
                + ntrend_loc
                + nind_loc
                + npsi
                + ntrend_sc
                + nind_sc
                + ngamma
            ] = np.sum(Depst * self.t)

        # Jacobian elements related to the shape parameters gamma_cov (defined by Victor)
        if nind_sh > 0:
            for i in range(nind_sh):
                Jx[
                    2
                    + self.ngamma0
                    + nmu
                    + ntrend_loc
                    + nind_loc
                    + npsi
                    + ntrend_sc
                    + nind_sc
                    + ngamma
                    + ntrend_sh
                    + i
                ] = np.sum(Depst * covariates_sh[:, i])

        # Change the Jacobian sign since the numerical problem is a minimization problem
        Jx = -Jx

        return Jx

    def _auxmin_loglikelihood_hess(
        self,
        x,
        nmu,
        npsi,
        ngamma,
        ntrend_loc=0,
        list_loc=[],
        ntrend_sc=0,
        list_sc=[],
        ntrend_sh=0,
        list_sh=[],
    ) -> np.ndarray:
        """
        Function used for minimizing in the 'self._optimize_parameters' where the Negative loglikelihood of the GEV will be minimized

        Parameters
        ----------
        x : np.ndarray
            Parameter vector to optimize
        nmu : int, default=0
            Number of harmonics included in location
        npsi : int, default=0
            Number of harmonics included in scale
        ngamma : int, default=0
            Number of harmonics included in shape
        ntrend_loc : int, default=0
            Whether to include trends in location
        list_loc : list, default=[]
            List of covariates indices to include in location
        ntrend_sc : int, default=0
            Whether to include trends in scale
        list_sc : list, default=[]
            List of covariates indices to include in scale
        ntrend_sh : int, default=0
            Whether to include trends in shape
        list_sh : list, default=[]
            List of covariates indices to include in shape

        Return
        ------
        Jx : np.ndarray
            Gradient of negative loglikelihood value of the Non-stationary GEV
        """
        # Cheking the inputs
        covariates_loc = self.covariates.iloc[:, list_loc].values
        covariates_sc = self.covariates.iloc[:, list_sc].values
        covariates_sh = self.covariates.iloc[:, list_sh].values

        # Check consistency of the data
        na1, nind_loc = covariates_loc.shape
        if nind_loc > 0 and na1 > 0:
            if na1 != len(self.xt) or na1 != len(self.t) or len(self.xt) != len(self.t):
                ValueError("Check data x, t, indices: funcion aux loglikelihood")

        na2, nind_sc = covariates_sc.shape
        if nind_sc > 0 and na2 > 0:
            if na2 != len(self.xt) or na2 != len(self.t) or len(self.xt) != len(self.t):
                ValueError("Check data x, t, indices: funcion aux loglikelihood")

        na3, nind_sh = covariates_sh.shape
        if nind_sh > 0 and na3 > 0:
            if na3 != len(self.xt) or na3 != len(self.t) or len(self.xt) != len(self.t):
                ValueError("Check data x, t, indices: funcion aux loglikelihood")

        # Evaluate the location parameter at each time t as a function of the actual values of the parameters given by x
        if ntrend_loc == 0 and nind_loc == 0:
            mut1 = self._parametro(x[0], x[1 : 1 + nmu])  # beta0, beta
        elif ntrend_loc == 0 and nind_loc != 0:
            mut1 = self._parametro(
                x[0],
                x[1 : 1 + nmu],
                None,
                x[1 + nmu + ntrend_loc : 1 + nmu + ntrend_loc + nind_loc],
                covariates_loc,
            )  # beta0, beta, beta_cov
        elif ntrend_loc != 0 and nind_loc == 0:
            mut1 = self._parametro(
                x[0], x[1 : 1 + nmu], x[1 + nmu]
            )  # beta0, beta, betaT
        else:
            mut1 = self._parametro(
                x[0],
                x[1 : 1 + nmu],
                x[1 + nmu],
                x[1 + nmu + ntrend_loc : 1 + nmu + ntrend_loc + nind_loc],
                covariates_loc,
            )  # beta0, beta, betaT, beta_cov

        # Evaluate the scale parameter at each time t as a function of the actual values of the parameters given by x
        if ntrend_sc == 0 and nind_sc == 0:
            psit1 = np.exp(
                self._parametro(
                    x[1 + nmu + ntrend_loc + nind_loc],
                    x[
                        2 + nmu + ntrend_loc + nind_loc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                    ],
                )
            )  # alpha0, alpha
        elif ntrend_sc == 0 and nind_sc != 0:
            psit1 = np.exp(
                self._parametro(
                    x[1 + nmu + ntrend_loc + nind_loc],
                    x[
                        2 + nmu + ntrend_loc + nind_loc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                    ],
                    None,
                    x[
                        2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                    ],
                    covariates_sc,
                )
            )  # alpha0, alpha, alpha_cov
        elif ntrend_sc != 0 and nind_sc == 0:
            psit1 = np.exp(
                self._parametro(
                    x[1 + nmu + ntrend_loc + nind_loc],
                    x[
                        2 + nmu + ntrend_loc + nind_loc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                    ],
                    x[2 + nmu + ntrend_loc + nind_loc + npsi],
                )
            )  # alpha0, alpha, alphaT
        else:
            psit1 = np.exp(
                self._parametro(
                    x[1 + nmu + ntrend_loc + nind_loc],
                    x[
                        2 + nmu + ntrend_loc + nind_loc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                    ],
                    x[2 + nmu + ntrend_loc + nind_loc + npsi],
                    x[
                        2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                    ],
                    covariates_sc,
                )
            )  # alpha0, alpha, alphaT, alpha_cov

        # Evaluate the shape parameter at each time t as a function of the actual values of the parameters given by x
        if self.ngamma0 != 0:
            if ntrend_sh == 0 and nind_sh == 0:
                epst = self._parametro(
                    x[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + nind_sc],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                )  # gamma0, gamma
            elif ntrend_sh == 0 and nind_sh != 0:
                epst = self._parametro(
                    x[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + nind_sc],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    None,
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                        + nind_sh
                    ],
                    covariates_sh,
                )  # gamma0, gamma, gamma_cov
            elif ntrend_sh != 0 and nind_sh == 0:
                epst = self._parametro(
                    x[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + nind_sc],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                )  # gamma0, gamma, gammaT
            else:
                epst = self._parametro(
                    x[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + nind_sc],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                        + nind_sh
                    ],
                    covariates_sh,
                )  # gamma0, gamma, gammaT, gamma_cov
        # If intercept in shape is not included
        else:
            if ntrend_sh == 0 and nind_sh == 0:
                epst = self._parametro(
                    None,
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                )  # gamma
            elif ntrend_sh == 0 and nind_sh != 0:
                epst = self._parametro(
                    None,
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    None,
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                        + nind_sh
                    ],
                    covariates_sh,
                )  # gamma, gamma_cov
            elif ntrend_sh != 0 and nind_sh == 0:
                epst = self._parametro(
                    None,
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                )  # gamma, gammaT
            else:
                epst = self._parametro(
                    None,
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                    ],
                    x[
                        2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma : 2
                        + self.ngamma0
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                        + ngamma
                        + nind_sh
                    ],
                    covariates_sh,
                )  # gamma, gammaT, gamma_cov

        # The values whose shape parameter is almost 0 correspond to the Gumbel distribution
        posG = np.where(np.abs(epst) <= 1e-8)[0]
        # The remaining values correspond to Weibull or Frechet
        pos = np.where(np.abs(epst) > 1e-8)[0]

        # The corresponding Gumbel values are set to 1 to avoid numerical problems, note that in this case, the Gumbel expressions are used
        epst[posG] = 1

        # Modify the parameters to include the length of the data
        mut = mut1
        psit = psit1
        mut[pos] = mut1[pos] + psit1[pos] * (self.kt[pos] ** epst[pos] - 1) / epst[pos]
        psit[pos] = psit1[pos] * self.kt[pos] ** epst[pos]
        # Modify the parameters to include the length of the data in Gumbel
        mut[posG] = mut[posG] + psit[posG] * np.log(self.kt[posG])

        # Evaluate auxiliarly variables
        xn = (self.xt - mut) / psit
        z = 1 + epst * xn

        # Since z-values must be greater than 0 in order to avoid numerical problems, their values are set to be greater than 1e-8
        z = np.maximum(1e-8, z)
        zn = z ** (-1 / epst)

        # Evaluate the loglikelihood function, not that the general and Gumbel expressions are used
        # f = -np.sum(
        #     -np.log(self.kt[pos])
        #     + np.log(psit[pos])
        #     + (1 + 1 / epst[pos]) * np.log(z[pos])
        #     + self.kt[pos] * zn[pos]
        # ) - np.sum(
        #     -np.log(self.kt[posG])
        #     + np.log(psit[posG])
        #     + xn[posG]
        #     + self.kt[posG] * np.exp(-xn[posG])
        # )

        ### Gradient of the loglikelihood
        # Derivatives given by equations (A.1)-(A.3) in the paper
        Dmut = (1 + epst - self.kt * zn) / (psit * z)
        Dpsit = -(1 - xn * (1 - self.kt * zn)) / (psit * z)
        Depst = (
            zn
            * (
                xn * (self.kt - (1 + epst) / zn)
                + z * (-self.kt + 1 / zn) * np.log(z) / epst
            )
            / (epst * z)
        )

        # Gumbel derivatives given by equations (A.4)-(A.5) in the paper
        Dmut[posG] = (1 - self.kt[posG] * np.exp(-xn[posG])) / psit[posG]
        Dpsit[posG] = (xn[posG] - 1 - self.kt[posG] * xn[posG] * np.exp(-xn[posG])) / (
            psit[posG]
        )
        Depst[posG] = 0

        ## New Derivatives
        Dmutastmut = np.ones_like(self.kt)
        Dmutastpsit = (-1 + self.kt**epst) / epst
        Dmutastepst = (
            psit1 * (1 + (self.kt**epst) * (epst * np.log(self.kt) - 1)) / (epst**2)
        )

        Dpsitastpsit = self.kt**epst
        Dpsitastepst = np.log(self.kt) * psit1 * (self.kt**epst)

        Dmutastpsit[posG] = np.log(self.kt[posG])
        Dmutastepst[posG] = 0

        Dpsitastpsit[posG] = 1
        Dpsitastepst[posG] = 0

        # Set the Jacobian to zero
        Jx = np.zeros(
            2
            + self.ngamma0
            + nmu
            + npsi
            + ngamma
            + ntrend_loc
            + nind_loc
            + ntrend_sc
            + nind_sc
            + ntrend_sh
            + nind_sh
        )
        # Jacobian elements related to the location parameters beta0 and beta, equation (A.6) in the paper
        Jx[0] = np.dot(Dmut, Dmutastmut)

        # If location harmonics are included
        if nmu > 0:
            for i in range(nmu):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dmut[k] * Dmutastmut[k] * self._Dparam(tt, i + 1)
                Jx[i + 1] = aux

        # Jacobian elements related to the location parameters betaT, beta_cov (equation A.9)
        if ntrend_loc > 0:
            Jx[1 + nmu] = np.sum(Dmut * self.t * Dmutastmut)  # betaT
        if nind_loc > 0:
            for i in range(nind_loc):
                Jx[1 + nmu + ntrend_loc + i] = np.sum(
                    Dmut * covariates_loc[:, i] * Dmutastmut
                )  # beta_cov_i

        # Jacobian elements related to the scale parameters alpha0, alpha (equation A.7)
        Jx[1 + nmu + ntrend_loc + nind_loc] = np.sum(
            psit1 * (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit)
        )  # alpha0
        # If scale harmonic are included
        if npsi > 0:
            for i in range(npsi):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += (
                        self._Dparam(tt, i + 1)
                        * psit1[k]
                        * (Dpsit[k] * Dpsitastpsit[k] + Dmut[k] * Dmutastpsit[k])
                    )
                Jx[2 + nmu + ntrend_loc + nind_loc + i] = aux  # alpha
        # Jacobian elements related to the scale parameters alphaT and beta_cov (equation A.10)
        if ntrend_sc > 0:
            Jx[2 + nmu + ntrend_loc + nind_loc + npsi] = np.sum(
                (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit) * self.t * psit1
            )  # alphaT
        if nind_sc > 0:
            for i in range(nind_sc):
                Jx[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i] = np.sum(
                    (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit)
                    * covariates_sc[:, i]
                    * psit1
                )  # alpha_cov

        # Jacobian elements related to the shape parameters gamma0 and gamma (equation A.10)
        if self.ngamma0 == 1:
            Jx[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + nind_sc] = np.sum(
                Depst + Dpsit * Dpsitastepst + Dmut * Dmutastepst
            )
        # If shape harmonics are included
        if ngamma > 0:
            for i in range(ngamma):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += (
                        Depst[k] + Dpsit[k] * Dpsitastepst[k] + Dmut[k] * Dmutastepst[k]
                    ) * self._Dparam(tt, i + 1)
                Jx[
                    2
                    + self.ngamma0
                    + nmu
                    + ntrend_loc
                    + nind_loc
                    + npsi
                    + ntrend_sc
                    + nind_sc
                    + i
                ] = aux

        # Jacobian elements related to the shape parameters trend (defined by Victor)
        if ntrend_sh > 0:
            Jx[
                2
                + self.ngamma0
                + nmu
                + ntrend_loc
                + nind_loc
                + npsi
                + ntrend_sc
                + nind_sc
                + ngamma
            ] = np.sum(Depst * self.t)

        # Jacobian elements related to the shape parameters gamma_cov (defined by Victor)
        if nind_sh > 0:
            for i in range(nind_sh):
                Jx[
                    2
                    + self.ngamma0
                    + nmu
                    + ntrend_loc
                    + nind_loc
                    + npsi
                    + ntrend_sc
                    + nind_sc
                    + ngamma
                    + ntrend_sh
                    + i
                ] = np.sum(Depst * covariates_sh[:, i])

        ### Hessian matrix
        # Derivatives given by equations (A.13)-(A.17) in the paper
        D2mut = (1 + epst) * zn * (-1 + epst * z ** (1 / epst)) / ((z * psit) ** 2)
        D2psit = (
            -zn * xn * ((1 - epst) * xn - 2) + ((1 - 2 * xn) - epst * (xn**2))
        ) / ((z * psit) ** 2)
        D2epst = (
            -zn
            * (
                xn
                * (
                    xn * (1 + 3 * epst)
                    + 2
                    + (-2 - epst * (3 + epst) * xn) * z ** (1 / epst)
                )
                + (z / (epst**2))
                * np.log(z)
                * (
                    2 * epst * (-xn * (1 + epst) - 1 + z ** (1 + 1 / epst))
                    + z * np.log(z)
                )
            )
            / ((epst * z) ** 2)
        )
        Dmutpsit = -(1 + epst - (1 - xn) * zn) / ((z * psit) ** 2)
        Dmutepst = (
            -zn
            * (epst * (-(1 + epst) * xn - epst * (1 - xn) / zn) + z * np.log(z))
            / (psit * epst**2 * z**2)
        )
        Dpsitepst = xn * Dmutepst

        # Corresponding Gumbel derivatives given by equations (A.18)-(A.20)
        D2mut[posG] = -(np.exp(-xn[posG])) / (psit[posG] ** 2)
        D2psit[posG] = (
            (1 - 2 * xn[posG]) + np.exp(-xn[posG]) * (2 - xn[posG]) * xn[posG]
        ) / (psit[posG] ** 2)
        D2epst[posG] = 0
        Dmutpsit[posG] = (-1 + np.exp(-xn[posG]) * (1 - xn[posG])) / (psit[posG] ** 2)
        Dmutepst[posG] = 0
        Dpsitepst[posG] = 0

        # Initialize the Hessian matrix
        Hxx = np.zeros(
            (
                2
                + self.ngamma0
                + nmu
                + npsi
                + ngamma
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc
                + ntrend_sh
                + nind_sh,
                2
                + self.ngamma0
                + nmu
                + npsi
                + ngamma
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc
                + ntrend_sh
                + nind_sh,
            )
        )
        # Elements of the Hessian matrix
        # Sub-blocks following the order shown in Table 4 of the paper

        ## DIAGONAL SUB-BLOCKS
        # Sub-block number 1, beta0^2
        Hxx[0, 0] = np.sum(D2mut)
        # Sub-block number 2, betaT^2
        if ntrend_loc > 0:
            Hxx[1 + nmu, 1 + nmu] = np.sum(D2mut * (self.t**2))
        # Sub-block number 3, beta_cov_i*beta_cov_j
        if nind_loc > 0:
            for i in range(nind_loc):
                for j in range(i + 1):
                    Hxx[1 + nmu + ntrend_loc + i, 1 + nmu + ntrend_loc + j] = np.sum(
                        D2mut * covariates_loc[:, i] * covariates_loc[:, j]
                    )
        # Sub-block number 4, alphaT^2
        if ntrend_sc > 0:
            Hxx[
                2 + nmu + npsi + ntrend_loc + nind_loc,
                2 + nmu + npsi + ntrend_loc + nind_loc,
            ] = np.sum((D2psit * psit + Dpsit) * psit * (self.t**2))
        # Sub-block number 5, alpha_cov_i*alpha_cov_j
        if nind_sc > 0:
            for i in range(nind_sc):
                for j in range(i + 1):
                    Hxx[
                        2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + i,
                        2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + j,
                    ] = np.sum(
                        (D2psit * psit + Dpsit)
                        * psit
                        * covariates_sc[:, i]
                        * covariates_sc[:, j]
                    )
        # Sub-block number 6, alpha0^2
        Hxx[1 + nmu + ntrend_loc + nind_loc, 1 + nmu + ntrend_loc + nind_loc] = np.sum(
            (D2psit * psit + Dpsit) * psit
        )
        # Sub-block number 7, gamma0^2
        if self.ngamma0 == 1:
            # If the shape parameter is added but later the result is GUMBEL
            if len(posG) == len(self.xt):
                Hxx[
                    2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                    2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                ] = -1
            else:
                Hxx[
                    2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                    2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                ] = np.sum(D2epst)
        # Sub-block added by Victor, gamma_cov_i*gamma_cov_j
        if nind_sh > 0:
            for i in range(nind_sh):
                for j in range(i + 1):
                    # Add -1 if the model is GUMBEL in all the values
                    if len(posG) == len(self.xt) and i == j:
                        Hxx[
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ngamma
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + ntrend_sh
                            + i,
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ngamma
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + ntrend_sh
                            + j,
                        ] = -1
                    else:
                        Hxx[
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ngamma
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + ntrend_sh
                            + i,
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ngamma
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + ntrend_sh
                            + j,
                        ] = np.sum(D2epst * covariates_sh[:, i] * covariates_sh[:, j])
        # Sub-block number , gammaT^2
        if ntrend_sh > 0:
            Hxx[
                2
                + self.ngamma0
                + nmu
                + npsi
                + ngamma
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc,
                2
                + self.ngamma0
                + nmu
                + npsi
                + ngamma
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc,
            ] = np.sum(D2epst * self.t**2)

        # Sub-block number 8 (Scale exponential involved), beta0*alpha0
        Hxx[1 + nmu + ntrend_loc + nind_loc, 0] = np.sum(Dmutpsit * psit)

        if self.ngamma0 == 1:
            # Sub-block number 9, beta0*gamma0
            Hxx[2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc, 0] = (
                np.sum(Dmutepst)
            )
            # Sub-block number 10 (Scale exponential involved), alpha0*gamma0
            Hxx[
                2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                1 + nmu + ntrend_loc + nind_loc,
            ] = np.sum(Dpsitepst * psit)
        # Sub-block number 11, beta0*betaT
        if ntrend_loc > 0:
            Hxx[1 + nmu, 0] = np.sum(D2mut * self.t)
        # Sub-block number 12 (Scale exponential involved), beta0*alphaT
        if ntrend_sc > 0:
            Hxx[2 + nmu + ntrend_loc + nind_loc + npsi, 0] = np.sum(
                Dmutpsit * self.t * psit
            )
        # Sub-block number 52 (Scale exponential involved), alphaT*alpha0
        if ntrend_sc > 0:
            Hxx[
                2 + nmu + ntrend_loc + nind_loc + npsi, 1 + nmu + ntrend_loc + nind_loc
            ] = np.sum((D2psit * psit + Dpsit) * self.t * psit)
        # Sub-block number 48 (Scale exponential involved), betaT*alphaT
        if ntrend_loc > 0 and ntrend_sc > 0:
            Hxx[2 + nmu + ntrend_loc + nind_loc + npsi, 1 + nmu] = np.sum(
                Dmutpsit * self.t * self.t * psit
            )
        if ntrend_sh > 0:
            # Sub-block number, beta0*gammaT
            Hxx[
                2
                + self.ngamma0
                + nmu
                + npsi
                + ngamma
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc,
                0,
            ] = np.sum(Dmutepst * self.t)
            # Sub-block number (Scale exponential involved), alpha0*gammaT
            Hxx[
                2
                + self.ngamma0
                + nmu
                + npsi
                + ngamma
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc,
                1 + nmu + ntrend_loc + nind_loc,
            ] = np.sum(Dpsitepst * psit * self.t)
            # Sub-block number, gamma0*gammaT
            Hxx[
                2
                + self.ngamma0
                + nmu
                + npsi
                + ngamma
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc,
                2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
            ] = np.sum(D2epst * self.t)
        if ntrend_sh > 0 and ntrend_sc > 0:
            # Sub-block number, alphaT*gammaT
            Hxx[
                2
                + self.ngamma0
                + nmu
                + npsi
                + ngamma
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc,
                2 + nmu + ntrend_loc + nind_loc + npsi,
            ] = np.sum(Dpsitepst * psit * self.t**2)
        if ntrend_sh > 0 and ntrend_loc > 0:
            # Sub-block number, betaT*gammaT
            Hxx[
                2
                + self.ngamma0
                + nmu
                + npsi
                + ngamma
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc,
                1 + nmu,
            ] = np.sum(Dmutepst * self.t**2)
        # Sub-block number 13, beta0*beta_cov_i
        if nind_loc > 0:
            for i in range(nind_loc):
                Hxx[1 + nmu + ntrend_loc + i, 0] = np.sum(D2mut * covariates_loc[:, i])
        # Sub-block number 14 (Scale exponential involved), beta0*alpha_cov_i
        if nind_sc > 0:
            for i in range(nind_sc):
                Hxx[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i, 0] = np.sum(
                    Dmutpsit * covariates_sc[:, i] * psit
                )
        # Sub-block number 53 (Scale exponential involved), alpha0*alpha_cov_i
        if nind_sc > 0:
            for i in range(nind_sc):
                Hxx[
                    2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i,
                    1 + nmu + ntrend_loc + nind_loc,
                ] = np.sum((D2psit * psit + Dpsit) * covariates_sc[:, i] * psit)
        # Sub-block number 49 (Scale exponential involved), betaT*alpha_cov_i
        if ntrend_loc > 0 and nind_sc > 0:
            for i in range(nind_sc):
                Hxx[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i, 1 + nmu] = (
                    np.sum(Dmutpsit * self.t * covariates_sc[:, i] * psit)
                )
        # Sub-block number 15, betaT*beta_cov_i
        if nind_loc > 0 and ntrend_loc > 0:
            for i in range(nind_loc):
                Hxx[1 + nmu + ntrend_loc + i, 1 + nmu] = np.sum(
                    D2mut * self.t * covariates_loc[:, i]
                )
        # Sub-block number 16, alphaT*alpha_cov_i
        if nind_sc > 0 and ntrend_sc > 0:
            for i in range(nind_sc):
                Hxx[
                    2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i,
                    2 + nmu + ntrend_loc + nind_loc + npsi,
                ] = np.sum(
                    (D2psit * psit + Dpsit) * self.t * covariates_sc[:, i] * psit
                )
        # Sub-block number (Scale exponential involved), alpha_cov_i*gammaT
        if nind_sc > 0 and ntrend_sh > 0:
            for i in range(nind_sc):
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ngamma
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc,
                    2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i,
                ] = np.sum(Dpsitepst * psit * covariates_sc[:, i] * self.t)
        # Sub-block number (Scale exponential involved), beta_cov_i*gammaT
        if nind_loc > 0 and ntrend_sh > 0:
            for i in range(nind_loc):
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ngamma
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc,
                    1 + nmu + ntrend_loc + i,
                ] = np.sum(Dmutepst * covariates_loc[:, i] * self.t)
        # Sub-block number, gamma_cov_i*gammaT
        if nind_sh > 0 and ntrend_sh > 0:
            for i in range(nind_sh):
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ngamma
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc
                    + ntrend_sh
                    + i,
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ngamma
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc,
                ] = np.sum(D2epst * covariates_sh[:, i] * self.t)

        # Sub-block number 17, alpha0*betaT
        if ntrend_loc > 0:
            Hxx[1 + nmu + ntrend_loc + nind_loc, 1 + nmu] = np.sum(
                Dmutpsit * self.t * psit
            )
        # Sub-block number 18, gamma0*betaT
        if ntrend_loc > 0 and self.ngamma0 == 1:
            Hxx[
                2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc, 1 + nmu
            ] = np.sum(Dmutepst * self.t)
        # Sub-block number 19 (Scale exponential involved), gamma0*alphaT
        if ntrend_sc > 0 and self.ngamma0 == 1:
            Hxx[
                2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                2 + nmu + ntrend_loc + nind_loc + npsi,
            ] = np.sum(Dpsitepst * self.t * psit)
        # Sub-block number 20 (Scale exponential involved), alpha0*beta_cov_i
        if nind_loc > 0:
            for i in range(nind_loc):
                Hxx[1 + nmu + ntrend_loc + nind_loc, 1 + nmu + ntrend_loc + i] = np.sum(
                    Dmutpsit * covariates_loc[:, i] * psit
                )
        # Sub-block number 21, gamma0*beta_cov_i
        if nind_loc > 0 and self.ngamma0 == 1:
            for i in range(nind_loc):
                Hxx[
                    2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                    1 + nmu + ntrend_loc + i,
                ] = np.sum(Dmutepst * covariates_loc[:, i])
        # Sub-block number 22 (Scale exponential involved), gamma0*alpha_cov_i
        if nind_sc > 0 and self.ngamma0 == 1:
            for i in range(nind_sc):
                Hxx[
                    2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                    2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i,
                ] = np.sum(Dpsitepst * covariates_sc[:, i] * psit)
        # Sub-block added by Victor, gamma0*gamma_cov_i
        if nind_sh > 0 and self.ngamma0 == 1:
            for i in range(nind_sh):
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ngamma
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc
                    + ntrend_sh
                    + i,
                    2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                ] = np.sum(D2epst * covariates_sh[:, i])

        if nmu > 0:
            for i in range(nmu):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += D2mut[k] * self._Dparam(tt, i + 1)
                # Sub-block number 23, beta_i*beta0
                Hxx[1 + i, 0] = aux
                for j in range(i + 1):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += (
                            D2mut[k] * self._Dparam(tt, i + 1) * self._Dparam(tt, j + 1)
                        )
                    # Sub-block number 24, beta_i,beta_j
                    Hxx[1 + i, 1 + j] = aux

            for i in range(nmu):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dmutpsit[k] * self._Dparam(tt, i + 1) * psit[k]
                # Sub-block number 25 (Scale exponential involved), beta_i*alpha0
                Hxx[1 + nmu + ntrend_loc + nind_loc, 1 + i] = aux

            if self.ngamma0 == 1:
                for i in range(nmu):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dmutepst[k] * self._Dparam(tt, i + 1)
                    # Sub-block number 26 (Scale exponential involved), beta_i*gamma0
                    Hxx[
                        2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                        1 + i,
                    ] = aux
            if ntrend_loc > 0:
                for i in range(nmu):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += D2mut[k] * tt * self._Dparam(tt, i + 1)
                    # Sub-block number 27, betaT*beta_i
                    Hxx[1 + nmu, 1 + i] = aux

            if ntrend_sc > 0:
                for i in range(nmu):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dmutpsit[k] * tt * self._Dparam(tt, i + 1) * psit[k]
                    # Sub-block number 46 (Scale exponential involved), alphaT*beta_i
                    Hxx[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc, 1 + i] = aux
            if ntrend_sh > 0:
                for i in range(nmu):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dmutepst[k] * tt * self._Dparam(tt, i + 1)
                    # Sub-block number 46 (Scale exponential involved), beta_i*gammaT
                    Hxx[
                        2
                        + self.ngamma0
                        + nmu
                        + npsi
                        + ngamma
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc,
                        1 + i,
                    ] = aux

            if nind_loc > 0:
                for i in range(nmu):
                    for j in range(nind_loc):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += (
                                D2mut[k]
                                * covariates_loc[k, j]
                                * self._Dparam(tt, i + 1)
                            )
                        # Sub-block number 28, beta_i*beta_cov_j
                        Hxx[1 + nmu + ntrend_loc + j, 1 + i] = aux
            if nind_sc > 0:
                for i in range(nmu):
                    for j in range(nind_sc):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += (
                                Dmutpsit[k]
                                * covariates_sc[k, j]
                                * self._Dparam(tt, i + 1)
                                * psit[k]
                            )
                        # Sub-block number 47 (Scale exponential involved), beta_i*alpha_cov_j
                        Hxx[
                            2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + j,
                            1 + i,
                        ] = aux
            if nind_sh > 0:
                for i in range(nmu):
                    for j in range(nind_sh):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += (
                                Dmutepst[k]
                                * covariates_sh[k, j]
                                * self._Dparam(tt, i + 1)
                            )
                        # Sub-block added by Victor, beta_j*gamma_cov_i
                        Hxx[
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ngamma
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + ntrend_sh
                            + j,
                            1 + i,
                        ] = aux
        if npsi > 0:
            for i in range(npsi):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += (
                        (D2psit[k] * psit[k] + Dpsit[k])
                        * self._Dparam(tt, i + 1)
                        * psit[k]
                    )
                # Sub-block number 29 (Scale exponential involved), alpha_i*alpha_0
                Hxx[
                    2 + nmu + ntrend_loc + nind_loc + i, 1 + ntrend_loc + nind_loc + nmu
                ] = aux
                for j in range(i + 1):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += (
                            (D2psit[k] * psit[k] + Dpsit[k])
                            * self._Dparam(tt, i + 1)
                            * self._Dparam(tt, j + 1)
                            * psit[k]
                        )
                    # Sub-block 30 (Scale exponential involved), alpha_i*alpha_j
                    Hxx[
                        2 + nmu + ntrend_loc + nind_loc + i,
                        2 + nmu + ntrend_loc + nind_loc + j,
                    ] = aux
            if self.ngamma0 == 1:
                for i in range(npsi):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dpsitepst[k] * self._Dparam(tt, i + 1) * psit[k]
                    # Sub-block number 31 (Scale exponential involved), alpha_i*gamma0
                    Hxx[
                        2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                        2 + nmu + ntrend_loc + nind_loc + i,
                    ] = aux
            for i in range(npsi):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dmutpsit[k] * self._Dparam(tt, i + 1) * psit[k]
                # Sub-block number 32 (Scale exponential involved), beta0*alpha_i
                Hxx[2 + nmu + ntrend_loc + nind_loc + i, 0] = aux
            if ntrend_loc > 0:
                for i in range(npsi):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dmutpsit[k] * tt * self._Dparam(tt, i + 1) * psit[k]
                    # Sub-block number 33 (Scale exponential involved), alpha_i*betaT
                    Hxx[2 + nmu + ntrend_loc + nind_loc + i, 1 + nmu] = aux
            if nind_loc > 0:
                for i in range(npsi):
                    for j in range(nind_loc):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += (
                                Dmutpsit[k]
                                * covariates_loc[k, j]
                                * self._Dparam(tt, i + 1)
                                * psit[k]
                            )
                        # Sub-block number 34 (Scale exponential involved), alpha_i*beta_cov_j
                        Hxx[
                            2 + nmu + ntrend_loc + nind_loc + i,
                            1 + nmu + ntrend_loc + j,
                        ] = aux
            if nind_sh > 0:
                for i in range(npsi):
                    for j in range(nind_sh):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += (
                                Dpsitepst[k]
                                * covariates_sh[k, j]
                                * self._Dparam(tt, i + 1)
                                * psit[k]
                            )
                        # Sub-block added by Victor (scale exponential involved), alpha_i*gamma_cov_j
                        Hxx[
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ngamma
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + ntrend_sh
                            + j,
                            2 + nmu + ntrend_loc + nind_loc + i,
                        ] = aux
            if ntrend_sh > 0:
                for i in range(nmu):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dpsitepst[k] * tt * psit[k] * self._Dparam(tt, i + 1)
                    # Sub-block number 46 (Scale exponential involved), alpha_i*gammaT
                    Hxx[
                        2
                        + self.ngamma0
                        + nmu
                        + npsi
                        + ngamma
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc,
                        2 + nmu + ntrend_loc + nind_loc + i,
                    ] = aux
        if ngamma > 0:
            for i in range(ngamma):
                # First element associated to the constant value (first column)
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += D2epst[k] * self._Dparam(tt, i + 1)
                # Sub-block number 35, gamma_i*gamma0
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc
                    + i,
                    2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                ] = aux
                for j in range(i + 1):
                    # If shape parameters included but later everything is GUMBEL
                    if j == i and len(posG) == len(self.xt):
                        Hxx[
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + i,
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + j,
                        ] = -1
                    else:
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += (
                                D2epst[k]
                                * self._Dparam(tt, i + 1)
                                * self._Dparam(tt, j + 1)
                            )
                        # Sub-block number 36, gamma_i*gamma_j
                        Hxx[
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + i,
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + j,
                        ] = aux
            for i in range(ngamma):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dpsitepst[k] * self._Dparam(tt, i + 1) * psit[k]
                # Sub-block number 37 (Scale exponential involved) gamma_i*alpha0
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc
                    + i,
                    1 + nmu + ntrend_loc + nind_loc,
                ] = aux
            for i in range(ngamma):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dmutepst[k] * self._Dparam(tt, i + 1)
                # Sub-block number 38, gamma_i*beta0
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc
                    + i,
                    0,
                ] = aux
            if ntrend_loc > 0:
                for i in range(ngamma):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dmutepst[k] * tt * self._Dparam(tt, i + 1)
                    # Sub-block number 39, gamma_i*betaT
                    Hxx[
                        2
                        + self.ngamma0
                        + nmu
                        + npsi
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + i,
                        1 + nmu,
                    ] = aux
            if ntrend_sc > 0:
                for i in range(ngamma):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dpsitepst[k] * tt * self._Dparam(tt, i + 1) * psit[k]
                    # Sub-block number 44 (Scale exponential involved), gamma_i*alphaT
                    Hxx[
                        2
                        + self.ngamma0
                        + nmu
                        + npsi
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + i,
                        2 + nmu + npsi + ntrend_loc + nind_loc,
                    ] = aux
            if ntrend_sh > 0:
                for i in range(nmu):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += D2epst[k] * tt * self._Dparam(tt, i + 1)
                    # Sub-block number 46 (Scale exponential involved), gamma_i*gammaT
                    Hxx[
                        2
                        + self.ngamma0
                        + nmu
                        + npsi
                        + ngamma
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc,
                        2
                        + self.ngamma0
                        + nmu
                        + npsi
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + i,
                    ] = aux
            if nind_loc > 0:
                for i in range(ngamma):
                    for j in range(nind_loc):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += (
                                Dmutepst[k]
                                * covariates_loc[k, j]
                                * self._Dparam(tt, i + 1)
                            )
                        # Sub-block number 40, gamma_i*beta_cov_j
                        Hxx[
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + i,
                            1 + nmu + ntrend_loc + j,
                        ] = aux
            if nind_sc > 0:
                for i in range(ngamma):
                    for j in range(nind_sc):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += (
                                Dpsitepst[k]
                                * covariates_sc[k, j]
                                * self._Dparam(tt, i + 1)
                                * psit[k]
                            )
                        # Sub-block number 45 (Scale exponential involved), gamma_i*alpha_cov_j
                        Hxx[
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + i,
                            2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + j,
                        ] = aux
            if nind_sh > 0:
                for i in range(ngamma):
                    for j in range(nind_sh):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += (
                                D2psit[k]
                                * covariates_sh[k, j]
                                * self._Dparam(tt, i + 1)
                            )
                        # Sub-block added by Victor, gamma_i*gamma_cov_j
                        Hxx[
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ngamma
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + ntrend_sh
                            + j,
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + i,
                        ] = aux

        if nind_loc > 0 and ntrend_sc > 0:
            for i in range(nind_loc):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dmutpsit[k] * tt * covariates_loc[k, i] * psit[k]
                # Sub-block number 50 (Scale exponential involved), beta_cov_i*alphaT
                Hxx[
                    2 + nmu + ntrend_loc + nind_loc + npsi, 1 + nmu + ntrend_loc + i
                ] = aux
        if nind_loc > 0 and nind_sc > 0:
            for i in range(nind_loc):
                for j in range(nind_sc):
                    # Sub-block number 51 (Scale exponential involved), beta_cov_i*alpha_cov_j
                    Hxx[
                        2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + j,
                        1 + nmu + ntrend_loc + i,
                    ] = np.sum(
                        Dmutpsit * covariates_sc[:, j] * covariates_loc[:, i] * psit
                    )
        if nind_loc > 0 and nind_sh > 0:
            for i in range(nind_loc):
                for j in range(nind_sh):
                    # Sub-block added by Victor, beta_cov_i*gamma_cov_j
                    Hxx[
                        2
                        + self.ngamma0
                        + nmu
                        + npsi
                        + ngamma
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + ntrend_sh
                        + j,
                        1 + nmu + ntrend_loc + i,
                    ] = np.sum(Dmutepst * covariates_loc[:, i] * covariates_sh[:, j])
        if nind_sc > 0 and nind_sh > 0:
            for i in range(nind_sc):
                for j in range(nind_sh):
                    # Sub-block added by Victor (scale exponential involved), alpha_cov_i*gamma_cov_j
                    Hxx[
                        2
                        + self.ngamma0
                        + nmu
                        + npsi
                        + ngamma
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + ntrend_sh
                        + j,
                        2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + i,
                    ] = np.sum(
                        Dpsitepst * covariates_sc[:, i] * covariates_sh[:, j] * psit
                    )
        if nind_sh > 0 and ntrend_loc > 0:
            for i in range(nind_sh):
                # aux = 0
                # for k, tt in enumerate(self.t):
                #     aux += Dmutepst[k] * tt * covariates_sh[k, i]
                # Sub-block added by Victor, betaT*gamma_cov_i
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ngamma
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc
                    + ntrend_sh
                    + i,
                    1 + nmu,
                ] = np.sum(Dmutepst * self.t * covariates_sh[:, i])
        if ntrend_sc > 0:
            for i in range(npsi):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += (
                        (D2psit[k] * psit[k] + Dpsit[k])
                        * tt
                        * self._Dparam(tt, i + 1)
                        * psit[k]
                    )
                # Sub-block number 54 (Scale exponential involved), alpha_i*alphaT
                Hxx[
                    2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc,
                    2 + nmu + ntrend_loc + nind_loc + i,
                ] = aux
        if nind_sc > 0:
            for i in range(npsi):
                for j in range(nind_sc):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += (
                            (D2psit[k] * psit[k] + Dpsit[k])
                            * covariates_sc[k, j]
                            * self._Dparam(tt, i + 1)
                            * psit[k]
                        )
                    # Sub-block number 55 (Scale exponential involved), alpha_i*alpha_cov_j
                    Hxx[
                        2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + j,
                        2 + nmu + ntrend_loc + nind_loc + i,
                    ] = aux
        if nmu > 0 and npsi > 0:
            for j in range(nmu):
                for i in range(npsi):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += (
                            Dmutpsit[k]
                            * self._Dparam(tt, i + 1)
                            * self._Dparam(tt, j + 1)
                            * psit[k]
                        )
                    # Sub-block number 41 (Scale exponential involved), beta_j*alpha_i
                    Hxx[2 + nmu + ntrend_loc + nind_loc + i, 1 + j] = aux
        if nmu > 0 and ngamma > 0:
            for j in range(nmu):
                for i in range(ngamma):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += (
                            Dmutepst[k]
                            * self._Dparam(tt, i + 1)
                            * self._Dparam(tt, j + 1)
                        )
                    # Sub-block number 42, beta_j*gamma_i
                    Hxx[
                        2
                        + self.ngamma0
                        + nmu
                        + npsi
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + i,
                        1 + j,
                    ] = aux
        if npsi > 0 and ngamma > 0:
            for j in range(npsi):
                for i in range(ngamma):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += (
                            Dpsitepst[k]
                            * self._Dparam(tt, i + 1)
                            * self._Dparam(tt, j + 1)
                            * psit[k]
                        )
                    # Sub-block number 43 (Scale exponential involved), alpha_j*gamma_i
                    Hxx[
                        2
                        + self.ngamma0
                        + nmu
                        + npsi
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + i,
                        2 + nmu + ntrend_loc + nind_loc + j,
                    ] = aux

        if nind_sh > 0:
            for i in range(nind_sh):
                # Sub-block added by Victor, beta0*gamma_cov_i
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ngamma
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc
                    + ntrend_sh
                    + i,
                    0,
                ] = np.sum(Dmutepst * covariates_sh[:, i])
                # Sub-block added by Victor (scale exponential involved), alpha0*gamma_cov_i
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ngamma
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc
                    + ntrend_sh
                    + i,
                    1 + nmu + ntrend_loc + nind_loc,
                ] = np.sum(Dpsitepst * psit * covariates_sh[:, i])

                # aux = 0
                # for k, tt in enumerate(self.t):
                #     aux += Dpsitepst[k] * tt * covariates_sh[k, i] * psit[k]
                # Sub-bloc added by Victor (scale exponential involved), alphaT*gamma_cov_i
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ngamma
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc
                    + ntrend_sh
                    + i,
                    1 + nmu + npsi + ntrend_loc + nind_loc,
                ] = np.sum(Dpsitepst * self.t * covariates_sh[:, i] * psit)

        # Simmetric part of the Hessian
        Hxx = Hxx + np.tril(Hxx, -1).T
        Hxx = -Hxx

        return Hxx

    def fit(
        self,
        nmu: int = 0,
        npsi: int = 0,
        ngamma: int = 0,
        ntrend_loc: int = 1,
        list_loc: Optional[Union[list, str]] = "all",
        ntrend_sc: int = 1,
        list_sc: Optional[Union[list, str]] = "all",
        ntrend_sh: int = 1,
        list_sh: Optional[Union[list, str]] = "all",
        options: dict = None,
        plot: bool = False,
    ) -> dict:
        """
        Function to determine the optimal parameters of Non-Stationary GEV for given covariates, trends and harmonics.

        By default the method fits a Non-Stationary GEV including trends in all the parameters, all possible covariates and no harmonics

        Parameters
        ----------
        nmu : int
            Number of harmonics to be included in the location parameter
        npsi : int
            Number of harmonics to be included in the scale parameter
        ngamma : int
            Number of harmonics to be included in the shape parameter
        ntrend_loc : int, default=1
            If trends in location are included.
        list_loc : list or str, default="all"
            List of indices of covariates to be included in the location parameter.
            If None,no covariates are included in the location parameter.
        ntrend_sc : int, default=1
            If trends in scale are included.
        list_sc : list or str, default="all"
            List of indices of covariates to be included in the scale parameter.
            If None,no covariates are included in the scale parameter.
        ntrend_sh : int, default=1
            If trends in shape are included.
        list_sh : list or str, default="all"
            List of indices of covariates to be included in the shape parameter.
            If None,no covariates are included in the shape parameter.
        plot : bool, default=False
            If True, plot the diagnostic plots

        Returns
        -------
        fit_result : dict
            Dictionary with the optimal parameters and other information about the fit.
            The keys of the dictionary are:
            - beta0, beta, betaT, beta_cov: Location parameters (intercept, harmonic, trend, covariates)
            - alpha0, alpha, alphaT, alpha_cov: Scale parameters (intercept, harmonic, trend, covariates)
            - gamma0, gamma, gammaT, gamma_cov: Shape parameters (intercept, harmonic, trend, covariates)
            - negloglikelihood: Negative log-likelihood value at the optimal solution
            - hessian: Hessian matrix of the log-likelihood function at the optimal solution
            - invI0: Inverse of Fisher information matrix
        """
        self.logger.debug("Fixed fit (.fit())")

        if list_loc == "all":
            list_loc = list(range(self.covariates.shape[1]))
        elif list_loc is None:
            list_loc = []
        if list_sc == "all":
            list_sc = list(range(self.covariates.shape[1]))
        elif list_sc is None:
            list_sc = []
        if list_sh == "all":
            list_sh = list(range(self.covariates.shape[1]))
        elif list_sh is None:
            list_sh = []

        self.nmu = nmu
        self.npsi = npsi
        self.ngamma = ngamma

        self.list_loc = list_loc
        self.nind_loc = len(list_loc)
        self.ntrend_loc = ntrend_loc
        self.list_sc = list_sc
        self.nind_sc = len(list_sc)
        self.ntrend_sc = ntrend_sc
        self.list_sh = list_sh
        self.nind_sh = len(list_sh)
        self.ntrend_sh = ntrend_sh

        fit_result = self._fit(
            nmu=nmu,
            npsi=npsi,
            ngamma=ngamma,
            list_loc=list_loc,
            ntrend_loc=ntrend_loc,
            list_sc=list_sc,
            ntrend_sc=ntrend_sc,
            list_sh=list_sh,
            ntrend_sh=ntrend_sh,
            options=options,
        )

        # Update parameters in the class
        self._update_params(**fit_result)

        # Compute the final loglikelihood and the information matrix
        f, Jx, Hxx = self._loglikelihood(
            beta0=self.beta0,
            beta=self.beta,
            betaT=self.betaT,
            beta_cov=self.beta_cov,
            alpha0=self.alpha0,
            alpha=self.alpha,
            alphaT=self.alphaT,
            alpha_cov=self.alpha_cov,
            gamma0=self.gamma0,
            gamma=self.gamma,
            gammaT=self.gammaT,
            gamma_cov=self.gamma_cov,
            list_loc=self.list_loc,
            list_sc=self.list_sc,
            list_sh=self.list_sh,
        )
        fit_result["loglikelihood"] = f
        fit_result["grad"] = Jx
        fit_result["hessian"] = Hxx

        self.invI0 = np.linalg.inv(-Hxx)
        fit_result["invI0"] = self.invI0

        std_params = np.sqrt(np.diag(self.invI0))
        self.std_params = std_params
        fit_result["std_param"] = std_params

        if plot:
            self.plot()

        return fit_result

    @staticmethod
    def _AIC(loglike, nparam) -> float:
        """
        Compute the AIC for a certain loglikelihood value (loglik) and the number of parameters (np)

        Parameters
        ----------
        loglike : float
            Loglikelihood value
        nparam : int
            Number of parameters in the model

        Returns
        -------
        aic : float
            AIC value
        """
        aic = -2 * loglike + 2 * nparam
        return aic

    def _loglikelihood(
        self,
        beta0: Optional[float] = None,
        beta: Optional[np.ndarray] = None,
        betaT: Optional[float | np.ndarray] = None,
        beta_cov: Optional[np.ndarray] = None,
        alpha0: Optional[float] = None,
        alpha: Optional[np.ndarray] = None,
        alphaT: Optional[float | np.ndarray] = None,
        alpha_cov: Optional[np.ndarray] = None,
        gamma0: Optional[float] = None,
        gamma: Optional[np.ndarray] = None,
        gammaT: Optional[float | np.ndarray] = None,
        gamma_cov: Optional[np.ndarray] = None,
        list_loc: Optional[list] = None,
        list_sc: Optional[list] = None,
        list_sh: Optional[list] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Function to calculate the loglikelihood function, the Jacobian and the Hessian for a given parameterization

        Parameters
        ----------
        beta0 : float, default=None
            Optimal constant parameter related to location
        beta : np.ndarray, default=None
            Optimal harmonic vector associated with location
        betaT : float or np.ndarray, default=None
            Optimal location trend parameter
        beta_cov : np.ndarray, default=None
            Optimal location covariate vector
        alpha0 : float, default=None
            Optimal constant parameter related to scale
        alpha : np.ndarray, default=None
            Optimal harmonic vector associated with scale
        alphaT : float or np.ndarray, default=None
            Optimal scale trend parameter
        alpha_cov : np.ndarray, default=None
            Optimal scale covariate vector
        gamma0 : float, default=None
            Optimal constant parameter related to shape
        gamma : np.ndarray, default=None
            Optimal harmonic vector associated with shape
        gammaT : float or np.ndarray, default=None
            Optimal shape trend parameter
        gamma_cov : np.ndarray, default=None
            Optimal shape covariate vector
        list_loc : list, default=[]
            list of covariates included in the location parameter
        list_sc : list, default=[]
            list of covariates included in the scale parameter
        list_sh : list, default=[]
            list of covariates included in the shape parameter

        Returns
        -------
        f : np.ndarray
            Optimal loglikelihood function
        Jx : np.ndarray
            Gradient of the log-likelihood function at the optimal solution
        Hxx : np.ndarray
            Hessian of the log-likelihood function at the optimal solution
        """

        # Location
        # if beta0 is None:
        #     beta0 = np.empty(0)
        if beta is None:
            #     beta = np.empty(0)
            nmu = 0
        else:
            nmu = beta.size
        if betaT is None or np.asarray(betaT).size == 0:
            # betaT = np.empty(0)
            ntrend_loc = 0
        else:
            ntrend_loc = 1

        # Scale
        # if alpha0 is None:
        #     alpha0 = np.empty(0)
        if alpha is None:
            #     alpha = np.empty(0)
            npsi = 0
        else:
            npsi = alpha.size
        if alphaT is None or np.asarray(alphaT).size == 0:
            # alphaT = np.empty(0)
            ntrend_sc = 0
        else:
            ntrend_sc = 1

        # Shape
        # if gamma0 is None:
        #     gamma0 = np.empty(0)
        if gamma is None:
            #     gamma = np.empty(0)
            ngamma = 0
        else:
            ngamma = gamma.size
        if gammaT is None or np.asarray(gammaT).size == 0:
            # gammaT = np.empty(0)
            ntrend_sh = 0
        else:
            ntrend_sh = 1

        if beta_cov is None:
            beta_cov = np.empty(0)
        if alpha_cov is None:
            alpha_cov = np.empty(0)
        if gamma_cov is None:
            gamma_cov = np.empty(0)

        if list_loc is None:
            list_loc = []
        if list_sc is None:
            list_sc = []
        if list_sh is None:
            list_sh = []

        covariates_loc = self.covariates.iloc[:, list_loc].values
        covariates_sc = self.covariates.iloc[:, list_sc].values
        covariates_sh = self.covariates.iloc[:, list_sh].values

        na1, nind_loc = np.asarray(covariates_loc).shape
        if nind_loc > 0 and na1 > 0:
            if na1 != len(self.xt) or na1 != len(self.t) or len(self.xt) != len(self.t):
                ValueError("Check data x, t, indices: funcion loglikelihood")

        na2, nind_sc = np.asarray(covariates_sc).shape
        if nind_sc > 0 and na2 > 0:
            if na2 != len(self.xt) or na2 != len(self.t) or len(self.xt) != len(self.t):
                ValueError("Check data x, t, indices: funcion loglikelihood")

        na3, nind_sh = np.asarray(covariates_sh).shape
        if nind_sc > 0 and na3 > 0:
            if na3 != len(self.xt) or na3 != len(self.t) or len(self.xt) != len(self.t):
                ValueError("Check data x, t, indices: funcion loglikelihood")

        nind_loc = beta_cov.size
        nind_sc = alpha_cov.size
        nind_sh = gamma_cov.size

        # Evaluate the parameters
        mut1, psit1, epst = self._evaluate_params(
            beta0=beta0,
            beta=beta,
            betaT=betaT,
            beta_cov=beta_cov,
            alpha0=alpha0,
            alpha=alpha,
            alphaT=alphaT,
            alpha_cov=alpha_cov,
            gamma0=gamma0,
            gamma=gamma,
            gammaT=gammaT,
            gamma_cov=gamma_cov,
            covariates_loc=covariates_loc,
            covariates_sc=covariates_sc,
            covariates_sh=covariates_sh,
        )

        # The values whose shape parameter is almost 0 correspond to the Gumbel distribution
        posG = np.where(np.abs(epst) <= 1e-8)[0]
        # The remaining values correspond to Weibull or Frechet
        pos = np.where(np.abs(epst) > 1e-8)[0]

        # The corresponding Gumbel values are set to 1 to avoid numerical problems, note that in this case, the Gumbel expressions are used
        epst[posG] = 1

        # Modify the parameters to include the length of the data
        mut = mut1
        psit = psit1
        mut[pos] = mut1[pos] + psit1[pos] * (self.kt[pos] ** epst[pos] - 1) / epst[pos]
        psit[pos] = psit1[pos] * self.kt[pos] ** epst[pos]
        # Modify the parameters to include the length of the data in Gumbel
        mut[posG] = mut[posG] + psit[posG] * np.log(self.kt[posG])

        # Evaluate auxiliarly variables
        xn = (self.xt - mut) / psit
        z = 1 + epst * xn

        # Since z-values must be greater than 0 in order to avoid numerical problems, their values are set to be greater than 1e-8
        z = np.maximum(1e-8, z)
        zn = z ** (-1 / epst)

        # Evaluate the loglikelihood function, not that the general and Gumbel expressions are used
        f = -np.sum(
            -np.log(self.kt[pos])
            + np.log(psit[pos])
            + (1 + 1 / epst[pos]) * np.log(z[pos])
            + self.kt[pos] * zn[pos]
        ) - np.sum(
            -np.log(self.kt[posG])
            + np.log(psit[posG])
            + xn[posG]
            + self.kt[posG] * np.exp(-xn[posG])
        )

        ### Gradient of the loglikelihood
        # Derivatives given by equations (A.1)-(A.3) in the paper
        Dmut = (1 + epst - self.kt * zn) / (psit * z)
        Dpsit = -(1 - xn * (1 - self.kt * zn)) / (psit * z)
        Depst = (
            zn
            * (
                xn * (self.kt - (1 + epst) / zn)
                + z * (-self.kt + 1 / zn) * np.log(z) / epst
            )
            / (epst * z)
        )

        # Gumbel derivatives given by equations (A.4)-(A.5) in the paper
        Dmut[posG] = (1 - self.kt[posG] * np.exp(-xn[posG])) / psit[posG]
        Dpsit[posG] = (xn[posG] - 1 - self.kt[posG] * xn[posG] * np.exp(-xn[posG])) / (
            psit[posG]
        )
        Depst[posG] = 0

        ## New Derivatives
        Dmutastmut = np.ones_like(self.kt)
        Dmutastpsit = (-1 + self.kt**epst) / epst
        Dmutastepst = (
            psit1 * (1 + (self.kt**epst) * (epst * np.log(self.kt) - 1)) / (epst**2)
        )

        Dpsitastpsit = self.kt**epst
        Dpsitastepst = np.log(self.kt) * psit1 * (self.kt**epst)

        Dmutastpsit[posG] = np.log(self.kt[posG])
        Dmutastepst[posG] = 0

        Dpsitastpsit[posG] = 1
        Dpsitastepst[posG] = 0

        # Set the Jacobian to zero
        Jx = np.zeros(
            2
            + self.ngamma0
            + nmu
            + npsi
            + ngamma
            + ntrend_loc
            + nind_loc
            + ntrend_sc
            + nind_sc
            + ntrend_sh
            + nind_sh
        )
        # Jacobian elements related to the location parameters beta0 and beta, equation (A.6) in the paper
        Jx[0] = np.dot(Dmut, Dmutastmut)

        # If location harmonics are included
        if nmu > 0:
            for i in range(nmu):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dmut[k] * Dmutastmut[k] * self._Dparam(tt, i + 1)
                Jx[i + 1] = aux

        # Jacobian elements related to the location parameters betaT, beta_cov (equation A.9)
        if ntrend_loc > 0:
            Jx[1 + nmu] = np.sum(Dmut * self.t * Dmutastmut)  # betaT
        if nind_loc > 0:
            for i in range(nind_loc):
                Jx[1 + nmu + ntrend_loc + i] = np.sum(
                    Dmut * covariates_loc[:, i] * Dmutastmut
                )  # beta_cov_i

        # Jacobian elements related to the scale parameters alpha0, alpha (equation A.7)
        Jx[1 + nmu + ntrend_loc + nind_loc] = np.sum(
            psit1 * (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit)
        )  # alpha0
        # If scale harmonic are included
        if npsi > 0:
            for i in range(npsi):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += (
                        self._Dparam(tt, i + 1)
                        * psit1[k]
                        * (Dpsit[k] * Dpsitastpsit[k] + Dmut[k] * Dmutastpsit[k])
                    )
                Jx[2 + nmu + ntrend_loc + nind_loc + i] = aux  # alpha
        # Jacobian elements related to the scale parameters alphaT and beta_cov (equation A.10)
        if ntrend_sc > 0:
            Jx[2 + nmu + ntrend_loc + nind_loc + npsi] = np.sum(
                (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit) * self.t * psit1
            )  # alphaT
        if nind_sc > 0:
            for i in range(nind_sc):
                Jx[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i] = np.sum(
                    (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit)
                    * covariates_sc[:, i]
                    * psit1
                )  # alpha_cov

        # Jacobian elements related to the shape parameters gamma0 and gamma (equation A.10)
        if self.ngamma0 == 1:
            Jx[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + nind_sc] = np.sum(
                Depst + Dpsit * Dpsitastepst + Dmut * Dmutastepst
            )
        # If shape harmonics are included
        if ngamma > 0:
            for i in range(ngamma):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += (
                        Depst[k] + Dpsit[k] * Dpsitastepst[k] + Dmut[k] * Dmutastepst[k]
                    ) * self._Dparam(tt, i + 1)
                Jx[
                    2
                    + self.ngamma0
                    + nmu
                    + ntrend_loc
                    + nind_loc
                    + npsi
                    + ntrend_sc
                    + nind_sc
                    + i
                ] = aux

        # Jacobian elements related to the shape parameters trend (defined by Victor)
        if ntrend_sh > 0:
            Jx[
                2
                + self.ngamma0
                + nmu
                + ntrend_loc
                + nind_loc
                + npsi
                + ntrend_sc
                + nind_sc
                + ngamma
            ] = np.sum(Depst * self.t)

        # Jacobian elements related to the shape parameters gamma_cov (defined by Victor)
        if nind_sh > 0:
            for i in range(nind_sh):
                Jx[
                    2
                    + self.ngamma0
                    + nmu
                    + ntrend_loc
                    + nind_loc
                    + npsi
                    + ntrend_sc
                    + nind_sc
                    + ngamma
                    + ntrend_sh
                    + i
                ] = np.sum(Depst * covariates_sh[:, i])

        ### Hessian matrix
        # Derivatives given by equations (A.13)-(A.17) in the paper
        D2mut = (1 + epst) * zn * (-1 + epst * z ** (1 / epst)) / ((z * psit) ** 2)
        D2psit = (
            -zn * xn * ((1 - epst) * xn - 2) + ((1 - 2 * xn) - epst * (xn**2))
        ) / ((z * psit) ** 2)
        D2epst = (
            -zn
            * (
                xn
                * (
                    xn * (1 + 3 * epst)
                    + 2
                    + (-2 - epst * (3 + epst) * xn) * z ** (1 / epst)
                )
                + (z / (epst**2))
                * np.log(z)
                * (
                    2 * epst * (-xn * (1 + epst) - 1 + z ** (1 + 1 / epst))
                    + z * np.log(z)
                )
            )
            / ((epst * z) ** 2)
        )
        Dmutpsit = -(1 + epst - (1 - xn) * zn) / ((z * psit) ** 2)
        Dmutepst = (
            -zn
            * (epst * (-(1 + epst) * xn - epst * (1 - xn) / zn) + z * np.log(z))
            / (psit * epst**2 * z**2)
        )
        Dpsitepst = xn * Dmutepst

        # Corresponding Gumbel derivatives given by equations (A.18)-(A.20)
        D2mut[posG] = -(np.exp(-xn[posG])) / (psit[posG] ** 2)
        D2psit[posG] = (
            (1 - 2 * xn[posG]) + np.exp(-xn[posG]) * (2 - xn[posG]) * xn[posG]
        ) / (psit[posG] ** 2)
        D2epst[posG] = 0
        Dmutpsit[posG] = (-1 + np.exp(-xn[posG]) * (1 - xn[posG])) / (psit[posG] ** 2)
        Dmutepst[posG] = 0
        Dpsitepst[posG] = 0

        # Initialize the Hessian matrix
        Hxx = np.zeros(
            (
                2
                + self.ngamma0
                + nmu
                + npsi
                + ngamma
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc
                + ntrend_sh
                + nind_sh,
                2
                + self.ngamma0
                + nmu
                + npsi
                + ngamma
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc
                + ntrend_sh
                + nind_sh,
            )
        )
        # Elements of the Hessian matrix
        # Sub-blocks following the order shown in Table 4 of the paper

        ## DIAGONAL SUB-BLOCKS
        # Sub-block number 1, beta0^2
        Hxx[0, 0] = np.sum(D2mut)
        # Sub-block number 2, betaT^2
        if ntrend_loc > 0:
            Hxx[1 + nmu, 1 + nmu] = np.sum(D2mut * (self.t**2))
        # Sub-block number 3, beta_cov_i*beta_cov_j
        if nind_loc > 0:
            for i in range(nind_loc):
                for j in range(i + 1):
                    Hxx[1 + nmu + ntrend_loc + i, 1 + nmu + ntrend_loc + j] = np.sum(
                        D2mut * covariates_loc[:, i] * covariates_loc[:, j]
                    )
        # Sub-block number 4, alphaT^2
        if ntrend_sc > 0:
            Hxx[
                2 + nmu + npsi + ntrend_loc + nind_loc,
                2 + nmu + npsi + ntrend_loc + nind_loc,
            ] = np.sum((D2psit * psit + Dpsit) * psit * (self.t**2))
        # Sub-block number 5, alpha_cov_i*alpha_cov_j
        if nind_sc > 0:
            for i in range(nind_sc):
                for j in range(i + 1):
                    Hxx[
                        2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + i,
                        2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + j,
                    ] = np.sum(
                        (D2psit * psit + Dpsit)
                        * psit
                        * covariates_sc[:, i]
                        * covariates_sc[:, j]
                    )
        # Sub-block number 6, alpha0^2
        Hxx[1 + nmu + ntrend_loc + nind_loc, 1 + nmu + ntrend_loc + nind_loc] = np.sum(
            (D2psit * psit + Dpsit) * psit
        )
        # Sub-block number 7, gamma0^2
        if self.ngamma0 == 1:
            # If the shape parameter is added but later the result is GUMBEL
            if len(posG) == len(self.xt):
                Hxx[
                    2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                    2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                ] = -1
            else:
                Hxx[
                    2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                    2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                ] = np.sum(D2epst)
        # Sub-block added by Victor, gamma_cov_i*gamma_cov_j
        if nind_sh > 0:
            for i in range(nind_sh):
                for j in range(i + 1):
                    # Add -1 if the model is GUMBEL in all the values
                    if len(posG) == len(self.xt) and i == j:
                        Hxx[
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ngamma
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + ntrend_sh
                            + i,
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ngamma
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + ntrend_sh
                            + j,
                        ] = -1
                    else:
                        Hxx[
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ngamma
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + ntrend_sh
                            + i,
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ngamma
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + ntrend_sh
                            + j,
                        ] = np.sum(D2epst * covariates_sh[:, i] * covariates_sh[:, j])
        # Sub-block number , gammaT^2
        if ntrend_sh > 0:
            Hxx[
                2
                + self.ngamma0
                + nmu
                + npsi
                + ngamma
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc,
                2
                + self.ngamma0
                + nmu
                + npsi
                + ngamma
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc,
            ] = np.sum(D2epst * self.t**2)

        # Sub-block number 8 (Scale exponential involved), beta0*alpha0
        Hxx[1 + nmu + ntrend_loc + nind_loc, 0] = np.sum(Dmutpsit * psit)

        if self.ngamma0 == 1:
            # Sub-block number 9, beta0*gamma0
            Hxx[2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc, 0] = (
                np.sum(Dmutepst)
            )
            # Sub-block number 10 (Scale exponential involved), alpha0*gamma0
            Hxx[
                2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                1 + nmu + ntrend_loc + nind_loc,
            ] = np.sum(Dpsitepst * psit)
        # Sub-block number 11, beta0*betaT
        if ntrend_loc > 0:
            Hxx[1 + nmu, 0] = np.sum(D2mut * self.t)
        # Sub-block number 12 (Scale exponential involved), beta0*alphaT
        if ntrend_sc > 0:
            Hxx[2 + nmu + ntrend_loc + nind_loc + npsi, 0] = np.sum(
                Dmutpsit * self.t * psit
            )
        # Sub-block number 52 (Scale exponential involved), alphaT*alpha0
        if ntrend_sc > 0:
            Hxx[
                2 + nmu + ntrend_loc + nind_loc + npsi, 1 + nmu + ntrend_loc + nind_loc
            ] = np.sum((D2psit * psit + Dpsit) * self.t * psit)
        # Sub-block number 48 (Scale exponential involved), betaT*alphaT
        if ntrend_loc > 0 and ntrend_sc > 0:
            Hxx[2 + nmu + ntrend_loc + nind_loc + npsi, 1 + nmu] = np.sum(
                Dmutpsit * self.t * self.t * psit
            )
        if ntrend_sh > 0:
            # Sub-block number, beta0*gammaT
            Hxx[
                2
                + self.ngamma0
                + nmu
                + npsi
                + ngamma
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc,
                0,
            ] = np.sum(Dmutepst * self.t)
            # Sub-block number (Scale exponential involved), alpha0*gammaT
            Hxx[
                2
                + self.ngamma0
                + nmu
                + npsi
                + ngamma
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc,
                1 + nmu + ntrend_loc + nind_loc,
            ] = np.sum(Dpsitepst * psit * self.t)
            # Sub-block number, gamma0*gammaT
            Hxx[
                2
                + self.ngamma0
                + nmu
                + npsi
                + ngamma
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc,
                2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
            ] = np.sum(D2epst * self.t)
        if ntrend_sh > 0 and ntrend_loc > 0:
            # Sub-block number, betaT*gammaT
            Hxx[
                2
                + self.ngamma0
                + nmu
                + npsi
                + ngamma
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc,
                2 + nmu + ntrend_loc + nind_loc + npsi,
            ] = np.sum(Dmutepst * self.t**2)
        if ntrend_sh > 0 and ntrend_sc > 0:
            # Sub-block number, alphaT*gammaT
            Hxx[
                2
                + self.ngamma0
                + nmu
                + npsi
                + ngamma
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc,
                1 + nmu,
            ] = np.sum(Dpsitepst * self.t**2 * psit)
        # Sub-block number 13, beta0*beta_cov_i
        if nind_loc > 0:
            for i in range(nind_loc):
                Hxx[1 + nmu + ntrend_loc + i, 0] = np.sum(D2mut * covariates_loc[:, i])
        # Sub-block number 14 (Scale exponential involved), beta0*alpha_cov_i
        if nind_sc > 0:
            for i in range(nind_sc):
                Hxx[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i, 0] = np.sum(
                    Dmutpsit * covariates_sc[:, i] * psit
                )
        # Sub-block number 53 (Scale exponential involved), alpha0*alpha_cov_i
        if nind_sc > 0:
            for i in range(nind_sc):
                Hxx[
                    2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i,
                    1 + nmu + ntrend_loc + nind_loc,
                ] = np.sum((D2psit * psit + Dpsit) * covariates_sc[:, i] * psit)
        # Sub-block number 49 (Scale exponential involved), betaT*alpha_cov_i
        if ntrend_loc > 0 and nind_sc > 0:
            for i in range(nind_sc):
                Hxx[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i, 1 + nmu] = (
                    np.sum(Dmutpsit * self.t * covariates_sc[:, i] * psit)
                )
        # Sub-block number 15, betaT*beta_cov_i
        if nind_loc > 0 and ntrend_loc > 0:
            for i in range(nind_loc):
                Hxx[1 + nmu + ntrend_loc + i, 1 + nmu] = np.sum(
                    D2mut * self.t * covariates_loc[:, i]
                )
        # Sub-block number 16, alphaT*alpha_cov_i
        if nind_sc > 0 and ntrend_sc > 0:
            for i in range(nind_sc):
                Hxx[
                    2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i,
                    2 + nmu + ntrend_loc + nind_loc + npsi,
                ] = np.sum(
                    (D2psit * psit + Dpsit) * self.t * covariates_sc[:, i] * psit
                )
        # Sub-block number (Scale exponential involved), alpha_cov_i*gammaT
        if nind_sc > 0 and ntrend_sh > 0:
            for i in range(nind_sc):
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ngamma
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc,
                    2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i,
                ] = np.sum(Dpsitepst * psit * covariates_sc[:, i] * self.t)
        # Sub-block number (Scale exponential involved), beta_cov_i*gammaT
        if nind_loc > 0 and ntrend_sh > 0:
            for i in range(nind_loc):
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ngamma
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc,
                    1 + nmu + ntrend_loc + i,
                ] = np.sum(Dmutepst * covariates_loc[:, i] * self.t)
        # Sub-block number, gamma_cov_i*gammaT
        if nind_sh > 0 and ntrend_sh > 0:
            for i in range(nind_sh):
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ngamma
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc
                    + ntrend_sh
                    + i,
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ngamma
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc,
                ] = np.sum(D2epst * covariates_sh[:, i] * self.t)

        # Sub-block number 17, alpha0*betaT
        if ntrend_loc > 0:
            Hxx[1 + nmu + ntrend_loc + nind_loc, 1 + nmu] = np.sum(
                Dmutpsit * self.t * psit
            )
        # Sub-block number 18, gamma0*betaT
        if ntrend_loc > 0 and self.ngamma0 == 1:
            Hxx[
                2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc, 1 + nmu
            ] = np.sum(Dmutepst * self.t)
        # Sub-block number 19 (Scale exponential involved), gamma0*alphaT
        if ntrend_sc > 0 and self.ngamma0 == 1:
            Hxx[
                2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                2 + nmu + ntrend_loc + nind_loc + npsi,
            ] = np.sum(Dpsitepst * self.t * psit)
        # Sub-block number 20 (Scale exponential involved), alpha0*beta_cov_i
        if nind_loc > 0:
            for i in range(nind_loc):
                Hxx[1 + nmu + ntrend_loc + nind_loc, 1 + nmu + ntrend_loc + i] = np.sum(
                    Dmutpsit * covariates_loc[:, i] * psit
                )
        # Sub-block number 21, gamma0*beta_cov_i
        if nind_loc > 0 and self.ngamma0 == 1:
            for i in range(nind_loc):
                Hxx[
                    2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                    1 + nmu + ntrend_loc + i,
                ] = np.sum(Dmutepst * covariates_loc[:, i])
        # Sub-block number 22 (Scale exponential involved), gamma0*alpha_cov_i
        if nind_sc > 0 and self.ngamma0 == 1:
            for i in range(nind_sc):
                Hxx[
                    2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                    2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i,
                ] = np.sum(Dpsitepst * covariates_sc[:, i] * psit)
        # Sub-block added by Victor, gamma0*gamma_cov_i
        if nind_sh > 0 and self.ngamma0 == 1:
            for i in range(nind_sh):
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ngamma
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc
                    + ntrend_sh
                    + i,
                    2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                ] = np.sum(D2epst * covariates_sh[:, i])

        if nmu > 0:
            for i in range(nmu):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += D2mut[k] * self._Dparam(tt, i + 1)
                # Sub-block number 23, beta_i*beta0
                Hxx[1 + i, 0] = aux
                for j in range(i + 1):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += (
                            D2mut[k] * self._Dparam(tt, i + 1) * self._Dparam(tt, j + 1)
                        )
                    # Sub-block number 24, beta_i,beta_j
                    Hxx[1 + i, 1 + j] = aux

            for i in range(nmu):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dmutpsit[k] * self._Dparam(tt, i + 1) * psit[k]
                # Sub-block number 25 (Scale exponential involved), beta_i*alpha0
                Hxx[1 + nmu + ntrend_loc + nind_loc, 1 + i] = aux

            if self.ngamma0 == 1:
                for i in range(nmu):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dmutepst[k] * self._Dparam(tt, i + 1)
                    # Sub-block number 26 (Scale exponential involved), beta_i*gamma0
                    Hxx[
                        2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                        1 + i,
                    ] = aux
            if ntrend_loc > 0:
                for i in range(nmu):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += D2mut[k] * tt * self._Dparam(tt, i + 1)
                    # Sub-block number 27, betaT*beta_i
                    Hxx[1 + nmu, 1 + i] = aux

            if ntrend_sc > 0:
                for i in range(nmu):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dmutpsit[k] * tt * self._Dparam(tt, i + 1) * psit[k]
                    # Sub-block number 46 (Scale exponential involved), alphaT*beta_i
                    Hxx[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc, 1 + i] = aux
            if ntrend_sh > 0:
                for i in range(nmu):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dmutepst[k] * tt * self._Dparam(tt, i + 1)
                    # Sub-block number 46 (Scale exponential involved), beta_i*gammaT
                    Hxx[
                        2
                        + self.ngamma0
                        + nmu
                        + npsi
                        + ngamma
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc,
                        1 + i,
                    ] = aux

            if nind_loc > 0:
                for i in range(nmu):
                    for j in range(nind_loc):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += (
                                D2mut[k]
                                * covariates_loc[k, j]
                                * self._Dparam(tt, i + 1)
                            )
                        # Sub-block number 28, beta_i*beta_cov_j
                        Hxx[1 + nmu + ntrend_loc + j, 1 + i] = aux
            if nind_sc > 0:
                for i in range(nmu):
                    for j in range(nind_sc):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += (
                                Dmutpsit[k]
                                * covariates_sc[k, j]
                                * self._Dparam(tt, i + 1)
                                * psit[k]
                            )
                        # Sub-block number 47 (Scale exponential involved), beta_i*alpha_cov_j
                        Hxx[
                            2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + j,
                            1 + i,
                        ] = aux
            if nind_sh > 0:
                for i in range(nmu):
                    for j in range(nind_sh):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += (
                                Dmutepst[k]
                                * covariates_sh[k, j]
                                * self._Dparam(tt, i + 1)
                            )
                        # Sub-block added by Victor, beta_j*gamma_cov_i
                        Hxx[
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ngamma
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + ntrend_sh
                            + j,
                            1 + i,
                        ] = aux
        if npsi > 0:
            for i in range(npsi):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += (
                        (D2psit[k] * psit[k] + Dpsit[k])
                        * self._Dparam(tt, i + 1)
                        * psit[k]
                    )
                # Sub-block number 29 (Scale exponential involved), alpha_i*alpha_0
                Hxx[
                    2 + nmu + ntrend_loc + nind_loc + i, 1 + ntrend_loc + nind_loc + nmu
                ] = aux
                for j in range(i + 1):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += (
                            (D2psit[k] * psit[k] + Dpsit[k])
                            * self._Dparam(tt, i + 1)
                            * self._Dparam(tt, j + 1)
                            * psit[k]
                        )
                    # Sub-block 30 (Scale exponential involved), alpha_i*alpha_j
                    Hxx[
                        2 + nmu + ntrend_loc + nind_loc + i,
                        2 + nmu + ntrend_loc + nind_loc + j,
                    ] = aux
            if self.ngamma0 == 1:
                for i in range(npsi):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dpsitepst[k] * self._Dparam(tt, i + 1) * psit[k]
                    # Sub-block number 31 (Scale exponential involved), alpha_i*gamma0
                    Hxx[
                        2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                        2 + nmu + ntrend_loc + nind_loc + i,
                    ] = aux
            for i in range(npsi):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dmutpsit[k] * self._Dparam(tt, i + 1) * psit[k]
                # Sub-block number 32 (Scale exponential involved), beta0*alpha_i
                Hxx[2 + nmu + ntrend_loc + nind_loc + i, 0] = aux
            if ntrend_loc > 0:
                for i in range(npsi):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dmutpsit[k] * tt * self._Dparam(tt, i + 1) * psit[k]
                    # Sub-block number 33 (Scale exponential involved), alpha_i*betaT
                    Hxx[2 + nmu + ntrend_loc + nind_loc + i, 1 + nmu] = aux
            if nind_loc > 0:
                for i in range(npsi):
                    for j in range(nind_loc):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += (
                                Dmutpsit[k]
                                * covariates_loc[k, j]
                                * self._Dparam(tt, i + 1)
                                * psit[k]
                            )
                        # Sub-block number 34 (Scale exponential involved), alpha_i*beta_cov_j
                        Hxx[
                            2 + nmu + ntrend_loc + nind_loc + i,
                            1 + nmu + ntrend_loc + j,
                        ] = aux
            if nind_sh > 0:
                for i in range(npsi):
                    for j in range(nind_sh):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += (
                                Dpsitepst[k]
                                * covariates_sh[k, j]
                                * self._Dparam(tt, i + 1)
                                * psit[k]
                            )
                        # Sub-block added by Victor (scale exponential involved), alpha_i*gamma_cov_j
                        Hxx[
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ngamma
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + ntrend_sh
                            + j,
                            2 + nmu + ntrend_loc + nind_loc + i,
                        ] = aux
            if ntrend_sh > 0:
                for i in range(nmu):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dpsitepst[k] * tt * psit[k] * self._Dparam(tt, i + 1)
                    # Sub-block number 46 (Scale exponential involved), alpha_i*gammaT
                    Hxx[
                        2
                        + self.ngamma0
                        + nmu
                        + npsi
                        + ngamma
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc,
                        2 + nmu + ntrend_loc + nind_loc + i,
                    ] = aux
        if ngamma > 0:
            for i in range(ngamma):
                # First element associated to the constant value (first column)
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += D2epst[k] * self._Dparam(tt, i + 1)
                # Sub-block number 35, gamma_i*gamma0
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc
                    + i,
                    2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                ] = aux
                for j in range(i + 1):
                    # If shape parameters included but later everything is GUMBEL
                    if j == i and len(posG) == len(self.xt):
                        Hxx[
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + i,
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + j,
                        ] = -1
                    else:
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += (
                                D2epst[k]
                                * self._Dparam(tt, i + 1)
                                * self._Dparam(tt, j + 1)
                            )
                        # Sub-block number 36, gamma_i*gamma_j
                        Hxx[
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + i,
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + j,
                        ] = aux
            for i in range(ngamma):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dpsitepst[k] * self._Dparam(tt, i + 1) * psit[k]
                # Sub-block number 37 (Scale exponential involved) gamma_i*alpha0
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc
                    + i,
                    1 + nmu + ntrend_loc + nind_loc,
                ] = aux
            for i in range(ngamma):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dmutepst[k] * self._Dparam(tt, i + 1)
                # Sub-block number 38, gamma_i*beta0
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc
                    + i,
                    0,
                ] = aux
            if ntrend_loc > 0:
                for i in range(ngamma):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dmutepst[k] * tt * self._Dparam(tt, i + 1)
                    # Sub-block number 39, gamma_i*betaT
                    Hxx[
                        2
                        + self.ngamma0
                        + nmu
                        + npsi
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + i,
                        1 + nmu,
                    ] = aux
            if ntrend_sc > 0:
                for i in range(ngamma):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dpsitepst[k] * tt * self._Dparam(tt, i + 1) * psit[k]
                    # Sub-block number 44 (Scale exponential involved), gamma_i*alphaT
                    Hxx[
                        2
                        + self.ngamma0
                        + nmu
                        + npsi
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + i,
                        2 + nmu + npsi + ntrend_loc + nind_loc,
                    ] = aux
            if ntrend_sh > 0:
                for i in range(nmu):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += D2epst[k] * tt * self._Dparam(tt, i + 1)
                    # Sub-block number 46 (Scale exponential involved), gamma_i*gammaT
                    Hxx[
                        2
                        + self.ngamma0
                        + nmu
                        + npsi
                        + ngamma
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc,
                        2
                        + self.ngamma0
                        + nmu
                        + npsi
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + i,
                    ] = aux
            if nind_loc > 0:
                for i in range(ngamma):
                    for j in range(nind_loc):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += (
                                Dmutepst[k]
                                * covariates_loc[k, j]
                                * self._Dparam(tt, i + 1)
                            )
                        # Sub-block number 40, gamma_i*beta_cov_j
                        Hxx[
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + i,
                            1 + nmu + ntrend_loc + j,
                        ] = aux
            if nind_sc > 0:
                for i in range(ngamma):
                    for j in range(nind_sc):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += (
                                Dpsitepst[k]
                                * covariates_sc[k, j]
                                * self._Dparam(tt, i + 1)
                                * psit[k]
                            )
                        # Sub-block number 45 (Scale exponential involved), gamma_i*alpha_cov_j
                        Hxx[
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + i,
                            2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + j,
                        ] = aux
            if nind_sh > 0:
                for i in range(ngamma):
                    for j in range(nind_sh):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += (
                                D2psit[k]
                                * covariates_sh[k, j]
                                * self._Dparam(tt, i + 1)
                            )
                        # Sub-block added by Victor, gamma_i*gamma_cov_j
                        Hxx[
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ngamma
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + ntrend_sh
                            + j,
                            2
                            + self.ngamma0
                            + nmu
                            + npsi
                            + ntrend_loc
                            + nind_loc
                            + ntrend_sc
                            + nind_sc
                            + i,
                        ] = aux

        if nind_loc > 0 and ntrend_sc > 0:
            for i in range(nind_loc):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dmutpsit[k] * tt * covariates_loc[k, i] * psit[k]
                # Sub-block number 50 (Scale exponential involved), beta_cov_i*alphaT
                Hxx[
                    2 + nmu + ntrend_loc + nind_loc + npsi, 1 + nmu + ntrend_loc + i
                ] = aux
        if nind_loc > 0 and nind_sc > 0:
            for i in range(nind_loc):
                for j in range(nind_sc):
                    # Sub-block number 51 (Scale exponential involved), beta_cov_i*alpha_cov_j
                    Hxx[
                        2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + j,
                        1 + nmu + ntrend_loc + i,
                    ] = np.sum(
                        Dmutpsit * covariates_sc[:, j] * covariates_loc[:, i] * psit
                    )
        if nind_loc > 0 and nind_sh > 0:
            for i in range(nind_loc):
                for j in range(nind_sh):
                    # Sub-block added by Victor, beta_cov_i*gamma_cov_j
                    Hxx[
                        2
                        + self.ngamma0
                        + nmu
                        + npsi
                        + ngamma
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + ntrend_sh
                        + j,
                        1 + nmu + ntrend_loc + i,
                    ] = np.sum(Dmutepst * covariates_loc[:, i] * covariates_sh[:, j])
        if nind_sc > 0 and nind_sh > 0:
            for i in range(nind_sc):
                for j in range(nind_sh):
                    # Sub-block added by Victor (scale exponential involved), alpha_cov_i*gamma_cov_j
                    Hxx[
                        2
                        + self.ngamma0
                        + nmu
                        + npsi
                        + ngamma
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + ntrend_sh
                        + j,
                        2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + i,
                    ] = np.sum(
                        Dpsitepst * covariates_sc[:, i] * covariates_sh[:, j] * psit
                    )
        if nind_sh > 0 and ntrend_loc > 0:
            for i in range(nind_sh):
                # aux = 0
                # for k, tt in enumerate(self.t):
                #     aux += Dmutepst[k] * tt * covariates_sh[k, i]
                # Sub-block added by Victor, betaT*gamma_cov_i
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ngamma
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc
                    + ntrend_sh
                    + i,
                    1 + nmu,
                ] = np.sum(Dmutepst * self.t * covariates_sh[:, i])
        if ntrend_sc > 0:
            for i in range(npsi):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += (
                        (D2psit[k] * psit[k] + Dpsit[k])
                        * tt
                        * self._Dparam(tt, i + 1)
                        * psit[k]
                    )
                # Sub-block number 54 (Scale exponential involved), alpha_i*alphaT
                Hxx[
                    2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc,
                    2 + nmu + ntrend_loc + nind_loc + i,
                ] = aux
        if nind_sc > 0:
            for i in range(npsi):
                for j in range(nind_sc):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += (
                            (D2psit[k] * psit[k] + Dpsit[k])
                            * covariates_sc[k, j]
                            * self._Dparam(tt, i + 1)
                            * psit[k]
                        )
                    # Sub-block number 55 (Scale exponential involved), alpha_i*alpha_cov_j
                    Hxx[
                        2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + j,
                        2 + nmu + ntrend_loc + nind_loc + i,
                    ] = aux
        if nmu > 0 and npsi > 0:
            for j in range(nmu):
                for i in range(npsi):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += (
                            Dmutpsit[k]
                            * self._Dparam(tt, i + 1)
                            * self._Dparam(tt, j + 1)
                            * psit[k]
                        )
                    # Sub-block number 41 (Scale exponential involved), beta_j*alpha_i
                    Hxx[2 + nmu + ntrend_loc + nind_loc + i, 1 + j] = aux
        if nmu > 0 and ngamma > 0:
            for j in range(nmu):
                for i in range(ngamma):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += (
                            Dmutepst[k]
                            * self._Dparam(tt, i + 1)
                            * self._Dparam(tt, j + 1)
                        )
                    # Sub-block number 42, beta_j*gamma_i
                    Hxx[
                        2
                        + self.ngamma0
                        + nmu
                        + npsi
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + i,
                        1 + j,
                    ] = aux
        if npsi > 0 and ngamma > 0:
            for j in range(npsi):
                for i in range(ngamma):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += (
                            Dpsitepst[k]
                            * self._Dparam(tt, i + 1)
                            * self._Dparam(tt, j + 1)
                            * psit[k]
                        )
                    # Sub-block number 43 (Scale exponential involved), alpha_j*gamma_i
                    Hxx[
                        2
                        + self.ngamma0
                        + nmu
                        + npsi
                        + ntrend_loc
                        + nind_loc
                        + ntrend_sc
                        + nind_sc
                        + i,
                        2 + nmu + ntrend_loc + nind_loc + j,
                    ] = aux

        if nind_sh > 0:
            for i in range(nind_sh):
                # Sub-block added by Victor, beta0*gamma_cov_i
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ngamma
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc
                    + ntrend_sh
                    + i,
                    0,
                ] = np.sum(Dmutepst * covariates_sh[:, i])
                # Sub-block added by Victor (scale exponential involved), alpha0*gamma_cov_i
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ngamma
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc
                    + ntrend_sh
                    + i,
                    1 + nmu + ntrend_loc + nind_loc,
                ] = np.sum(Dpsitepst * psit * covariates_sh[:, i])

                # aux = 0
                # for k, tt in enumerate(self.t):
                #     aux += Dpsitepst[k] * tt * covariates_sh[k, i] * psit[k]
                # Sub-bloc added by Victor (scale exponential involved), alphaT*gamma_cov_i
                Hxx[
                    2
                    + self.ngamma0
                    + nmu
                    + npsi
                    + ngamma
                    + ntrend_loc
                    + nind_loc
                    + ntrend_sc
                    + nind_sc
                    + ntrend_sh
                    + i,
                    1 + nmu + npsi + ntrend_loc + nind_loc,
                ] = np.sum(Dpsitepst * self.t * covariates_sh[:, i] * psit)

        # Simmetric part of the Hessian
        Hxx = Hxx + np.tril(Hxx, -1).T

        return f, Jx, Hxx

    def _update_params(self, **kwargs: dict) -> None:
        """
        Updating the parameters of the class with the given keyword arguments.
        Used in the fitting process to update based on fit_result.

        Parameters
        ----------
        **kwargs : dict
            Dictionary containing the parameters to update.
        """
        self.beta0 = kwargs.get("beta0", None)
        self.beta = kwargs.get("beta", np.empty(0))
        self.betaT = kwargs.get("betaT", None)
        self.beta_cov = kwargs.get("beta_cov", np.empty(0))

        self.alpha0 = kwargs.get("alpha0", None)
        self.alpha = kwargs.get("alpha", np.empty(0))
        self.alphaT = kwargs.get("alphaT", None)
        self.alpha_cov = kwargs.get("alpha_cov", np.empty(0))

        # self.ngamma0 = kwargs.get("ngamma0", 0)
        self.gamma0 = kwargs.get("gamma0", None)
        self.gamma = kwargs.get("gamma", np.empty(0))
        self.gammaT = kwargs.get("gammaT", None)
        self.gamma_cov = kwargs.get("gamma_cov", np.empty(0))

        self.xopt = kwargs.get("x", None)

    def _parametro(
        self,
        beta0: Optional[float] = None,
        beta: Optional[np.ndarray] = None,
        betaT: Optional[float] = None,
        beta_cov: Optional[np.ndarray] = None,
        covariates: Optional[np.ndarray] = None,
        indicesint: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
        x: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if beta is None:
            beta = np.empty(0)
        if betaT is None or betaT.size == 0:
            betaT = np.empty(0)
            ntend = 0
        else:
            ntend = 1
        if beta_cov is None:
            beta_cov = np.empty(0)
        if covariates is None:
            covariates = np.empty((0, 0))
        if indicesint is None:
            indicesint = np.empty(0)
        if times is None:
            times = np.empty(0)
        if x is not None:
            x = np.asarray([x])
        else:
            x = self.t

        return self.parametro(
            beta0, beta, betaT, beta_cov, covariates, indicesint, times, x, ntend
        )

    @staticmethod
    @njit(fastmath=True)
    def parametro(
        beta0: Optional[float] = None,
        beta: Optional[np.ndarray] = None,
        betaT: Optional[float] = None,
        beta_cov: Optional[np.ndarray] = None,
        covariates: Optional[np.ndarray] = None,
        indicesint: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
        x: Optional[np.ndarray] = None,
        ntend: Optional[int] = None,
    ) -> np.ndarray:
        """This function computes the location, scale and shape parameters for given parameters. Expressions by (2)-(3) in the paper

        Parameters
        ----------
        beta0 : float, optional
            Value of the intercept
        beta : np.ndarray, optional
            Value of the harmonics terms
        betaT : float, optional
            Trend parameter
        beta_cov : np.ndarray, optional
            Covariate parameters
        covariates : np.ndarray, optional
            Covariate matrix, where each column corresponds to a covariate and each row to a time point
        indicesint : np.ndarray, optional
            Covariate mean values in the integral interval
        times : np.ndarray, optional
            Times when covariates are known, used to find the nearest value
        t : np.ndarray, optional
            Specific time point to evaluate the parameters at, if None, uses the times given

        Returns
        -------
        y : np.ndarray
            Values of the parameter
        """

        m = len(x)

        na, nind = covariates.shape
        nparam = beta.size
        # Chek if the number of parameters is even
        if nparam % 2 != 0:
            raise ValueError("Parameter number must be even")

        # Adding the intercept term
        if beta0 is not None and np.asarray(beta0).size > 0:
            y = beta0 * np.ones(m)
        else:
            y = np.zeros(m)

        # Adding the harmonic part
        if nparam > 0:
            for i in prange(nparam // 2):
                y = (
                    y
                    + beta[2 * i] * np.cos((i + 1) * 2 * np.pi * x)
                    + beta[2 * i + 1] * np.sin((i + 1) * 2 * np.pi * x)
                )

        # Adding the tendency part
        if ntend > 0:
            y = y + betaT * x

        # Adding the covariate part
        if nind > 0:
            if indicesint.shape[0] > 0:
                if times.shape[0] == 0:
                    # for i in prange(nind):
                    #     y = y + beta_cov[i] * indicesint[i]
                    y = y + indicesint @ beta_cov
                else:
                    # for i in prange(nind):
                    #     indicesintaux = search(
                    #         times, covariates[:, i], x.flatten()
                    #     )
                    #     y = y + beta_cov[i] * indicesintaux
                    idx = np.searchsorted(times, x, side="right")
                    valid = idx < times.size

                    y_add = np.zeros_like(x, dtype=np.float64)
                    if np.any(valid):
                        # pick rows from covariates and do one matvec
                        A = covariates[idx[valid], :]  # (k, nind)
                        y_add[valid] = A @ beta_cov  # (k,)

                    y = y + y_add
            else:
                # for i in prange(nind):
                #     y = y + beta_cov[i] * covariates[:, i]
                y = y + covariates @ beta_cov

        return y

    def _evaluate_params(
        self,
        beta0: Optional[float] = None,
        beta: Optional[np.ndarray] = None,
        betaT: Optional[float] = None,
        beta_cov: Optional[np.ndarray] = None,
        alpha0: Optional[float] = None,
        alpha: Optional[np.ndarray] = None,
        alphaT: Optional[float] = None,
        alpha_cov: Optional[np.ndarray] = None,
        gamma0: Optional[float] = None,
        gamma: Optional[np.ndarray] = None,
        gammaT: Optional[float] = None,
        gamma_cov: Optional[np.ndarray] = None,
        covariates_loc: Optional[np.ndarray] = None,
        covariates_sc: Optional[np.ndarray] = None,
        covariates_sh: Optional[np.ndarray] = None,
    ):
        """
        Function to evaluate the parameters in the corresponding values

        Parameters
        ----------
        beta0 : float, optional
            Intercept for location parameter
        beta : np.ndarray, optional
            Harmonic coefficients for location parameter
        betaT : float, optional
            Trend parameter for location parameter
        beta_cov : np.ndarray, optional
            Covariate coefficients for location parameter
        alpha0 : float, optional
            Intercept for scale parameter
        alpha : np.ndarray, optional
            Harmonic coefficients for scale parameter
        alphaT : float, optional
            Trend parameter for scale parameter
        alpha_cov : np.ndarray, optional
            Covariate coefficients for scale parameter
        gamma0 : float, optional
            Intercept for shape parameter
        gamma : np.ndarray, optional
            Harmonic coefficients for shape parameter
        gammaT : float, optional
            Trend parameter for shape parameter
        gamma_cov : np.ndarray, optional
            Covariate coefficients for shape parameter
        covariates_loc : np.ndarray, optional
            Covariates for location parameter
        covariates_sc : np.ndarray, optional
            Covariates for scale parameter
        covariates_sh : np.ndarray, optional
            Covariates for shape parameter
        """
        # Evaluate the location parameter at each time t as function of the actual values of the parameters given by p
        mut1 = self._parametro(beta0, beta, betaT, beta_cov, covariates_loc)
        # Evaluate the scale parameter at each time t as function of the actual values of the parameters given by p
        psit1 = np.exp(self._parametro(alpha0, alpha, alphaT, alpha_cov, covariates_sc))
        # Evaluate the shape parameter at each time t as function of the actual values of the parameters given by p
        epst = self._parametro(gamma0, gamma, gammaT, gamma_cov, covariates_sh)

        return mut1, psit1, epst

    @staticmethod
    def _Dparam(t: float | np.ndarray, i: int) -> float | np.ndarray:
        """
        Derivative of the location, scale and shape fucntions with respect to harmonic parameters. It corresponds to the rhs in equation (A.11) of the paper

        Parameters
        ----------
        t : float or np.ndarray
            Time in yearly scale
        i : int
            Harmonic index

        Returns
        -------
        dp : float or np.ndarray
            Corresponding derivative
        """

        if i % 2 == 0:
            dp = np.sin(i / 2 * 2 * np.pi * t)
        else:
            dp = np.cos((i + 1) / 2 * 2 * np.pi * t)
        return dp

    def _quantile(self, prob=None, harm=False) -> np.ndarray:
        """
        Calculates the quantile q associated with a given parameterization, the main input is quanval introduced in __init__ (default 0.95)

        Parameters
        ----------
        harm : bool, default=False
            If True, the quantile is calculated for the harmonic parameters only, otherwise it includes the trend and covariates parameters

        Returns
        -------
        Q : np.ndarray
            Quantile values at each time t
        """

        if harm:
            betaT = None
            alphaT = None
            gammaT = None
            cov_loc = None
            beta_cov = None
            cov_sc = None
            alpha_cov = None
            cov_sh = None
            gamma_cov = None
        else:
            betaT = self.betaT
            alphaT = self.alphaT
            gammaT = self.gammaT
            cov_loc = self.covariates.iloc[:, self.list_loc].values
            beta_cov = self.beta_cov
            cov_sc = self.covariates.iloc[:, self.list_sc].values
            alpha_cov = self.alpha_cov
            cov_sh = self.covariates.iloc[:, self.list_sh].values
            gamma_cov = self.gamma_cov

        if prob is None:
            prob = self.quanval

        Q = np.zeros(len(self.xt))

        # Evaluate the parameters
        mut1, psit1, epst = self._evaluate_params(
            beta0=self.beta0,
            beta=self.beta,
            betaT=betaT,
            beta_cov=beta_cov,
            alpha0=self.alpha0,
            alpha=self.alpha,
            alphaT=alphaT,
            alpha_cov=alpha_cov,
            gamma0=self.gamma0,
            gamma=self.gamma,
            gammaT=gammaT,
            gamma_cov=gamma_cov,
            covariates_loc=cov_loc,
            covariates_sc=cov_sc,
            covariates_sh=cov_sh,
        )

        # The values whose shape parameter is almost cero corresponds to the GUMBEL distribution, locate their positions if they exist
        posG = np.where(np.abs(epst) <= 1e-8)[0]
        # The remaining values correspond to WEIBULL or FRECHET
        pos = np.where(np.abs(epst) > 1e-8)[0]
        # The corresponding GUMBEl values are set to 1 to avoid numerical problems, note that for those cases the GUMBEL expressions are used
        epst[posG] = 1

        mut = mut1
        psit = psit1
        mut[pos] = mut1[pos] + psit1[pos] * (self.kt[pos] ** epst[pos] - 1) / epst[pos]
        psit[pos] = psit1[pos] * self.kt[pos] ** epst[pos]
        # Modify the parameters to include Gumbel
        mut[posG] += psit[posG] * np.log(self.kt[posG])

        # Evaluate the quantile
        Q[pos] = (
            mut[pos]
            - (1 - (-np.log(prob) / self.kt[pos]) ** (-epst[pos]))
            * psit[pos]
            / epst[pos]
        )
        Q[posG] = mut[posG] - psit[posG] * np.log(-np.log(prob) / self.kt[posG])

        return Q

    def plot(self, return_plot: bool = False, save: bool = False, init_year: int = 0):
        """
        Plot the location, scale and shape parameters, also the PP plot and QQ plot

        Return period plot is plotted if and only if no covariates and trends are included

        Parameters
        ----------
        return_plot : bool, default=True
            If True, return period plot is plotted
        save : bool, default=False
            If True, save all the figures in a "Figures/"
        init_year : int, default=0
            Initial year for plotting purposes
        """

        # Parameter Evaluation
        mut1, psit1, epst = self._evaluate_params(
            beta0=self.beta0,
            beta=self.beta,
            betaT=self.betaT,
            beta_cov=self.beta_cov,
            alpha0=self.alpha0,
            alpha=self.alpha,
            alphaT=self.alphaT,
            alpha_cov=self.alpha_cov,
            gamma0=self.gamma0,
            gamma=self.gamma,
            gammaT=self.gammaT,
            gamma_cov=self.gamma_cov,
            covariates_loc=self.covariates.iloc[:, self.list_loc].values,
            covariates_sc=self.covariates.iloc[:, self.list_sc].values,
            covariates_sh=self.covariates.iloc[:, self.list_sh].values,
        )

        posG = np.where(np.abs(epst) <= 1e-8)[0]
        pos = np.where(np.abs(epst) > 1e-8)[0]
        epst[posG] = 1

        mut = mut1.copy()
        psit = psit1.copy()
        mut[pos] = mut1[pos] + psit1[pos] * (self.kt[pos] ** epst[pos] - 1) / epst[pos]
        psit[pos] = psit1[pos] * self.kt[pos] ** epst[pos]
        mut[posG] += psit[posG] * np.log(self.kt[posG])

        # Confidence intervals (TODO: AÑADIR EN OTRA FUNCION QUIZAS)
        Dq = self._DQuantile()
        Dermut, Derpsit, Derepst = self._Dmupsiepst()

        stdmut = np.sqrt(
            np.sum(
                (
                    Dermut.T
                    @ self.invI0[
                        : 1 + 2 * self.nmu + self.ntrend_loc + self.nind_loc,
                        : 1 + 2 * self.nmu + self.ntrend_loc + self.nind_loc,
                    ]
                )
                * Dermut.T,
                axis=1,
            )
        )
        stdpsit = np.sqrt(
            np.sum(
                (
                    Derpsit.T
                    @ self.invI0[
                        1 + 2 * self.nmu + self.ntrend_loc + self.nind_loc : 2
                        + 2 * self.nmu
                        + self.ntrend_loc
                        + self.nind_loc
                        + 2 * self.npsi
                        + self.ntrend_sc
                        + self.nind_sc,
                        1 + 2 * self.nmu + self.ntrend_loc + self.nind_loc : 2
                        + 2 * self.nmu
                        + self.ntrend_loc
                        + self.nind_loc
                        + 2 * self.npsi
                        + self.ntrend_sc
                        + self.nind_sc,
                    ]
                )
                * Derpsit.T,
                axis=1,
            )
        )
        if self.ngamma0 == 1 or self.ngamma > 0 or self.ntrend_sh or self.nind_sh > 0:
            stdepst = np.sqrt(
                np.sum(
                    (
                        Derepst.T
                        @ self.invI0[
                            2
                            + 2 * self.nmu
                            + self.ntrend_loc
                            + self.nind_loc
                            + 2 * self.npsi
                            + self.ntrend_sc
                            + self.nind_sc : 2
                            + 2 * self.nmu
                            + self.ntrend_loc
                            + self.nind_loc
                            + 2 * self.npsi
                            + self.ntrend_sc
                            + self.nind_sc
                            + self.ngamma0
                            + 2 * self.ngamma
                            + self.ntrend_sh
                            + self.nind_sh,
                            2
                            + 2 * self.nmu
                            + self.ntrend_loc
                            + self.nind_loc
                            + 2 * self.npsi
                            + self.ntrend_sc
                            + self.nind_sc : 2
                            + 2 * self.nmu
                            + self.ntrend_loc
                            + self.nind_loc
                            + 2 * self.npsi
                            + self.ntrend_sc
                            + self.nind_sc
                            + self.ngamma0
                            + 2 * self.ngamma
                            + self.ntrend_sh
                            + self.nind_sh,
                        ]
                    )
                    * Derepst.T,
                    axis=1,
                )
            )
        else:
            stdepst = 0

        stdDq = np.sqrt(np.sum((Dq.T @ self.invI0) * Dq.T, axis=1))

        # Confidence interval for mut
        ci_up_mut = mut + norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdmut
        ci_low_mut = mut - norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdmut
        ci_up_psit = (
            psit + norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdpsit
        )
        ci_low_psit = (
            psit - norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdpsit
        )

        # Location and Scale parameter plotting
        t_anual = np.mod(self.t, 1)
        quan95 = self._quantile()
        rp10 = self._quantile(1 - 1 / 10)
        rp100 = self._quantile(1 - 1 / 100)

        if (
            self.ntrend_loc == 0
            and self.ntrend_sc == 0
            and self.ntrend_sh == 0
            and self.nind_loc == 0
            and self.nind_sc == 0
            and self.nind_sh == 0
        ):
            fig, ax1 = plt.subplots(figsize=(10, 6))

            #############
            month_initials = [
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
            ]
            month_positions = [i / 12 for i in range(12)]

            # Reorder the return periods to start from July
            rt_10 = np.zeros(13)
            rt_50 = np.zeros(13)
            rt_100 = np.zeros(13)
            for i in range(12):
                # Map i to the correct month index (July = 0, June = 11)
                month_idx = (i + 6) % 12
                rt_10[i] = self._aggquantile(
                    1 - 1 / 10, month_idx / 12, (month_idx + 1) / 12
                )
                rt_50[i] = self._aggquantile(
                    1 - 1 / 50, month_idx / 12, (month_idx + 1) / 12
                )
                rt_100[i] = self._aggquantile(
                    1 - 1 / 100, month_idx / 12, (month_idx + 1) / 12
                )

            rt_10[12] = rt_10[0]
            rt_50[12] = rt_50[0]
            rt_100[12] = rt_100[0]

            # For the data points, shift the time values by 0.5 years
            t_shifted = t_anual.copy()
            t_shifted[t_anual < 0.5] += 0.5
            t_shifted[t_anual >= 0.5] -= 0.5
            t_ord = np.argsort(t_shifted)

            ### Added by Tomas Carlotto
            # Creating variables to store the evolution of pdf and sf over time
            var_grid_resolution = 50
            # t_anual_ord = t_anual[t_ord]
            mu_t = mut[t_ord]
            psi_t = psit[t_ord]
            xi_t = epst[t_ord]
            # Definition of the Hs value grid
            lim_max = np.max(self.xt) + 1
            lim_min = np.min(self.xt)
            hvar = np.linspace(lim_min, lim_max, var_grid_resolution)
            t_grid, x_grid = np.meshgrid(t_shifted[t_ord], hvar)

            # Calculating the 1-CDF for each grid point (Exceedance Probabilities)
            sf = np.array(
                [
                    genextreme.sf(x_grid[:, i], c=-xi_t[i], loc=mu_t[i], scale=psi_t[i])
                    for i in range(len(t_shifted[t_ord]))
                ]
            ).T
            # Calculating the Probability Density Function (pdf) for each grid point
            pdf = np.array(
                [
                    genextreme.pdf(
                        x_grid[:, i], c=-xi_t[i], loc=mu_t[i], scale=psi_t[i]
                    )
                    for i in range(len(t_shifted[t_ord]))
                ]
            ).T

            cf = ax1.contourf(t_shifted[t_ord], hvar, sf, levels=50, cmap="viridis_r")
            cbar = fig.colorbar(cf, ax=ax1)
            cbar.set_label("Exceedance probability", fontsize=12)
            # ===============

            # Use t_shifted for plotting data points
            ax1.plot(
                t_shifted[t_ord],
                self.xt[t_ord],
                marker="o",
                linestyle="None",
                color="black",
                markersize=5,
                label="Data",
                alpha=0.9,
            )

            # Use t_shifted for other lines as well
            ax1.plot(
                t_shifted[t_ord],
                mut[t_ord],
                label="Location",
                linewidth=2,
                color=self.colors[0],
                alpha=1,
            )

            ax1.fill_between(
                t_shifted[t_ord],
                mut[t_ord] - psit[t_ord],
                mut[t_ord] + psit[t_ord],
                label=r"Location $\pm$ Scale",
                color="tab:blue",
                alpha=0.3,
            )

            month_positions_aux = [i / 12 for i in range(13)]
            ax1.step(
                month_positions_aux,
                rt_10,
                where="post",
                linestyle="-",
                linewidth=1,
                label="10 years",
                color="tab:red",
            )
            ax1.step(
                month_positions_aux,
                rt_50,
                where="post",
                linestyle="-",
                linewidth=1,
                label="50 years",
                color="tab:purple",
            )
            ax1.step(
                month_positions_aux,
                rt_100,
                where="post",
                linestyle="-",
                linewidth=1,
                label="100 years",
                color="tab:green",
            )

            ax1.set_title(f"Parameters Evolution ({self.var_name})")
            ax1.set_xlabel("Time (Months)")
            ax1.set_ylabel(f"{self.var_name}")
            ax1.grid(True)
            ax1.legend(loc="best")
            ax1.set_xlim(0, 1)
            ax1.set_xticks(month_positions, month_initials, rotation=45)
            if save:
                plt.savefig(
                    f"Figures/Adjustment_Evolution_{self.var_name}.png",
                    dpi=300,
                )
            plt.show()

        else:
            fig, ax1 = plt.subplots(figsize=(20, 6))
            ax1.plot(
                self.t + init_year,
                self.xt,
                marker="+",
                linestyle="None",
                color="black",
                markersize=5,
                label="Data",
            )
            ax1.plot(
                self.t + init_year,
                mut,
                label="Location",
                linewidth=2,
                # color=self.colors[0],
                color="tab:blue",
                alpha=1,
            )
            ax1.fill_between(
                self.t + init_year,
                mut - psit,
                mut + psit,
                label=r"Location $\pm$ Scale",
                color="tab:blue",
                alpha=0.4,
            )

            # ax1.plot(
            #     self.t + init_year,
            #     rp10,
            #     label="10 years",
            #     linewidth=1,
            #     color="tab:red",
            #     linestyle="--",
            #     alpha=0.9,
            # )

            # ax1.plot(
            #     self.t + init_year,
            #     rp100,
            #     label="100 years",
            #     linewidth=1,
            #     color="tab:green",
            #     linestyle="--",
            #     alpha=0.9,
            # )

            ax1.plot(
                self.t + init_year,
                epst,
                label="Shape",
                linewidth=1,
                color="tab:orange",
                alpha=0.9,
            )

            # Aggregated return period lines
            if return_plot:
                n_years = int(np.ceil(self.t[-1]))
                rt_10 = np.zeros(n_years)
                for year in range(n_years):
                    rt_10[year] = self._aggquantile(
                        1 - 1 / 10, year, year + 1
                    )  # 10-year return level at each year
                rt_50 = np.zeros(n_years)
                for year in range(n_years):
                    rt_50[year] = self._aggquantile(
                        1 - 1 / 50, year, year + 1
                    )  # 50-year return level at each year
                rt_100 = np.zeros(n_years)
                for year in range(n_years):
                    rt_100[year] = self._aggquantile(
                        1 - 1 / 100, year, year + 1
                    )  # 100-year return level at each year

                ax1.step(
                    np.arange(init_year, init_year + n_years),
                    rt_10,
                    where="post",
                    linestyle="-",
                    linewidth=1,
                    label="10 years",
                    color="tab:red",
                )
                ax1.step(
                    np.arange(init_year, init_year + n_years),
                    rt_50,
                    where="post",
                    linestyle="-",
                    linewidth=1,
                    label="50 years",
                    color="tab:purple",
                )
                ax1.step(
                    np.arange(init_year, init_year + n_years),
                    rt_100,
                    where="post",
                    linestyle="-",
                    linewidth=1,
                    label="100 years",
                    color="tab:green",
                )

        ax1.set_xlabel("Time (years)")
        ax1.set_ylabel(f"{self.var_name}")
        # ax1.set_title(f"Evolution of location and scale parameters ({self.var_name})")
        ax1.set_title(f"Evolution of parameters ({self.var_name})")
        ax1.grid(True)
        ax1.legend(loc="best")
        # ax2.set_ylim(0,1.5)
        # ax1.set_xlim(-0.07, 40)
        ax1.margins(x=0.01)
        plt.tight_layout()
        if save:
            os.makedirs("Figures", exist_ok=True)
            # plt.savefig(f"Figures/Location_Scale_Parameters_{self.var_name}.png", dpi=300)
            plt.savefig(f"Figures/Adjustment_Evolution_{self.var_name}.png", dpi=300)
        plt.show()

        ###### 1st Year PLOT
        month_initials = [
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
        ]
        month_positions = [i / 12 for i in range(12)]

        #### Creating the first year plot
        mask_year = (self.t >= 0) & (self.t <= 1)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(
            self.t[mask_year],
            self.xt[mask_year],
            marker="+",
            linestyle="None",
            color="black",
            markersize=5,
            label=f"{self.var_name}",
        )
        ax1.plot(
            self.t[mask_year],
            mut[mask_year],
            label=r"$\mu_t$",
            linewidth=2,
            color=self.colors[0],
            alpha=1,
        )
        uppermaxs = np.maximum(
            mut[mask_year] - psit[mask_year], mut[mask_year] + psit[mask_year]
        )
        lowermins = np.minimum(
            mut[mask_year] - psit[mask_year], mut[mask_year] + psit[mask_year]
        )
        ax1.fill_between(
            self.t[mask_year], lowermins, uppermaxs, color=self.colors[0], alpha=0.3
        )
        ax1.plot(
            self.t[mask_year],
            quan95[mask_year],
            linestyle="dashed",
            color=self.colors[2],
            label="95th Quantile",
        )
        ax1.set_title(f"Location and Scale Parameters (First Year) ({self.var_name})")
        ax1.set_xlabel("Time")
        ax1.set_ylabel(r"$\mu_t$")
        ax1.grid(True)
        ax1.legend(loc="best")
        ax1.margins(x=0.01)
        plt.xticks(month_positions, month_initials)
        if save:
            plt.savefig(
                f"Figures/Location_Scale_Parameters_FirstYear_{self.var_name}.png",
                dpi=300,
            )
        plt.show()

        ### FIRST MONTH PLOT Creating the first monthly plot if not monthly or annual data
        # mask_month = (self.t >= 0) & (self.t <= 1 / 12)
        # if sum(mask_month) > 1:
        #     fig, ax1 = plt.subplots(figsize=(10, 6))
        #     ax1.plot(
        #         self.t[mask_month],
        #         self.xt[mask_month],
        #         marker="+",
        #         linestyle="None",
        #         color="black",
        #         markersize=5,
        #         label=f"{self.var_name}",
        #     )
        #     # ax2 = ax1.twinx()
        #     ax1.plot(
        #         self.t[mask_month],
        #         mut[mask_month],
        #         label=r"$\mu_t$",
        #         linewidth=2,
        #         color=self.colors[0],
        #         alpha=1,
        #     )
        #     uppermaxs = np.maximum(
        #         mut[mask_month] - psit[mask_month], mut[mask_month] + psit[mask_month]
        #     )
        #     lowermins = np.minimum(
        #         mut[mask_month] - psit[mask_month], mut[mask_month] + psit[mask_month]
        #     )
        #     ax1.fill_between(
        #         self.t[mask_month],
        #         lowermins,
        #         uppermaxs,
        #         color=self.colors[0],
        #         alpha=0.3,
        #         label="Location +- scale",
        #     )

        #     # ax1.plot(
        #     #     self.t[mask_month],
        #     #     rt_10[mask_month],
        #     #     linestyle="dashed",
        #     #     color=self.colors[2],
        #     #     label="10 years",
        #     # )
        #     # ax1.plot(
        #     #     self.t[mask_month],
        #     #     rt_50[mask_month],
        #     #     linestyle="dashed",
        #     #     color=self.colors[2],
        #     #     label="50 years",
        #     # )
        #     # ax1.plot(
        #     #     self.t[mask_month],
        #     #     rt_100[mask_month],
        #     #     linestyle="dashed",
        #     #     color=self.colors[2],
        #     #     label="100 years",
        #     # )
        #     ax1.set_title(
        #         f"Location and Scale Parameters (First Month) ({self.var_name})"
        #     )
        #     ax1.set_xlabel("Time (yearly scale)")
        #     ax1.set_ylabel(r"$\mu_t$")
        #     # ax2.set_ylabel(r'$\psi_t$')
        #     ax1.grid(True)
        #     # handles = [art for art in l0 + l1 + l2 + l3 if not art.get_label().startswith('_')]
        #     ax1.legend(loc="best")
        #     ax1.margins(x=0.01)
        #     if save:
        #         plt.savefig(
        #             f"Figures/Evolution_Location_FirstYear{self.var_name}.png", dpi=300
        #         )
        #     plt.show()

        #### Shape parameter plot
        # Confidence interval for epst
        # ci_up = (
        #     epst + norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdepst
        # )
        # ci_low = (
        #     epst - norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdepst
        # )

        ##### LOCATION #####
        plt.figure(figsize=(20, 6))
        plt.plot(self.t + init_year, mut, color="tab:blue")
        # plt.fill_between(
        #     t_anual[t_ord],
        #     ci_low[t_ord],
        #     ci_up[t_ord],
        #     color=self.colors[0],
        #     alpha=0.3,
        #     label=r"$\xi_t$ Confidence Interval",
        # )
        plt.title(f"Location parameter ({self.var_name})")
        plt.xlabel("Time (yearly scale)")
        plt.ylabel(r"$\mu_t$")
        plt.grid(True)
        if save:
            plt.savefig(f"Figures/Evolution_Location{self.var_name}.png", dpi=300)
        plt.show()

        ##### SCALE #####
        plt.figure(figsize=(20, 6))
        plt.plot(self.t + init_year, psit, color="tab:green")
        # plt.fill_between(
        #     t_anual[t_ord],
        #     ci_low[t_ord],
        #     ci_up[t_ord],
        #     color=self.colors[0],
        #     alpha=0.3,
        #     label=r"$\xi_t$ Confidence Interval",
        # )
        plt.title(f"Scale parameter ({self.var_name})")
        plt.xlabel("Time (yearly scale)")
        plt.ylabel(r"$\psi_t$")
        plt.grid(True)
        if save:
            plt.savefig(f"Figures/Evolution_Scale{self.var_name}.png", dpi=300)
        plt.show()

        ##### SHAPE #####
        plt.figure(figsize=(20, 6))
        plt.plot(self.t + init_year, epst, color="tab:orange")
        # plt.fill_between(
        #     t_anual[t_ord],
        #     ci_low[t_ord],
        #     ci_up[t_ord],
        #     color=self.colors[0],
        #     alpha=0.3,
        #     label=r"$\xi_t$ Confidence Interval",
        # )
        plt.title(f"Shape parameter ({self.var_name})")
        plt.xlabel("Time (yearly scale)")
        plt.ylabel(r"$\xi_t$")
        plt.grid(True)
        if save:
            plt.savefig(f"Figures/Evolution_Shape{self.var_name}.png", dpi=300)
        plt.show()

        # ### Harmonic Location parameter plot
        # if self.nmu > 0:
        #     t_ord = np.argsort(t_anual)
        #     quan95_2 = self._quantile(harm=True)

        #     mut2 = self._parametro(self.beta0, self.beta)
        #     # Confidence interval for mut
        #     ci_up = mut2 + norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdmut
        #     ci_low = (
        #         mut2 - norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdmut
        #     )

        #     plt.figure(figsize=(10, 6))
        #     plt.plot(
        #         t_anual[t_ord],
        #         self.xt[t_ord],
        #         marker="+",
        #         linestyle="None",
        #         color="black",
        #         markersize=5,
        #         label=f"{self.var_name}",
        #     )
        #     plt.plot(
        #         t_anual[t_ord],
        #         mut2[t_ord],
        #         label=r"$\mu_t$",
        #         linewidth=2,
        #         color=self.colors[0],
        #     )
        #     plt.fill_between(
        #         t_anual[t_ord],
        #         ci_low[t_ord],
        #         ci_up[t_ord],
        #         color=self.colors[0],
        #         alpha=0.3,
        #     )
        #     # Confidence interval for the quantile
        #     ci_up = (
        #         quan95_2 + norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdDq
        #     )
        #     ci_low = (
        #         quan95_2 - norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdDq
        #     )
        #     plt.plot(
        #         t_anual[t_ord],
        #         quan95_2[t_ord],
        #         linestyle="dashed",
        #         color=self.colors[1],
        #         markersize=5,
        #         label=rf"$q_{self.quanval}$",
        #     )
        #     plt.fill_between(
        #         t_anual[t_ord],
        #         ci_low[t_ord],
        #         ci_up[t_ord],
        #         color=self.colors[1],
        #         alpha=0.3,
        #     )
        #     plt.title(f"Harmonic part of Location parameter ({self.var_name})")
        #     plt.xlabel("Time (yearly scale)")
        #     plt.ylabel(r"$\mu_t$")
        #     plt.xticks(month_positions, month_initials)
        #     plt.legend(loc="best")
        #     plt.grid(True)
        #     if save:
        #         plt.savefig(
        #             f"Figures/Harmonic_Location_Parameter_{self.var_name}.png", dpi=300
        #         )
        #     plt.show()

        # ### Scale parameter plot
        # if self.npsi > 0:
        #     t_ord = np.argsort(t_anual)

        #     psit2 = np.exp(self._parametro(self.alpha0, self.alpha))
        #     # Confidence interval for psit
        #     ci_up = (
        #         psit2 + norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdpsit
        #     )
        #     ci_low = (
        #         psit2 - norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdpsit
        #     )

        #     plt.figure(figsize=(10, 6))
        #     plt.plot(
        #         t_anual[t_ord],
        #         psit2[t_ord],
        #         label=r"$\psi_t$",
        #         linewidth=2,
        #         color=self.colors[0],
        #     )
        #     plt.fill_between(
        #         t_anual[t_ord],
        #         ci_low[t_ord],
        #         ci_up[t_ord],
        #         color=self.colors[0],
        #         alpha=0.3,
        #         label=r"$\psi_t$ Confidence Interval",
        #     )
        #     # plt.plot(t_anual[t_ord], quan95[t_ord], linestyle='dashed', color=self.colors[2], markersize=5, label=fr"$q_{self.quanval}$")
        #     plt.title(f"Harmonic part of Scale parameter ({self.var_name})")
        #     plt.xlabel("Time (yearly scale)")
        #     plt.xticks(month_positions, month_initials)
        #     plt.ylabel(r"$\psi_t$")
        #     plt.grid(True)
        #     if save:
        #         plt.savefig(
        #             f"Figures/Harmonic_Scale_Parameter_{self.var_name}.png", dpi=300
        #         )
        #     plt.show()

        #### PP Plot
        self.PPplot(save=save)

        #### QQ plot
        self.QQplot(save=save)

        #### Parameters Heatmap plot
        # self.paramplot(save=save)

        #### 3D plot of pdf
        if (
            self.ntrend_loc == 0
            and self.ntrend_sc == 0
            and self.ntrend_sh == 0
            and self.nind_loc == 0
            and self.nind_sc == 0
            and self.nind_sh == 0
        ):
            mu_t = mut[t_ord]
            psi_t = psit[t_ord]
            xi_t = epst[t_ord]
            # Definition of the Hs value grid
            lim_max = np.max(self.xt) + 1
            lim_min = np.min(self.xt)
            hvar = np.linspace(lim_min, lim_max, var_grid_resolution)
            t_grid, x_grid = np.meshgrid(t_shifted[t_ord], hvar)
            # Calculating the Probability Density Function (pdf) for each grid point
            pdf = np.array(
                [
                    genextreme.pdf(
                        x_grid[:, i], c=-xi_t[i], loc=mu_t[i], scale=psi_t[i]
                    )
                    for i in range(len(t_shifted[t_ord]))
                ]
            ).T

            fig = plt.figure(figsize=(15, 6))
            ax = fig.add_subplot(projection="3d")
            ax.plot_surface(t_grid, x_grid, pdf, cmap="viridis_r")
            ax.set_zlim(0, 1)
            ax.set_xlabel("Time (Months)")
            ax.set_ylabel(f"{self.var_name}")
            ax.set_zlabel("PDF")
            ax.set_xticks(month_positions, month_initials)
            ax.view_init(elev=40, azim=-25)
            plt.show()

        #### Return periods
        if (
            self.ntrend_loc == 0
            and self.ntrend_sc == 0
            and self.ntrend_sh == 0
            and self.nind_loc == 0
            and self.nind_sc == 0
            and self.nind_sh == 0
        ) and return_plot:
            self.returnperiod_plot()

    def paramplot(self, save: bool = False):
        """
        Create a heatmap of parameter sensitivities for location, scale and shape.

        Parameters
        ----------
        save : bool, False
        """
        # Get parameters
        loc_params = []
        param_names = []
        if self.fit_result["beta0"] is not None:
            loc_params.append(self.fit_result["beta0"])
            param_names.append("Intercept")
        if self.fit_result["beta"].size > 0:
            for i, b in enumerate(self.fit_result["beta"]):
                loc_params.append(b)
                param_names.append(f"Harm {i}")
        if self.fit_result["beta_cov"].size > 0:
            for i, b in enumerate(self.fit_result["beta_cov"]):
                loc_params.append(b)
                param_names.append(f"{self.covariates.columns[i]}")
        if self.fit_result["betaT"].size > 0:
            loc_params.append(self.fit_result["betaT"])
            param_names.append("Trend")

        # Scale parameters
        scale_params = []
        if self.fit_result["alpha0"] is not None:
            scale_params.append(self.fit_result["alpha0"])
        if self.fit_result["alpha"].size > 0:
            scale_params.extend(self.fit_result["alpha"])
        if self.fit_result["alpha_cov"].size > 0:
            scale_params.extend(self.fit_result["alpha_cov"])
        if self.fit_result["alphaT"].size > 0:
            scale_params.append(self.fit_result["alphaT"])

        # Shape parameters
        shape_params = []
        if self.fit_result["gamma0"] is not None:
            shape_params.append(self.fit_result["gamma0"])
        if self.fit_result["gamma"].size > 0:
            shape_params.extend(self.fit_result["gamma"])
        if self.fit_result["gamma_cov"].size > 0:
            shape_params.extend(self.fit_result["gamma_cov"])
        if self.fit_result["gammaT"].size > 0:
            shape_params.append(self.fit_result["gammaT"])

        # Create data matrix
        max_len = max(len(param_names), len(scale_params), len(shape_params))
        data = np.zeros((3, len(param_names)))
        data[0, : len(loc_params)] = loc_params
        data[1, : len(scale_params)] = scale_params
        data[2, : len(shape_params)] = shape_params

        # Create figure
        fig, ax = plt.subplots(figsize=(20, 4))

        # Create heatmap
        im = ax.imshow(data, cmap="RdBu", aspect="auto", vmin=-2, vmax=2)

        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label("Coefficient value")

        # Add text annotations
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                text = ax.text(
                    j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black"
                )

        # Configure axes
        ax.set_xticks(np.arange(len(param_names)))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(param_names)
        ax.set_yticklabels(["Location", "Scale", "Shape"])

        # Rotate x labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add title and adjust layout
        ax.set_title("Time-dependent GEV Parameters")
        ax.set_xlabel("Components")

        plt.tight_layout()

        if save:
            plt.savefig(
                f"Figures/Parameters_Heatmap_{self.var_name}.png",
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()

    def QQplot(self, save: bool = False):
        """
        QQ plot

        Parameters
        ----------
        save : bool, default=False
            If True, save the plot in "Figures/"
        """
        Ze = -np.log(-np.log(np.arange(1, len(self.xt) + 1) / (len(self.xt) + 1)))
        Zm = self.kt * self._Zstandardt()
        # TODO: Chequear intervalos
        Dwei = self._Dzweibull()
        stdDwei = np.sqrt(np.sum((Dwei.T @ self.invI0) * Dwei.T, axis=1))

        Zmsort = np.sort(Zm)
        t_ord = np.argsort(Zm)

        plt.figure(figsize=(10, 6))
        plt.plot([min(Ze), max(Ze)], [min(Ze), max(Ze)], self.colors[1])
        plt.plot(
            Ze,
            Zmsort,
            "o",
            markeredgecolor=self.colors[0],
            markerfacecolor=self.colors[0],
            markersize=3,
        )
        # If no covariables or trends, plot the confidence interval
        if (
            self.nind_loc == 0
            and self.nind_sc == 0
            and self.nind_sh == 0
            and self.ntrend_loc == 0
            and self.ntrend_sc == 0
        ):
            plt.fill_between(
                Ze,
                Zmsort
                - norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdDwei[t_ord],
                Zmsort
                + norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdDwei[t_ord],
                color=self.colors[0],
                alpha=0.3,
            )

            # If dashed lines prefered
            # plt.plot(Ze, Zmsort+norm.ppf(1-(1-self.quanval)/2, loc=0, scale=1)*stdDwei[t_ord], linestyle='dashed', color=self.colors[2], markersize=5)
            # plt.plot(Ze, Zmsort-norm.ppf(1-(1-self.quanval)/2, loc=0, scale=1)*stdDwei[t_ord], linestyle='dashed', color=self.colors[2], markersize=5)
        plt.title(f"Best model QQ plot ({self.var_name})")
        plt.xlabel("Empirical")
        plt.ylabel("Fitted")
        plt.axis("square")
        plt.grid(True)
        plt.margins(x=0.1)
        if save:
            plt.savefig(f"Figures/QQplot_{self.var_name}.png", dpi=300)
        plt.show()

    def _Zstandardt(self):
        """
        Calculates the standardized variable corresponding to the given parameters

        Return
        ------
        Zt :
            Standarized variable of the given parameters
        """

        Zt = np.zeros(len(self.xt))

        # Evaluate the parameters
        mut1, psit1, epst = self._evaluate_params(
            beta0=self.beta0,
            beta=self.beta,
            betaT=self.betaT,
            beta_cov=self.beta_cov,
            alpha0=self.alpha0,
            alpha=self.alpha,
            alphaT=self.alphaT,
            alpha_cov=self.alpha_cov,
            gamma0=self.gamma0,
            gamma=self.gamma,
            gammaT=self.gammaT,
            gamma_cov=self.gamma_cov,
            covariates_loc=self.covariates.iloc[:, self.list_loc].values,
            covariates_sc=self.covariates.iloc[:, self.list_sc].values,
            covariates_sh=self.covariates.iloc[:, self.list_sh].values,
        )

        # The values whose shape parameter is almost cero corresponds to the GUMBEL distribution, locate their positions if they exist
        posG = np.where(np.abs(epst) <= 1e-8)[0]
        # The remaining values correspond to WEIBULL or FRECHET
        pos = np.where(np.abs(epst) > 1e-8)[0]
        # The corresponding GUMBEl values are set to 1 to avoid numerical problems, note that for those cases the GUMBEL expressions are used
        epst[posG] = 1

        mut = mut1
        psit = psit1
        mut[pos] = mut1[pos] + psit1[pos] * (self.kt[pos] ** epst[pos] - 1) / epst[pos]
        psit[pos] = psit1[pos] * self.kt[pos] ** epst[pos]
        # Modify the parameters to include Gumbel
        mut[posG] += psit[posG] * np.log(self.kt[posG])

        # WEIBULL or FRECHET value
        Zt[pos] = (1 / epst[pos]) * np.log(
            1 + epst[pos] * ((self.xt[pos] - mut[pos]) / psit[pos])
        )
        # GUMBEL value
        Zt[posG] = (self.xt[posG] - mut[posG]) / psit[posG]

        return Zt

    def _Dzweibull(self) -> np.ndarray:
        """
        Calculates the derivatives of the standardized maximum with respect to parameters

        Return
        ------
        Dq : np.ndarray
            Derivative of standarized variable of the given parameters
        """

        nd = len(self.t)

        # Evaluate the parameters
        mut1, psit1, epst = self._evaluate_params(
            beta0=self.beta0,
            beta=self.beta,
            betaT=self.betaT,
            beta_cov=self.beta_cov,
            alpha0=self.alpha0,
            alpha=self.alpha,
            alphaT=self.alphaT,
            alpha_cov=self.alpha_cov,
            gamma0=self.gamma0,
            gamma=self.gamma,
            gammaT=self.gammaT,
            gamma_cov=self.gamma_cov,
            covariates_loc=self.covariates.iloc[:, self.list_loc].values,
            covariates_sc=self.covariates.iloc[:, self.list_sc].values,
            covariates_sh=self.covariates.iloc[:, self.list_sh].values,
        )

        # The values whose shape parameter is almost cero corresponds to the GUMBEL distribution, locate their positions if they exist
        posG = np.where(np.abs(epst) <= 1e-8)[0]
        # The remaining values correspond to WEIBULL or FRECHET
        pos = np.where(np.abs(epst) > 1e-8)[0]
        # The corresponding GUMBEl values are set to 1 to avoid numerical problems, note that for those cases the GUMBEL expressions are used
        epst[posG] = 1

        mut = mut1
        psit = psit1
        mut[pos] = mut1[pos] + psit1[pos] * (self.kt[pos] ** epst[pos] - 1) / epst[pos]
        psit[pos] = psit1[pos] * self.kt[pos] ** epst[pos]
        # Modify the parameters to include Gumbel
        mut[posG] += psit[posG] * np.log(self.kt[posG])

        # Evaluate auxiliarly variables
        xn = (self.xt - mut) / psit
        z = 1 + epst * xn

        # Since z-values must be greater than 0 in order to avoid numerical problems, their values are set to be greater than 1e-8
        z = np.maximum(1e-8, z)
        # zn = z ** (-1 / epst)

        Dmut = np.zeros(nd)
        Dpsit = np.zeros(nd)
        Depst = np.zeros(nd)

        # Derivatives of the quantile function with respect to location, scale and shape parameters
        Dmut[pos] = -1 / (z[pos] * psit[pos])
        Dpsit[pos] = xn[pos] * Dmut[pos]
        Depst[pos] = (1 - 1 / z[pos] - np.log(z[pos])) / (epst[pos] * epst[pos])

        # Gumbel derivatives
        Dmut[posG] = -1 / psit[posG]
        Dpsit[posG] = -xn[posG] / psit[posG]
        Depst[posG] = 0

        ## New Derivatives
        Dmutastmut = np.ones_like(self.kt)
        Dmutastpsit = (-1 + self.kt**epst) / epst
        Dmutastepst = (
            psit1 * (1 + (self.kt**epst) * (epst * np.log(self.kt) - 1)) / (epst**2)
        )

        Dpsitastpsit = self.kt**epst
        Dpsitastepst = np.log(self.kt) * psit1 * (self.kt**epst)

        Dmutastpsit[posG] = np.log(self.kt[posG])
        Dmutastepst[posG] = 0

        Dpsitastpsit[posG] = 1
        Dpsitastepst[posG] = 0

        # Set the Jacobian to zero matrix
        Dq = np.zeros(
            (
                2
                + self.ngamma0
                + 2 * self.nmu
                + 2 * self.npsi
                + 2 * self.ngamma
                + self.ntrend_loc
                + self.nind_loc
                + self.ntrend_sc
                + self.nind_sc
                + self.ntrend_sh
                + self.nind_sh,
                nd,
            )
        )
        # Jacobian elements related to the location parameters beta0 and beta, equation (A.6) in the paper
        Dq[0, :] = Dmut * Dmutastmut

        # If location harmonics are included
        if self.nmu > 0:
            for i in range(2 * self.nmu):
                for k in range(len(self.t)):
                    Dq[i + 1, k] = (
                        Dmut[k] * Dmutastmut[k] * self._Dparam(self.t[k], i + 1)
                    )

        # Jacobian elements related to the location parameters betaT, beta_cov (equation A.9)
        if self.ntrend_loc > 0:
            Dq[1 + 2 * self.nmu, :] = Dmut * self.t * Dmutastmut  # betaT
        if self.nind_loc > 0:
            for i in range(self.nind_loc):
                Dq[1 + 2 * self.nmu + self.ntrend_loc + i, :] = (
                    Dmut * self.covariates.iloc[:, self.list_loc[i]] * Dmutastmut
                )  # beta_cov_i

        # Jacobian elements related to the scale parameters alpha0, alpha (equation A.7)
        Dq[1 + 2 * self.nmu + self.ntrend_loc + self.nind_loc, :] = psit1 * (
            Dpsit * Dpsitastpsit + Dmut * Dmutastpsit
        )  # alpha0
        # If scale harmonic are included
        if self.npsi > 0:
            for i in range(2 * self.npsi):
                for k in range(len(self.t)):
                    Dq[2 + 2 * self.nmu + self.ntrend_loc + self.nind_loc + i, k] = (
                        self._Dparam(self.t[k], i + 1)
                        * psit1[k]
                        * (Dpsit[k] * Dpsitastpsit[k] + Dmut[k] * Dmutastpsit[k])
                    )  # alpha
        # Jacobian elements related to the scale parameters alphaT and beta_cov (equation A.10)
        if self.ntrend_sc > 0:
            Dq[
                2 + 2 * self.nmu + self.ntrend_loc + self.nind_loc + 2 * self.npsi, :
            ] = (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit) * self.t * psit1  # alphaT
        if self.nind_sc > 0:
            for i in range(self.nind_sc):
                Dq[
                    2
                    + 2 * self.nmu
                    + self.ntrend_loc
                    + self.nind_loc
                    + 2 * self.npsi
                    + self.ntrend_sc
                    + i,
                    :,
                ] = (
                    (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit)
                    * self.covariates.iloc[:, self.list_sc[i]]
                    * psit1
                )  # alpha_cov

        # Jacobian elements related to the shape parameters gamma0 and gamma (equation A.10)
        if self.ngamma0 == 1:
            Dq[
                2
                + 2 * self.nmu
                + self.ntrend_loc
                + self.nind_loc
                + 2 * self.npsi
                + self.ntrend_sc
                + self.nind_sc,
                :,
            ] = Depst + Dpsit * Dpsitastepst + Dmut * Dmutastepst
        # If shape harmonics are included
        if self.ngamma > 0:
            for i in range(self.ngamma):
                for k in range(len(self.t)):
                    Dq[
                        2
                        + self.ngamma0
                        + 2 * self.nmu
                        + self.ntrend_loc
                        + self.nind_loc
                        + 2 * self.npsi
                        + self.ntrend_sc
                        + self.nind_sc
                        + i,
                        k,
                    ] = (
                        Depst[k] + Dpsit[k] * Dpsitastepst[k] + Dmut[k] * Dmutastepst[k]
                    ) * self._Dparam(self.t[k], i + 1)
        # If shape trend is included
        if self.ntrend_sh > 0:
            Dq[
                2
                + self.ngamma0
                + 2 * self.nmu
                + self.ntrend_loc
                + self.nind_loc
                + 2 * self.npsi
                + self.ntrend_sc
                + self.nind_sc
                + 2 * self.ngamma,
                :,
            ] = Depst + Dpsit * Dpsitastepst + Dmut * Dmutastepst * self.t  # gammaT
        # If shape covariates are included
        if self.nind_sh > 0:
            for i in range(self.nind_sh):
                Dq[
                    2
                    + self.ngamma0
                    + 2 * self.nmu
                    + self.ntrend_loc
                    + self.nind_loc
                    + 2 * self.npsi
                    + self.ntrend_sc
                    + self.nind_sc
                    + 2 * self.ngamma
                    + self.ntrend_sh
                    + i,
                    :,
                ] = (
                    Depst + Dpsit * Dpsitastepst + Dmut * Dmutastepst
                ) * self.covariates.iloc[:, self.list_sh[i]]  # gamma_cov

        return Dq

    def _Dmupsiepst(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the derivatives of the standarized maximum with respect to parameters

        Return
        ------
        Dermut : np.ndarray
            Derivative of standarized maximum of location
        Derpsit : np.ndarray
            Derivative of standarized maximum of scale
        Derepst : np.ndarray
            Derivative of standarized maximum of shape
        """

        t = self.t % 1
        nd = len(self.t)

        # Evaluate the parameters
        mut1, psit1, epst = self._evaluate_params(
            beta0=self.beta0,
            beta=self.beta,
            betaT=self.betaT,
            beta_cov=self.beta_cov,
            alpha0=self.alpha0,
            alpha=self.alpha,
            alphaT=self.alphaT,
            alpha_cov=self.alpha_cov,
            gamma0=self.gamma0,
            gamma=self.gamma,
            gammaT=self.gammaT,
            gamma_cov=self.gamma_cov,
            covariates_loc=self.covariates.iloc[:, self.list_loc].values,
            covariates_sc=self.covariates.iloc[:, self.list_sc].values,
            covariates_sh=self.covariates.iloc[:, self.list_sh].values,
        )

        # The values whose shape parameter is almost cero corresponds to the GUMBEL distribution, locate their positions if they exist
        posG = np.where(np.abs(epst) <= 1e-8)[0]
        # The remaining values correspond to WEIBULL or FRECHET
        pos = np.where(np.abs(epst) > 1e-8)[0]
        # The corresponding GUMBEl values are set to 1 to avoid numerical problems, note that for those cases the GUMBEL expressions are used
        epst[posG] = 1

        mut = mut1
        psit = psit1
        mut[pos] = mut1[pos] + psit1[pos] * (self.kt[pos] ** epst[pos] - 1) / epst[pos]
        psit[pos] = psit1[pos] * self.kt[pos] ** epst[pos]
        # Modify the parameters to include Gumbel
        mut[posG] += psit[posG] * np.log(self.kt[posG])

        Dmut = np.ones(nd)
        Dpsit = np.ones(nd)
        Depst = np.ones(nd)

        ## New Derivatives
        Dmutastmut = np.ones_like(self.kt)
        Dmutastpsit = (-1 + self.kt**epst) / epst
        Dmutastepst = (
            psit1 * (1 + (self.kt**epst) * (epst * np.log(self.kt) - 1)) / (epst**2)
        )

        Dpsitastpsit = self.kt**epst
        Dpsitastepst = np.log(self.kt) * psit1 * (self.kt**epst)

        Dmutastpsit[posG] = np.log(self.kt[posG])
        Dmutastepst[posG] = 0

        Dpsitastpsit[posG] = 1
        Dpsitastepst[posG] = 0

        # Derivatives of location, scale and shape parameters respect the model parameters (beta0, beta, ...)
        Dermut = np.zeros((1 + 2 * self.nmu + self.ntrend_loc + self.nind_loc, nd))
        Derpsit = np.zeros((1 + 2 * self.npsi + self.ntrend_sc + self.nind_sc, nd))
        Derepst = np.zeros(
            (self.ngamma0 + 2 * self.ngamma + self.ntrend_sh + self.nind_sh, nd)
        )
        # Jacobian elements related to the location parameters beta0 and beta
        Dermut[0, :] = Dmut * Dmutastmut

        # If location harmonics are included
        if self.nmu > 0:
            for i in range(2 * self.nmu):
                for k in range(len(self.t)):
                    Dermut[i + 1, k] = (
                        Dmut[k] * Dmutastmut[k] * self._Dparam(t[k], i + 1)
                    )

        # Jacobian elements related to the location parameters betaT, beta_cov (equation A.9)
        if self.ntrend_loc > 0:
            Dermut[1 + 2 * self.nmu, :] = Dmut * t * Dmutastmut  # betaT
        if self.nind_loc > 0:
            for i in range(self.nind_loc):
                Dermut[1 + 2 * self.nmu + self.ntrend_loc + i, :] = (
                    Dmut * self.covariates.iloc[:, self.list_loc[i]] * Dmutastmut
                )  # beta_cov_i

        # Jacobian elements related to the scale parameters alpha0, alpha (equation A.7)
        Derpsit[0, :] = psit1 * (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit)  # alpha0
        # If scale harmonic are included
        if self.npsi > 0:
            for i in range(2 * self.npsi):
                for k in range(len(self.t)):
                    Derpsit[i + 1, k] = (
                        self._Dparam(t[k], i + 1)
                        * psit1[k]
                        * (Dpsit[k] * Dpsitastpsit[k] + Dmut[k] * Dmutastpsit[k])
                    )  # alpha
        # Jacobian elements related to the scale parameters alphaT and beta_cov (equation A.10)
        if self.ntrend_sc > 0:
            Derpsit[1 + 2 * self.npsi, :] = (
                (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit) * t * psit1
            )  # alphaT
        if self.nind_sc > 0:
            for i in range(self.nind_sc):
                Derpsit[1 + 2 * self.npsi + self.ntrend_sc + i, :] = (
                    (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit)
                    * self.covariates.iloc[:, self.list_sc[i]]
                    * psit1
                )  # alpha_cov

        # Jacobian elements related to the shape parameters gamma0 and gamma (equation A.10)
        if self.ngamma0 == 1:
            Derepst[0, :] = Depst + Dpsit * Dpsitastepst + Dmut * Dmutastepst
        # If shape harmonics are included
        if self.ngamma > 0:
            for i in range(2 * self.ngamma):
                for k in range(len(self.t)):
                    Derepst[self.ngamma0 + i, k] = (
                        Depst[k] + Dpsit[k] * Dpsitastepst[k] + Dmut[k] * Dmutastepst[k]
                    ) * self._Dparam(t[k], i + 1)
        # If shape trend is included
        if self.ntrend_sh > 0:
            Derepst[self.ngamma0 + 2 * self.ngamma, :] = (
                Depst + Dpsit * Dpsitastepst + Dmut * Dmutastepst * self.t
            )  # gammaT
        # If shape covariates are included
        if self.nind_sh > 0:
            for i in range(self.nind_sh):
                Derepst[self.ngamma0 + 2 * self.ngamma + self.ntrend_sh + i, :] = (
                    Depst + Dpsit * Dpsitastepst + Dmut * Dmutastepst
                ) * self.covariates.iloc[:, self.list_sh[i]]  # gamma_cov

        return Dermut, Derpsit, Derepst

    def _DQuantile(self) -> np.ndarray:
        """
        Calculates the quantile derivative associated with a given parameterization with respect model parameters

        Return
        ------
        Dq : np.ndarray
            Quantile derivative
        """

        t = self.t % 1
        nd = len(self.t)

        # Evaluate the parameters
        mut1, psit1, epst = self._evaluate_params(
            beta0=self.beta0,
            beta=self.beta,
            betaT=self.betaT,
            beta_cov=self.beta_cov,
            alpha0=self.alpha0,
            alpha=self.alpha,
            alphaT=self.alphaT,
            alpha_cov=self.alpha_cov,
            gamma0=self.gamma0,
            gamma=self.gamma,
            gammaT=self.gammaT,
            gamma_cov=self.gamma_cov,
            covariates_loc=self.covariates.iloc[:, self.list_loc].values,
            covariates_sc=self.covariates.iloc[:, self.list_sc].values,
            covariates_sh=self.covariates.iloc[:, self.list_sh].values,
        )

        # The values whose shape parameter is almost cero corresponds to the GUMBEL distribution, locate their positions if they exist
        posG = np.where(np.abs(epst) <= 1e-8)[0]
        # The remaining values correspond to WEIBULL or FRECHET
        pos = np.where(np.abs(epst) > 1e-8)[0]
        # The corresponding GUMBEl values are set to 1 to avoid numerical problems, note that for those cases the GUMBEL expressions are used
        epst[posG] = 1

        mut = mut1
        psit = psit1
        mut[pos] = mut1[pos] + psit1[pos] * (self.kt[pos] ** epst[pos] - 1) / epst[pos]
        psit[pos] = psit1[pos] * self.kt[pos] ** epst[pos]
        # Modify the parameters to include Gumbel
        mut[posG] += psit[posG] * np.log(self.kt[posG])

        Dqmut = np.zeros(nd)
        Dqpsit = np.zeros(nd)
        Dqepst = np.zeros(nd)

        # Derivatives of the quantile function with respect to location, scale and shape parameters
        Dqmut[pos] = np.ones_like(mut[pos])
        Dqpsit[pos] = (
            -(1 - (-np.log(1 - self.quanval) / self.kt[pos]) ** (-epst[pos]))
            / epst[pos]
        )
        Dqepst[pos] = (
            psit[pos]
            * (
                1
                - (-np.log(1 - self.quanval) / self.kt[pos]) ** (-epst[pos])
                * (1 + epst[pos] * np.log(-np.log(1 - self.quanval) / self.kt[pos]))
            )
            / (epst[pos] * epst[pos])
        )

        # Gumbel derivatives
        Dqmut[posG] = np.ones_like(mut[posG])
        Dqpsit[posG] = -np.log(-np.log(1 - self.quanval) / self.kt[posG])
        Dqepst[posG] = np.zeros_like(mut[posG])

        ## New Derivatives
        Dmutastmut = np.ones_like(self.kt)
        Dmutastpsit = (-1 + self.kt**epst) / epst
        Dmutastepst = (
            psit1 * (1 + (self.kt**epst) * (epst * np.log(self.kt) - 1)) / (epst**2)
        )

        Dpsitastpsit = self.kt**epst
        Dpsitastepst = np.log(self.kt) * psit1 * (self.kt**epst)

        Dmutastpsit[posG] = np.log(self.kt[posG])
        Dmutastepst[posG] = 0

        Dpsitastpsit[posG] = 1
        Dpsitastepst[posG] = 0

        # Set the Jacobian to zero matrix
        Dq = np.zeros(
            (
                2
                + self.ngamma0
                + 2 * self.nmu
                + 2 * self.npsi
                + 2 * self.ngamma
                + self.ntrend_loc
                + self.nind_loc
                + self.ntrend_sc
                + self.nind_sc
                + self.ntrend_sh
                + self.nind_sh,
                nd,
            )
        )
        # Jacobian elements related to the location parameters beta0 and beta, equation (A.6) in the paper
        Dq[0, :] = Dqmut * Dmutastmut

        # If location harmonics are included
        if self.nmu > 0:
            for i in range(2 * self.nmu):
                for k in range(len(self.t)):
                    Dq[i + 1, k] = Dqmut[k] * Dmutastmut[k] * self._Dparam(t[k], i + 1)

        # Jacobian elements related to the location parameters betaT, beta_cov (equation A.9)
        if self.ntrend_loc > 0:
            Dq[1 + 2 * self.nmu, :] = Dqmut * t * Dmutastmut  # betaT
        if self.nind_loc > 0:
            for i in range(self.nind_loc):
                Dq[1 + 2 * self.nmu + self.ntrend_loc + i, :] = (
                    Dqmut * self.covariates.iloc[:, self.list_loc[i]] * Dmutastmut
                )  # beta_cov_i

        # Jacobian elements related to the scale parameters alpha0, alpha (equation A.7)
        Dq[1 + 2 * self.nmu + self.ntrend_loc + self.nind_loc, :] = psit1 * (
            Dqpsit * Dpsitastpsit + Dqmut * Dmutastpsit
        )  # alpha0
        # If scale harmonic are included
        if self.npsi > 0:
            for i in range(2 * self.npsi):
                for k in range(len(self.t)):
                    Dq[2 + 2 * self.nmu + self.ntrend_loc + self.nind_loc + i, k] = (
                        self._Dparam(t[k], i + 1)
                        * psit1[k]
                        * (Dqpsit[k] * Dpsitastpsit[k] + Dqmut[k] * Dmutastpsit[k])
                    )  # alpha
        # Jacobian elements related to the scale parameters alphaT and beta_cov (equation A.10)
        if self.ntrend_sc > 0:
            Dq[
                2 + 2 * self.nmu + self.ntrend_loc + self.nind_loc + 2 * self.npsi, :
            ] = (Dqpsit * Dpsitastpsit + Dqmut * Dmutastpsit) * t * psit1  # alphaT
        if self.nind_sc > 0:
            for i in range(self.nind_sc):
                Dq[
                    2
                    + 2 * self.nmu
                    + self.ntrend_loc
                    + self.nind_loc
                    + 2 * self.npsi
                    + self.ntrend_sc
                    + i,
                    :,
                ] = (
                    (Dqpsit * Dpsitastpsit + Dqmut * Dmutastpsit)
                    * self.covariates.iloc[:, self.list_sc[i]]
                    * psit1
                )  # alpha_cov

        # Jacobian elements related to the shape parameters gamma0 and gamma (equation A.10)
        if self.ngamma0 == 1:
            Dq[
                2
                + 2 * self.nmu
                + self.ntrend_loc
                + self.nind_loc
                + 2 * self.npsi
                + self.ntrend_sc
                + self.nind_sc,
                :,
            ] = Dqepst + Dqpsit * Dpsitastepst + Dqmut * Dmutastepst
        # If shape harmonics are included
        if self.ngamma > 0:
            for i in range(self.ngamma):
                for k in range(len(self.t)):
                    Dq[
                        2
                        + self.ngamma0
                        + 2 * self.nmu
                        + self.ntrend_loc
                        + self.nind_loc
                        + 2 * self.npsi
                        + self.ntrend_sc
                        + self.nind_sc
                        + i,
                        k,
                    ] = (
                        Dqepst[k]
                        + Dqpsit[k] * Dpsitastepst[k]
                        + Dqmut[k] * Dmutastepst[k]
                    ) * self._Dparam(t[k], i + 1)
        # If shape trend is included
        if self.ntrend_sh > 0:
            Dq[
                2
                + self.ngamma0
                + 2 * self.nmu
                + self.ntrend_loc
                + self.nind_loc
                + 2 * self.npsi
                + self.ntrend_sc
                + self.nind_sc
                + 2 * self.ngamma,
                :,
            ] = Dqepst + Dqpsit * Dpsitastepst + Dqmut * Dmutastepst * self.t
        # If shape covariates are included
        if self.nind_sh > 0:
            for i in range(self.nind_sh):
                Dq[
                    2
                    + self.ngamma0
                    + 2 * self.nmu
                    + self.ntrend_loc
                    + self.nind_loc
                    + 2 * self.npsi
                    + self.ntrend_sc
                    + self.nind_sc
                    + 2 * self.ngamma
                    + self.ntrend_sh
                    + i,
                    :,
                ] = (
                    Dqepst + Dqpsit * Dpsitastepst + Dqmut * Dmutastepst
                ) * self.covariates.iloc[:, self.list_sh[i]]  # gamma_cov

        return Dq

    def PPplot(self, save=False):
        """
        PP plot

        Parameters
        ----------
        save : bool, default=False
            If True, save the plot in "Figures/"
        """
        # Empirical distribution function value
        Fe = np.arange(1, len(self.xt) + 1) / (len(self.xt) + 1)
        Fm = self._CDFGEVt()
        # Computing the standard errors
        Zm = self._Zstandardt()
        Dwei = self._Dzweibull()
        stdDwei = np.sqrt(np.sum((Dwei.T @ self.invI0) * Dwei.T, axis=1))

        # Sort the data
        Fmsort = np.sort(Fm)
        t_ord = np.argsort(Fm)

        plt.figure(figsize=(10, 6))
        plt.plot([0, 1], [0, 1], self.colors[1])
        plt.plot(
            Fe,
            Fmsort,
            "o",
            markeredgecolor=self.colors[0],
            markerfacecolor=self.colors[0],
            markersize=3,
        )
        # If no covariables or trends, plot the confidence interval
        if (
            self.nind_loc == 0
            and self.nind_sc == 0
            and self.nind_sh == 0
            and self.ntrend_loc == 0
            and self.ntrend_sc == 0
        ):
            plt.fill_between(
                Fe,
                np.exp(
                    -np.exp(
                        -Zm[t_ord]
                        - norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1)
                        * stdDwei[t_ord]
                    )
                ),
                np.exp(
                    -np.exp(
                        -Zm[t_ord]
                        + norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1)
                        * stdDwei[t_ord]
                    )
                ),
                color=self.colors[0],
                alpha=0.3,
            )

            # If dashed lines prefered
            # plt.plot(Fe, np.exp(-np.exp(-Zm[t_ord]+norm.ppf(1-(1-self.quanval)/2, loc=0, scale=1)*stdDwei[t_ord])), linestyle='dashed', color=self.colors[2], markersize=5)
            # plt.plot(Fe, np.exp(-np.exp(-Zm[t_ord]-norm.ppf(1-(1-self.quanval)/2, loc=0, scale=1)*stdDwei[t_ord])), linestyle='dashed', color=self.colors[2], markersize=5)
        plt.title(f"Best model PP plot ({self.var_name})")
        plt.xlabel("Empirical")
        plt.ylabel("Fitted")
        plt.grid(True)
        plt.axis("square")
        plt.margins(x=0.1)
        if save:
            plt.savefig(f"Figures/PPplot_{self.var_name}.png", dpi=300)
        plt.show()

    def _CDFGEVt(self):
        """
        Calculates the GEV distribution function corresponding to the given parameters

        Return
        ------
        F : np.ndarray
            Cumulative distribution function values of Non-stationary GEV for the data
        """

        F = np.zeros(len(self.xt))

        # Evaluate the parameters
        mut1, psit1, epst = self._evaluate_params(
            beta0=self.beta0,
            beta=self.beta,
            betaT=self.betaT,
            beta_cov=self.beta_cov,
            alpha0=self.alpha0,
            alpha=self.alpha,
            alphaT=self.alphaT,
            alpha_cov=self.alpha_cov,
            gamma0=self.gamma0,
            gamma=self.gamma,
            gammaT=self.gammaT,
            gamma_cov=self.gamma_cov,
            covariates_loc=self.covariates.iloc[:, self.list_loc].values,
            covariates_sc=self.covariates.iloc[:, self.list_sc].values,
            covariates_sh=self.covariates.iloc[:, self.list_sh].values,
        )

        # The values whose shape parameter is almost cero corresponds to the GUMBEL distribution, locate their positions if they exist
        posG = np.where(np.abs(epst) <= 1e-8)[0]
        # The remaining values correspond to WEIBULL or FRECHET
        pos = np.where(np.abs(epst) > 1e-8)[0]
        # The corresponding GUMBEl values are set to 1 to avoid numerical problems, note that for those cases the GUMBEL expressions are used
        epst[posG] = 1

        mut = mut1
        psit = psit1
        mut[pos] = mut1[pos] + psit1[pos] * (self.kt[pos] ** epst[pos] - 1) / epst[pos]
        psit[pos] = psit1[pos] * self.kt[pos] ** epst[pos]
        # Modify the parameters to include Gumbel
        mut[posG] += psit[posG] * np.log(self.kt[posG])

        # WEIBULL or FRECHET distribution function
        F[pos] = np.exp(
            -self.kt[pos]
            * (1 + epst[pos] * ((self.xt[pos] - mut[pos]) / psit[pos]))
            ** (-1 / epst[pos])
        )
        # GUMBEL distribution function
        F[posG] = np.exp(
            -self.kt[posG] * np.exp(-((self.xt[posG] - mut[posG]) / psit[posG]))
        )

        return F

    def returnperiod_plot(
        self,
        annualplot: bool = True,
        conf_int: bool = False,
        monthly_plot: bool = False,
        save: bool = False,
    ):
        """
        Funtion to plot the Aggregated Return period plot for each month and the annual Return period

        Parameters
        ----------
        annualplot : bool, default=True
            Whether to plot the annual return period plot
        conf_int : bool, default=False
            Whether to plot the confidence bands for annual return periods
            Heavy computational time
        monthly_plot : bool, default=False
            Wheter to plot the return periods grouped by months
        save : bool, default=False
            Whether to save the plot
        """
        self.logger.debug("Calling: Return Period plot (.returnperiod_plot())")

        Ts = np.array(
            [
                1.1,
                1.5,
                2,
                3,
                4,
                5,
                7.5,
                10,
                15,
                20,
                30,
                40,
                50,
                75,
                100,
                150,
                200,
                500,
                1000,
            ]
        )
        # Ts = np.concatenate(
        #     (np.arange(2, 10, 1), np.arange(10, 100, 10), np.arange(100, 501, 100))
        # )

        nts = len(Ts)
        quanaggrA = np.zeros(nts)
        quanaggr = np.zeros((12, nts))
        stdDqX = np.zeros((12, nts))
        if monthly_plot:
            for i in range(12):
                for j in range(nts):
                    quanaggr[i, j] = self._aggquantile(
                        1 - 1 / Ts[j], i / 12, (i + 1) / 12
                    )[0]
                    # DO NOT COMPUTE CONFIDENCE INTERVAL IN MONTHLY RETURN PERIODS
                    # stdQuan = self._ConfidInterQuanAggregate(
                    #     1 - 1 / Ts[j], i / 12, (i + 1) / 12
                    # )
                    # # stdQuan = 0.1
                    # stdDqX[i, j] = stdQuan * norm.ppf(
                    #     1 - (1 - self.quanval) / 2, loc=0, scale=1
                    # )

        # If annual data has to be plotted
        if annualplot:
            self.logger.debug("ReturnPeriodPlot: Annual return period.")
            for j in range(nts):
                quanaggrA[j] = self._aggquantile(1 - 1 / Ts[j], 0, 1)[0]
                self.logger.debug(
                    f"ReturnPeriodPlot: Annual return period {j} finished."
                )
            # Confidence intervals
            if conf_int:
                stdup = np.zeros(nts)
                stdlo = np.zeros(nts)
                for i in range(nts):
                    stdQuan = self._ConfidInterQuanAggregate(1 - 1 / Ts[i], 0, 1)
                    # stdQuan = 0.1
                    stdup[i] = quanaggrA[i] + stdQuan * norm.ppf(
                        1 - (1 - self.quanval) / 2, loc=0, scale=1
                    )
                    stdlo[i] = quanaggrA[i] - stdQuan * norm.ppf(
                        1 - (1 - self.quanval) / 2, loc=0, scale=1
                    )

        ## Plot the return periods
        # datemax_mod = self.t % 1
        labels = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        colors = [
            "#FF5733",
            "#33FF57",
            "#3357FF",
            "#FF33A8",
            "#33FFF6",
            "#FFD633",
            "#8D33FF",
            "#FF8C33",
            "#33FF8C",
            "#3366FF",
            "#FF3333",
            "#33FF33",
        ]
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot()
        if monthly_plot:
            for i in range(12):
                ax.semilogx(
                    Ts,
                    quanaggr[i, :],
                    color=colors[i],
                    linestyle="-",
                    linewidth=1.2,
                    label=labels[i],
                )

        # Anual return periods
        if annualplot:
            ax.semilogx(Ts, quanaggrA, color="black", linewidth=2, label="Annual")
            # ny = int(np.ceil(self.t[-1]))
            # hmax1 = np.zeros(ny)
            # for j in range(ny):
            #     hmax1[j] = np.max(
            #         self.xt[np.where((self.t >= j) & (self.t < j + 1))[0]]
            #     )

            # Vectorized way
            ny = int(np.ceil(self.t[-1]))
            # Create mask for each year's data
            year_masks = [(self.t >= j) & (self.t < j + 1) for j in range(ny)]
            # Calculate max values using masks and broadcasting
            hmax1 = np.array(
                [
                    np.max(self.xt[mask]) if np.any(mask) else np.nan
                    for mask in year_masks
                ]
            )
            # Remove any NaN values if present
            hmax1 = hmax1[~np.isnan(hmax1)]

            hmaxsort = np.sort(hmax1)
            ProHsmaxsort = np.arange(1, len(hmaxsort) + 1) / (len(hmaxsort) + 1)
            Tapprox = 1 / (1 - ProHsmaxsort)
            # idx = np.where(Tapprox >= 2)[0]
            # ax.semilogx(Tapprox[idx], hmaxsort[idx], "ok", markersize=1.6)
            ax.semilogx(
                Tapprox, hmaxsort, "+", markersize=7, label="Data", color="black"
            )
            if conf_int:
                ax.semilogx(Ts, stdlo, linewidth=1.1, color="gray", linestyle="dashed")
                ax.semilogx(Ts, stdup, linewidth=1.1, color="gray", linestyle="dashed")

        ax.set_xlabel("Return Period (Years)")
        ax.set_ylabel(f"{self.var_name}")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 250, 500])
        ax.set_xticklabels([1, 2, 5, 10, 20, 50, 100, 250, 500])
        ax.set_xlim(left=0.9, right=Ts[-1] + 50)
        # ax.set_ylim(bottom=0)
        # ax.set_title(f"Aggregate Quantiles ({self.var_name})")
        ax.grid(True)
        plt.tight_layout()
        if save:
            plt.savefig(f"Figures/ReturnPeriod_{self.var_name}.png", dpi=300)
        plt.show()

    def _aggquantile(
        self,
        q,
        t0,
        t1,
        beta0=None,
        beta=None,
        alpha0=None,
        alpha=None,
        gamma0=None,
        gamma=None,
        betaT=None,
        alphaT=None,
        gammaT=None,
        beta_cov=None,
        alpha_cov=None,
        gamma_cov=None,
        list_loc=None,
        list_sc=None,
        list_sh=None,
        covariates=None,
    ) -> np.ndarray:
        """
        Function to compute the aggregated quantile between two time stamps

        Parameters
        ----------
        q :
            Quantile value
        t0 :
            Starting point of integration interval
        t1 :
            Ending point of integration interval
        beta0 : default=None
            Stationary part of location parameter
        beta : default=None
            Harmonic part of location parameter
        alpha0 : default=None,
            Stationary part of scale parameter
        alpha : default=None
            Harmonic part of scale parameter
        gamma0: default=None
            Stationary part of shape parameter
        gamma : default=None,
            Harmonic part of shape parameter
        betaT : default=None
            Trend part of location parameter
        alphaT : default=None
            Trend part of scale parameter
        gammaT : default=None
            Trend part of shape parameter
        beta_cov : default=None
            Covariate part of location parameter
        alpha_cov : default=None
            Covariate part of scale parameter
        gamma_cov : default=None
            Covariate part of shape parameter

        Return
        ------
        zqout : np.ndarray
            Aggregated return period
        """
        if beta0 is None:
            beta0 = self.beta0
        if beta is None:
            beta = self.beta
        if alpha0 is None:
            alpha0 = self.alpha0
        if alpha is None:
            alpha = self.alpha
        if gamma0 is None:
            gamma0 = self.gamma0
        if gamma is None:
            gamma = self.gamma
        if betaT is None:
            betaT = self.betaT
        if alphaT is None:
            alphaT = self.alphaT
        if beta_cov is None:
            beta_cov = self.beta_cov
        if alpha_cov is None:
            alpha_cov = self.alpha_cov
        if gamma_cov is None:
            gamma_cov = self.gamma_cov
        if list_loc is None:
            list_loc = self.list_loc
        if list_sc is None:
            list_sc = self.list_sc
        if list_sh is None:
            list_sh = self.list_sh
        if covariates is None:
            covariates = self.covariates

        # Deal with scalars and vectors
        beta_cov = np.atleast_1d(beta_cov)
        alpha_cov = np.atleast_1d(alpha_cov)
        gamma_cov = np.atleast_1d(gamma_cov)

        q = np.array([q])
        t0 = np.array([t0])
        t1 = np.array([t1])
        m = len(q)
        m0 = len(t0)
        m1 = len(t1)
        if m != m0:
            ValueError(
                "Initial quantile aggregated integration time size must be equal than the quantile size"
            )
        if m != m1:
            ValueError(
                "Final quantile aggregated integration time size must be equal than the quantile size"
            )

        # For the required period the mean value of the corresponding covariates is calculated and considered constant for the rest of the study
        if len(self.t) > 0:
            pos = np.where((self.t >= t0) & (self.t <= t1))[0]
            cov_locint = np.zeros_like(beta_cov)
            cov_scint = np.zeros_like(alpha_cov)
            cov_shint = np.zeros_like(gamma_cov)
            if pos.size:
                for i in range(beta_cov.size):
                    cov_locint[i] = np.mean(covariates.iloc[pos, list_loc[i]].values)
                for i in range(alpha_cov.size):
                    cov_scint[i] = np.mean(covariates.iloc[pos, list_sc[i]].values)
                for i in range(gamma_cov.size):
                    cov_shint[i] = np.mean(covariates.iloc[pos, list_sh[i]].values)
        else:
            cov_locint = np.zeros(beta_cov.size)
            cov_scint = np.zeros(alpha_cov.size)
            cov_shint = np.zeros(gamma_cov.size)

        # Require quantile
        zqout = np.zeros(m)

        media, _ = quad(
            lambda x: self._parametro(
                beta0,
                beta,
                betaT,
                beta_cov,
                covariates.iloc[:, list_loc].values,
                cov_locint,
                self.t,
                x,
            ),
            0,
            1,
        )
        std, _ = quad(
            lambda x: np.exp(
                self._parametro(
                    alpha0,
                    alpha,
                    alphaT,
                    alpha_cov,
                    self.covariates.iloc[:, self.list_sc].values,
                    cov_scint,
                    self.t,
                    x,
                )
            ),
            0,
            1,
        )

        a = media - 10
        b = media + 20

        for il in range(m):
            # function of z whose root we want
            def F(z):
                self.logger.debug("Inicio quad()")
                integ, _ = quad(
                    lambda x: self._fzeroquanint(
                        x,
                        z,
                        q[il],
                        cov_locint,
                        cov_scint,
                        cov_shint,
                        beta0,
                        beta,
                        alpha0,
                        alpha,
                        gamma0,
                        gamma,
                        betaT,
                        alphaT,
                        gammaT,
                        beta_cov,
                        alpha_cov,
                        gamma_cov,
                        self.t,
                        self.kt,
                    ),
                    float(t0[il]),
                    float(t1[il]),
                    epsabs=1e-5,
                    epsrel=1e-5,
                )
                self.logger.debug("Fin quad()")
                return integ + np.log(q[il]) / 12.0

            try:
                # sol = root_scalar(
                #     F,
                #     x0=media,
                #     x1=media + 1.0,  # secant starting points
                #     method="secant",
                #     xtol=1e-6,
                #     rtol=1e-6,
                #     maxiter=200,
                # )

                sol = root_scalar(
                    F,
                    bracket=(a, b),
                    method="toms748",
                    xtol=1e-6,
                    rtol=1e-6,
                    maxiter=100,
                )
                if sol.converged:
                    if abs(F(sol.root)) < 1e-2:
                        zqout[il] = sol.root
                    else:
                        zqout[il] = np.nan  # "False zero" check
                else:
                    zqout[il] = np.nan
            except Exception:
                zqout[il] = np.nan

        return zqout

    def _fzeroquanint(
        self,
        t,
        zq,
        q,
        indicesint,
        indices2int,
        indices3int,
        beta0,
        beta,
        alpha0,
        alpha,
        gamma0,
        gamma,
        betaT,
        alphaT,
        gammaT,
        beta_cov,
        alpha_cov,
        gamma_cov,
        times=None,
        ktold=None,
    ) -> np.ndarray:
        """
        Auxiliar function to solve the quantile

        Return
        ------
        zn : np.ndarray
        """
        self.logger.debug("Inicio _fzeroquanint()")
        if ktold is None:
            ktold = self.kt

        # Evaluate the location parameter at each time t as a function of the actual values of the parameters given by p
        mut1 = self._parametro(
            beta0,
            beta,
            betaT,
            beta_cov,
            self.covariates.iloc[:, self.list_loc].values,
            indicesint,
            self.t,
            t,
        )
        # Evaluate the scale parameter at each time t as a function of the actual values of the parameters given by p
        psit1 = np.exp(
            self._parametro(
                alpha0,
                alpha,
                alphaT,
                alpha_cov,
                self.covariates.iloc[:, self.list_sc].values,
                indices2int,
                self.t,
                t,
            )
        )
        # Evaluate the sahpe parameter at each time t as a function of the actual values of the parameters given by p
        epst = self._parametro(
            gamma0,
            gamma,
            gammaT,
            gamma_cov,
            self.covariates.iloc[:, self.list_sh].values,
            indices3int,
            self.t,
            t,
        )

        # The values whose shape parameter is almost cero corresponds to the GUMBEL distribution, locate their positions if they exist
        posG = np.where(np.abs(epst) <= 1e-8)[0]
        # The remaining values correspond to WEIBULL or FRECHET
        pos = np.where(np.abs(epst) > 1e-8)[0]
        # The corresponding GUMBEl values are set to 1 to avoid numerical problems, note that for those cases the GUMBEL expressions are used
        epst[posG] = 1

        if times is not None:
            kt2 = np.interp(
                t, np.asarray(self.t, float), np.asarray(ktold, float)
            ).flatten()
        else:
            kt2 = np.ones_like(mut1)

        mut = mut1
        psit = psit1
        mut[pos] = mut1[pos] + psit1[pos] * (kt2[pos] ** epst[pos] - 1) / epst[pos]
        psit[pos] = psit1[pos] * kt2[pos] ** epst[pos]
        # Modify the parameters to include Gumbel
        mut[posG] += psit[posG] * np.log(kt2[posG])

        # Evaluate the auxiliary variable
        xn = (zq - mut) / psit
        z = 1 + epst * xn
        # Since the z-values must be greater than zero in order to avoid numerical problems their values are set to be greater than 1e-4
        z = np.maximum(1e-8, z)
        zn = z ** (-1 / epst)
        # GUMBEL case
        zn[posG] = np.exp(-xn[posG])

        self.logger.debug("Fin _fzeroquanint()")
        return zn

    def _fzeroderiquanint(
        self,
        t,
        zq,
        q,
        indicesint,
        indices2int,
        indices3int,
        beta0,
        beta,
        alpha0,
        alpha,
        gamma0,
        gamma,
        betaT,
        alphaT,
        gammaT,
        beta_cov,
        alpha_cov,
        gamma_cov,
        times=None,
        ktold=None,
    ):
        """
        Auxiliar Function to solve the quantile derivative

        Return
        ------
        zn : np.ndarray
        """
        if ktold is None:
            ktold = self.kt

        # Evaluate the location parameter at each time t as a function of the actual values of the parameters given by p
        mut1 = self._parametro(
            beta0,
            beta,
            betaT,
            beta_cov,
            self.covariates.iloc[:, self.list_loc].values,
            indicesint,
            self.t,
            t,
        )
        # Evaluate the scale parameter at each time t as a function of the actual values of the parameters given by p
        psit1 = np.exp(
            self._parametro(
                alpha0,
                alpha,
                alphaT,
                alpha_cov,
                self.covariates.iloc[:, self.list_sc].values,
                indices2int,
                self.t,
                t,
            )
        )
        # Evaluate the sahpe parameter at each time t as a function of the actual values of the parameters given by p
        epst = self._parametro(
            gamma0,
            gamma,
            gammaT,
            gamma_cov,
            self.covariates.iloc[:, self.list_sh].values,
            indices3int,
            self.t,
            t,
        )

        # The values whose shape parameter is almost cero corresponds to the GUMBEL distribution, locate their positions if they exist
        posG = np.where(np.abs(epst) <= 1e-8)[0]
        # The remaining values correspond to WEIBULL or FRECHET
        pos = np.where(np.abs(epst) > 1e-8)[0]
        # The corresponding GUMBEl values are set to 1 to avoid numerical problems, note that for those cases the GUMBEL expressions are used
        epst[posG] = 1

        if times is not None:
            kt2 = np.interp(
                t, np.asarray(self.t, float), np.asarray(ktold, float)
            ).flatten()
        else:
            kt2 = np.ones_like(mut1)

        mut = mut1
        psit = psit1
        mut[pos] = mut1[pos] + psit1[pos] * (kt2[pos] ** epst[pos] - 1) / epst[pos]
        psit[pos] = psit1[pos] * kt2[pos] ** epst[pos]
        # Modify the parameters to include Gumbel
        mut[posG] += psit[posG] * np.log(kt2[posG])

        # Evaluate the auxiliary variable
        xn = (zq - mut) / psit
        z = 1 + epst * xn
        # Since the z-values must be greater than zero in order to avoid numerical problems their values are set to be greater than 1e-4
        z = np.maximum(1e-8, z)
        zn = z ** (-1 - 1 / epst) / psit
        # GUMBEL case
        zn[posG] = -np.exp(-xn[posG]) / psit[posG]

        return zn

    def _ConfidInterQuanAggregate(self, q, t0, t1) -> np.ndarray:
        """
        Auxiliar function to compute the std for the aggregated quantiles

        Return
        ------
        stdQuan : np.ndarray
            Standard deviation of quantile
        """
        # Total length of the data
        n = (
            2
            + self.ngamma0
            + 2 * self.nmu
            + 2 * self.npsi
            + 2 * self.ngamma
            + self.ntrend_loc
            + self.nind_loc
            + self.ntrend_sc
            + self.nind_sc
            + self.ntrend_sh
            + self.nind_sh
        )

        # Initialize the Jacobian
        jacob = np.zeros(n)

        epsi = 1e-4

        # beta0 derivative
        aux = 0
        jacob[aux] = (
            self._aggquantile(q, t0, t1, beta0=self.beta0 * (1 + epsi))[0]
            - self._aggquantile(q, t0, t1, beta0=self.beta0 * (1 - epsi))[0]
        ) / (2 * self.beta0 * epsi)

        # beta derivatives
        if self.nmu != 0:
            for i in range(2 * self.nmu):
                aux += 1
                beta1 = self.beta
                beta2 = self.beta
                beta2[i] = self.beta[i] * (1 + epsi)
                beta1[i] = self.beta[i] * (1 - epsi)
                jacob[aux] = (
                    self._aggquantile(q, t0, t1, beta=beta2)[0]
                    - self._aggquantile(q, t0, t1, beta=beta1)[0]
                ) / (2 * self.beta[i] * epsi)

        # betaT derivative
        if self.ntrend_loc != 0:
            aux += 1
            jacob[aux] = (
                self._aggquantile(q, t0, t1, betaT=self.betaT * (1 + epsi))[0]
                - self._aggquantile(q, t0, t1, betaT=self.betaT * (1 - epsi))[0]
            ) / (2 * self.betaT * epsi)

        # beta_cov derivative
        if self.nind_loc != 0:
            for i in range(self.nind_loc):
                aux += 1
                if self.beta_cov[i] != 0:
                    beta_covlb = self.beta_cov
                    beta_covub = self.beta_cov
                    beta_covlb[i] = self.beta_cov[i] * (1 + epsi)
                    beta_covub[i] = self.beta_cov[i] * (1 - epsi)
                    jacob[aux] = (
                        self._aggquantile(
                            q,
                            t0,
                            t1,
                            beta_cov=np.atleast_1d(beta_covlb[i]),
                            list_loc=np.atleast_1d(self.list_loc[i]),
                        )[0]
                        - self._aggquantile(
                            q,
                            t0,
                            t1,
                            beta_cov=np.atleast_1d(beta_covlb[i]),
                            list_loc=np.atleast_1d(self.list_loc[i]),
                        )[0]
                    ) / (2 * self.beta_cov[i] * epsi)
                else:
                    jacob[aux] = 0

        # alpha0 derivative
        aux = 0
        jacob[aux] = (
            self._aggquantile(q, t0, t1, alpha0=self.alpha0 * (1 + epsi))[0]
            - self._aggquantile(q, t0, t1, alpha0=self.alpha0 * (1 - epsi))[0]
        ) / (2 * self.alpha0 * epsi)

        # alpha derivatives
        if self.npsi != 0:
            for i in range(2 * self.npsi):
                aux += 1
                alpha1 = self.alpha
                alpha2 = self.alpha
                alpha2[i] = self.alpha[i] * (1 + epsi)
                alpha1[i] = self.alpha[i] * (1 - epsi)
                jacob[aux] = (
                    self._aggquantile(q, t0, t1, alpha=alpha2)[0]
                    - self._aggquantile(q, t0, t1, alpha=alpha1)[0]
                ) / (2 * self.alpha[i] * epsi)

        # alphaT derivative
        if self.ntrend_sc != 0:
            aux += 1
            jacob[aux] = (
                self._aggquantile(q, t0, t1, alphaT=self.alphaT * (1 + epsi))[0]
                - self._aggquantile(q, t0, t1, alphaT=self.alphaT * (1 - epsi))[0]
            ) / (2 * self.alphaT * epsi)

        # alpha_cov derivative
        if self.nind_sc != 0:
            for i in range(self.nind_sc):
                aux += 1
                if self.alpha_cov[i] != 0:
                    alpha_covlb = self.alpha_cov
                    alpha_covub = self.alpha_cov
                    alpha_covlb[i] = self.alpha_cov[i] * (1 + epsi)
                    alpha_covub[i] = self.alpha_cov[i] * (1 - epsi)
                    jacob[aux] = (
                        self._aggquantile(
                            q,
                            t0,
                            t1,
                            alpha_cov=np.atleast_1d(alpha_covlb[i]),
                            list_sc=np.atleast_1d(self.list_sc[i]),
                        )[0]
                        - self._aggquantile(
                            q,
                            t0,
                            t1,
                            alpha_cov=np.atleast_1d(alpha_covlb[i]),
                            list_sc=np.atleast_1d(self.list_sc[i]),
                        )[0]
                    ) / (2 * self.alpha_cov[i] * epsi)
                else:
                    jacob[aux] = 0

        # gamma0 derivative
        if self.ngamma0 != 0:
            aux += 1
            jacob[aux] = (
                self._aggquantile(q, t0, t1, gamma0=self.gamma0 * (1 + epsi))[0]
                - self._aggquantile(q, t0, t1, gamma0=self.gamma0 * (1 - epsi))[0]
            ) / (2 * self.gamma0 * epsi)

        if self.ngamma != 0:
            for i in range(2 * self.ngamma):
                aux += 1
                gamma1 = self.gamma
                gamma2 = self.gamma
                gamma2[i] = self.gamma[i] * (1 + epsi)
                gamma1[i] = self.gamma[i] * (1 - epsi)
                jacob[aux] = (
                    self._aggquantile(q, t0, t1, gamma=gamma2)[0]
                    - self._aggquantile(q, t0, t1, gamma=gamma1)[0]
                ) / (2 * self.gamma[i] * epsi)

        # gammaT derivative
        if self.ntrend_sh != 0:
            aux += 1
            jacob[aux] = (
                self._aggquantile(q, t0, t1, gammaT=self.gammaT * (1 + epsi))[0]
                - self._aggquantile(q, t0, t1, alphaT=self.gammaT * (1 - epsi))[0]
            ) / (2 * self.gammaT * epsi)

        if self.nind_sh != 0:
            for i in range(self.nind_sh):
                aux += 1
                if self.gamma_cov[i] != 0:
                    gamma_covlb = self.gamma_cov
                    gamma_covub = self.gamma_cov
                    gamma_covlb[i] = self.gamma_cov[i] * (1 + epsi)
                    gamma_covub[i] = self.gamma_cov[i] * (1 - epsi)
                    jacob[aux] = (
                        self._aggquantile(
                            q,
                            t0,
                            t1,
                            gamma_cov=gamma_covlb[i],
                            list_sh=np.atleast_1d(self.list_sh[i]),
                        )[0]
                        - self._aggquantile(
                            q,
                            t0,
                            t1,
                            gamma_cov=gamma_covub[i],
                            list_sh=np.atleast_1d(self.list_sh[i]),
                        )[0]
                    ) / (2 * self.gamma_cov[i] * epsi)
                else:
                    jacob[aux] = 0

        # Computing the standard deviations for the quantiles
        stdQuan = np.sqrt(jacob.T @ self.invI0 @ jacob)

        return stdQuan

    def summary(self):
        """
        Print a summary of the fitted model, including parameter estimates, standard errors and fit statistics.
        """
        std_params = self.std_params
        param_idx = 0
        z_norm = norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1)

        print(f"\nFitted Time-Dependent GEV model for {self.var_name}")
        print("=" * 70)

        # Header format
        param_header = "Parameter".ljust(15)
        estimate_header = "Estimate".rjust(12)
        se_header = "Std. Error".rjust(15)
        ci_header = f"{self.quanval * 100:.0f}% CI".center(25)
        header = f"{param_header} {estimate_header} {se_header} {ci_header}"

        print("\nLocation Parameters")
        print("-" * 70)
        print(header)
        print("-" * 70)

        # Parameter line format
        def format_line(name, value, std_err):
            ci_low = value - std_err * z_norm
            ci_up = value + std_err * z_norm
            return f"{name:<15} {value:>12.4f} {std_err:>15.4f} {f'[{ci_low:>8.4f},{ci_up:>8.4f}]':>25}"

        # Location parameters
        print(format_line("Beta0", self.beta0, std_params[param_idx]))
        param_idx += 1

        for i in range(self.nmu):
            print(
                format_line(
                    f"Beta{i + 1} (sin)", self.beta[2 * i], std_params[param_idx]
                )
            )
            param_idx += 1
            print(
                format_line(
                    f"Beta{i + 1} (cos)", self.beta[2 * i + 1], std_params[param_idx]
                )
            )
            param_idx += 1

        if self.ntrend_loc > 0:
            print(format_line("BetaT", self.betaT, std_params[param_idx]))
            param_idx += 1

        for i in range(self.nind_loc):
            print(
                format_line(
                    f"{self.covariates.columns[self.list_loc[i]]}",
                    self.beta_cov[i],
                    std_params[param_idx],
                )
            )
            param_idx += 1

        print("\nScale Parameters")
        print("-" * 70)
        print(header)
        print("-" * 70)

        print(format_line("Alpha0", self.alpha0, std_params[param_idx]))
        param_idx += 1

        for i in range(self.npsi):
            print(
                format_line(
                    f"Alpha{i + 1} (sin)", self.alpha[2 * i], std_params[param_idx]
                )
            )
            param_idx += 1
            print(
                format_line(
                    f"Alpha{i + 1} (cos)", self.alpha[2 * i + 1], std_params[param_idx]
                )
            )
            param_idx += 1

        if self.ntrend_sc > 0:
            print(format_line("AlphaT", self.alphaT, std_params[param_idx]))
            param_idx += 1

        for i in range(self.nind_sc):
            print(
                format_line(
                    f"{self.covariates.columns[self.list_sc[i]]}",
                    self.alpha_cov[i],
                    std_params[param_idx],
                )
            )
            param_idx += 1

        print("\nShape Parameters")
        print("-" * 70)
        print(header)
        print("-" * 70)

        if self.ngamma0 > 0:
            print(format_line("Gamma0", self.gamma0, std_params[param_idx]))
            param_idx += 1

        for i in range(self.ngamma):
            print(
                format_line(
                    f"Gamma{i + 1} (sin)", self.gamma[2 * i], std_params[param_idx]
                )
            )
            param_idx += 1
            print(
                format_line(
                    f"Gamma{i + 1} (cos)", self.gamma[2 * i + 1], std_params[param_idx]
                )
            )
            param_idx += 1

        if self.ntrend_sh > 0:
            print(format_line("GammaT", self.gammaT, std_params[param_idx]))
            param_idx += 1

        for i in range(self.nind_sh):
            print(
                format_line(
                    f"{self.covariates.columns[self.list_sh[i]]}",
                    self.gamma_cov[i],
                    std_params[param_idx],
                )
            )
            param_idx += 1

        print("\nFit Statistics")
        print("-" * 70)
        stats_width = 30
        print(
            f"{'Log-likelihood:':<{stats_width}} {-self.fit_result['negloglikelihood']:>.4f}"
        )
        print(f"{'AIC:':<{stats_width}} {self.fit_result['AIC']:>.4f}")
        print(f"{'Number of parameters:':<{stats_width}} {self.fit_result['n_params']}")
        print(f"Success: {self.fit_result['success']}")
        print(f"Message: {self.fit_result['message']}")


def bsimp(fun, a, b, n=None, epsilon=1e-8, trace=0):
    """
    BSIMP   Numerically evaluate integral, low order method.
    I = BSIMP('F',A,B) approximates the integral of F(X) from A to B
    within a relative error of 1e-3 using an iterative
    Simpson's rule.  'F' is a string containing the name of the
    function.  Function F must return a vector of output values if given
    a vector of input values.%
    I = BSIMP('F',A,B,EPSILON) integrates to a total error of EPSILON.  %
    I = BSIMP('F',A,B,N,EPSILON,TRACE,TRACETOL) integrates to a
    relative error of EPSILON,
    beginning with n subdivisions of the interval [A,B],for non-zero
    TRACE traces the function
    evaluations with a point plot.
    [I,cnt] = BSIMP(F,a,b,epsilon) also returns a function evaluation count.%
    Roberto Minguez Solana%   Copyright (c) 2001 by Universidad de Cantabria
    """
    if n is None:
        n = int(365 * 8 * (b - a))

    # The number of initial subintervals must be pair
    if n % 2 != 0:
        n = n + 1

    # Step 1:
    h = (b - a) / n

    # Step 3:
    x = np.linspace(a, b, n + 1)
    y = fun(x)

    # print("y size: ", y.size)
    # print("n: ", n)
    # TODO:
    # if trace:

    auxI = 0
    ainit = y[0] + y[n]
    auxI1 = 0
    auxI2 = 0

    auxI1 = auxI1 + sum(y[1:n:2])
    auxI2 = auxI2 + sum(y[2:n:2])

    cnt = n

    # Step 4
    integral = (ainit + 4 * auxI1 + 2 * auxI2) * h / 3
    auxtot = auxI1 + auxI2

    # Step 5
    integral1 = integral + epsilon * 2
    j = 2

    # Step 6
    error = 1
    while error > epsilon:
        cnt = cnt + int(j * n // 2)
        # Step 7
        integral1 = integral
        # Step 8
        x = np.linspace(a + h / j, b - h / j, int(j * n // 2))
        y = fun(x)

        auxI = auxI + sum(y[0 : int(j * n // 2)])

        # Step 9
        integral = (ainit + 4 * auxI + 2 * auxtot) * h / (3 * j)

        # Step 10
        j = j * 2

        # Step 11
        auxtot = auxtot + auxI
        auxI = 0
        # Error
        if np.abs(integral1) > 1e-5:
            error = np.abs((integral - integral1) / integral1)
        else:
            error = np.abs(integral - integral1)

    return integral
