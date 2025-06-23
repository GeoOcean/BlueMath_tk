from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm

from ..core.models import BlueMathModel


class NonStatGEV(BlueMathModel):
    """
    Class to implement the Non-Stationary GEV including trends and/or covariates
    in the location, scale and shape parameters. This methodology selects the
    covariates and trends based on which of them minimize the Akaike Information Criteria (AIC)

    Parameters
    ----------
    xt : np.ndarray
        Data to fit Non Stationary GEV.
    t : np.ndarray, default=None.
        Time associated to the data.
    covariates: np.ndarray | pd.DataFrame, default=None.
        Covariates to include for location, scale and shape parameters.
    kt : np.ndarray, default=None.
        Frequency of blocks, if None, it is assumed to be 1.
    trends: bool, defaul=False.
        Whether trends should be included, if so, t must be passed.
    quanval : float, default=0.95.
        Confidence interval value

    Methods
    ----------
    fit:
        Fit the Non-Stationary GEV with desired Trends and Covariates.
    auto_adjust:
        Automatically selects the best covariates and trends based on AIC.

    """

    def __init__(
        self,
        xt: np.ndarray,
        t: np.ndarray,
        covariates: np.ndarray | pd.DataFrame = None,
        trends: bool = False,
        kt: np.ndarray = None,
        quanval: float = 0.95,
    ):
        """
        Initiliaze the NonStationary GEV.
        """

        # Initialize arguments
        self.xt = xt
        self.t = t
        self.covariates = covariates
        self.trends = trends
        if kt is not None:
            self.kt = kt
        else:
            self.kt = np.ones_like(xt) 
        self.quanval = quanval

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

    def auto_adjust(self, max_iter: int = 1000) -> dict:
        """
        This method automatically select and calculate the parameters which minimize the AIC related to
        Non-Stationary GEV distribution using the Maximum Likelihood method within an iterative scheme,
        including one parameter at a time based on a perturbation criteria.
        The process is repeated until no further improvement in the objective function is achieved.

        Parameters
        ----------
        max_iter : int, default=1000
            Number of iteration in optimization process.

        Return
        ----------
        fit_result : dict
            Dictionary with the optimal parameters and values related to the Non-Stationary GEV distribution.
            The keys of the dictionary are:
            - beta0, beta, betaT, beta_cov: Location parameters (intercept, harmonic, trend, covariates)
            - alpha0, alpha, alphaT, alpha_cov: Scale parameters (intercept, harmonic, trend, covariates)
            - gamma0, gamma, gammaT, gamma_cov: Shape parameters (intercept, harmonic, trend, covariates)
            - negloglikelihood: Negative log-likelihood value at the optimal solution
            - hessian: Hessian matrix of the log-likelihood function at the optimal solution
            - AIC: Akaike Information Criterion value at the optimal solution

        TODO: CHANGE THIS
        Output:
            - beta0     -> Optimal constant parameter related to location
            - beta      -> Optimal harmonic vector associated with location
            - alpha0    -> Optimal constant parameter related to scale
            - alpha     -> Optimal harmonic vector associated with scale
            - gamma0    -> Optimal constant parameter related to shape
            - gamma     -> Optimal harmonic vector associated with shape
            - betaT     -> Optimal location trend parameter
            - auxvarphi -> Optimal location covariate vector
            - betaT     -> Optimal scale trend parameter
            - auxvarphi -> Optimal scale covariate vector
            - grad      -> Gradient of the log-likelihood function with the sign changed at the optimal solution
            - hessian   -> Hessian of the log-likelihood function with the sign changed at the optimal solution
            - popt      -> vector including the optimal values for the parameters in the following order:
                        beta0, beta, betaT, varphi, alpha0, alpha, betaT2, varphi2, gamma0, gamma
            - stdpara   -> vector including the standard deviation of the optimal values for the parameters in the following order:
                        beta0, beta, betaT, varphi, alpha0, alpha, betaT2, varphi2, gamma0, gamma
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
        print("Starting Harmonic iterative process")
        for iter in range(self.max_iter):
            # TODO: AÑADIR EL PROCESO ARMONICO
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
            self.AIC_iter[iter] = self._AIC(-fit_result["negloglikelihood"], n_params)

            ### Step 4: Sensitivity of optimal loglikelihood respect to possible additional harmonics
            # for the location, scale and shape parameters.
            # Note that the new parameter values are set to zero since derivatives do not depend on them
            fit_result_aux = fit_result.copy()
            # Location
            if fit_result["beta"] is not None:
                fit_result_aux["beta"] = np.concatenate((fit_result["beta"], [0, 0]))
            else:
                fit_result_aux["beta"] = np.array([0, 0])
            # Scale
            if fit_result["alpha"] is not None:
                fit_result_aux["alpha"] = np.concatenate((fit_result["alpha"], [0, 0]))
            else:
                fit_result_aux["alpha"] = np.array([0, 0])
            # Shape
            if fit_result["gamma"] is not None:
                fit_result_aux["gamma"] = np.concatenate((fit_result["gamma"], [0, 0]))
            else:
                fit_result_aux["gamma"] = np.array([0, 0])

            auxf, auxJx, auxHxx = self._loglikelihood(
                **fit_result_aux
            )  # TODO: ADD THE PARAMETERS

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
            max_val = np.abs(
                auxJx[2 * nmu : 2 * nmu + 2].T
                @ auxI0[2 * nmu : 2 * nmu + 2, 2 * nmu : 2 * nmu + 2]
                @ auxJx[2 * nmu : 2 * nmu + 2]
            )
            # TODO: AÑADIR BIEN LAS COVARIABLES EN EL PARAMETRO DE FORMA
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

            # TODO: AÑADIR BIEN LAS COVARIABLES EN EL PARAMETRO DE FORMA
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

            if iter > 0:
                if self.AIC_iter[iter] >= self.AIC_iter[iter - 1]:
                    model = modelant
                    self.AICini = self.AIC_iter[iter - 1]
                    loglikeobjITini = self.loglike_iter[iter - 1]
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

        self._update_params(fit_result)
        self.nmu = nmu
        self.npsi = npsi
        self.ngamma = ngamma

        ######### COVARIATES Iterative process #########
        nrows, nind_cov = self.covariates.shape  # Number of covariates

        # Auxiliar variables related to location parameter
        beta_cov = np.asarray([])
        list_loc = []  # List of covariates for location
        nind_loc = (
            0  # TODO: IT IS ZERO BY DEFAULT, CHECK IF IT POSSIBLE TO REMOVE THIS LINE
        )
        auxcov_loc = self.covariates.iloc[:, list_loc].values
        # Auxiliar variables related to scale parameter
        alpha_cov = np.asarray([])
        list_sc = []
        nind_sc = (
            0  # TODO: IT IS ZERO BY DEFAULT, CHECK IF IT POSSIBLE TO REMOVE THIS LINE
        )
        auxcov_sc = self.covariates.iloc[:, list_sc].values
        # Auxiliar variables related to shape parameter
        gamma_cov = np.asarray([])
        list_sh = []
        nind_sh = (
            0  # TODO: IT IS ZERO BY DEFAULT, CHECK IF IT POSSIBLE TO REMOVE THIS LINE
        )
        auxcov_sh = self.covariates.iloc[:, list_sh].values

        if self.covariates is None:
            print("No covariates provided, skipping Covariates iterative process")
        else:
            print("Starting Covariates iterative process")
            for iter in range(self.niter_harm + 1, self.max_iter):
                self.ngamma0 = 1

                ### Step 9: Calculate the sensitivities of the optimal log-likelihood objective function with respect to possible
                # additional covariates for the location and  scale parameters
                auxf, auxJx, auxHxx = self._loglikelihood()  # TODO: ADD THE PARAMETERS

                # Step 10: Include in the parameter vector the corresponding covariate
                auxI0 = np.linalg.inv(-auxHxx)
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
                        + 2 * ngamma : 2
                        + self.ngamma0
                        + 2 * nmu
                        + ntrend_loc
                        + nind_cov
                        + 2 * npsi
                        + ntrend_sc
                        + nind_cov
                        + 2 * ngamma
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
                auxcov_loc = self.covariates.iloc[:, list_loc].values
                auxcov_sc = self.covariates.iloc[:, list_sc].values
                auxcov_sh = self.covariates.iloc[:, list_sh].values

                ### Step 11: Obtain the maximum-likelihood estimators for the selected parameters and
                # calculate the Akaike Information criterion objective function AIC
                # TODO: DEFINE PROPERLY
                # concatvalues = [popt[0:1 + 2 * nmu], varphiini_loc, popt[1 + 2 * nmu : 2 + 2 * nmu + 2 * npsi], varphiini_sc, np.zeros(self.ngamma0), np.zeros(2 * ngamma), varphiini_sh]
                # pini = np.concatenate([np.asarray(v) for v in concatvalues if v is not None])
                fit_result = self._fit(self.nmu, self.npsi, self.ngamma)

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

                if self.AIC_iter[iter] <= self.AIC_iter[iter - 1]:
                    # Update the parameters
                    self._update_params(fit_result)
                    self.list_loc = list_loc
                    self.list_sc = list_sc
                    self.list_sh = list_sh
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

                    self.niter_cov = iter - self.niter_harm
                    self.nit = iter

                    self.list_loc = list_loc
                    self.list_sc = list_sc
                    self.list_sh = list_sh
                    break

        print("End of Covariates iterative process")
        print("Covariates AIC:", self.AICini, "\n")

        ######### TRENDS Iterative process #########
        if self.trends:
            print("Starting Trends process")
            # Location trends
            ntrend_loc = 1

            # concatvalues = [popt[0:1 + 2 * nmu], np.zeros(ntrend_loc), varphiini_loc, popt[1 + 2 * nmu : 2 + 2 * nmu + 2 * npsi], varphiini_sc, np.zeros(self.ngamma0), np.zeros(2 * ngamma), varphiini_sh]
            # pini = np.concatenate([np.asarray(v) for v in concatvalues if v is not None])
            fit_result = self._fit()

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
                -fit_result["negloglikelihood"], n_params
            )
            self.loglike_iter[self.nit + 1] = -fit_result["negloglikelihood"]

            if self.AIC_iter[self.nit + 1] < self.AIC_iter[self.nit]:
                self.AICini = self.AIC_iter[self.nit + 1]
                print("Location trend is significative")
                print("Location trend AIC: ", self.AICini)
                # Update the parameters
                self._update_params(fit_result)
                self.ntrend_loc = ntrend_loc
            else:
                print("Location trend is NOT significative")
                self.ntrend_loc = 0

            # Scale trend
            ntrend_sc = 1

            # concatvalues = [popt[0:1 + 2 * nmu], np.zeros(ntrend_loc), varphiini_loc, popt[1 + 2 * nmu : 2 + 2 * nmu + 2 * npsi], varphiini_sc, np.zeros(self.ngamma0), np.zeros(2 * ngamma), varphiini_sh]
            # pini = np.concatenate([np.asarray(v) for v in concatvalues if v is not None])
            fit_result = self._fit()

            n_params = (
                2
                + self.ngamma0
                + 2 * nmu
                + 2 * npsi
                + 2 * ngamma
                + self.ntrend_loc
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
                self._update_params(fit_result)
                self.ntrend_sc = ntrend_sc
            else:
                print("Scale trend is NOT significative")
                self.ntrend_sc = 0

            # Shape trends
            ntrend_sh = 1

            # concatvalues = [popt[0:1 + 2 * nmu], np.zeros(ntrend_loc), varphiini_loc, popt[1 + 2 * nmu : 2 + 2 * nmu + 2 * npsi], varphiini_sc, np.zeros(self.ngamma0), np.zeros(2 * ngamma), varphiini_sh]
            # pini = np.concatenate([np.asarray(v) for v in concatvalues if v is not None])
            fit_result = self._fit()

            n_params = (
                2
                + self.ngamma0
                + 2 * nmu
                + 2 * npsi
                + 2 * ngamma
                + self.ntrend_loc
                + nind_loc
                + self.ntrend_sc
                + nind_sc
                + ntrend_sh
                + nind_sh
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
                self._update_params(fit_result)
                self.ntrend_sc = ntrend_sh
            else:
                print("Shape trend is NOT significative")
                self.ntrend_sh = 0

        return fit_result

    def _fit(
        self,
        nmu=0,
        npsi=0,
        ngamma=0,
        list_loc=[],
        ntrend_loc=0,
        list_sc=[],
        ntrend_sc=0,
        list_sh=[],
        ntrend_sh=0,
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
        """

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
        x_ini = np.zeros(n_params)
        x_ini[0] = np.mean(self.xt)  # Initial value for intercept location
        x_ini[1] = np.log(np.std(self.xt))  # Initial value for intercept scale
        if self.ngamma0 == 1:
            x_ini[2] = 0.1  # Initial value for intercept shape
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
                + ngamma : 2
                + self.ngamma0
                + nmu
                + npsi
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc
                + ngamma
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
                + ngamma : 2
                + self.ngamma0
                + nmu
                + npsi
                + ntrend_loc
                + nind_loc
                + ntrend_sc
                + nind_sc
                + ngamma
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
            jac="2-point",  # Gradient information
            hess="2-point",  # Hessian information, if applicable
            x0=x_ini,
            bounds=bounds,
            args=(
                nmu,
                npsi,
                ngamma,
                ntrend_loc,
                self.covariates.iloc[:, list_loc],
                ntrend_sc,
                self.covariates.iloc[:, list_loc],
                ntrend_sh,
                self.covariates.iloc[:, list_loc],
            ),  # TODO: COMPROBAR ESTOS ARGUMENTOS
            options={"gtol": 1e-8},  # Options
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
                jac="2-point",  # Gradient information
                hess="2-point",  # Hessian information, if applicable
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
                options={"gtol": 1e-8},  # Options
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
            fit_result["beta"] = None
        if ntrend_loc > 0:
            fit_result["betaT"] = fit_result["x"][1 + nmu]
        else:
            fit_result["betaT"] = None
        if nind_loc > 0:
            fit_result["beta_cov"] = fit_result["x"][
                1 + nmu + ntrend_loc : 1 + nmu + ntrend_loc + nind_loc
            ]
        else:
            fit_result["beta_cov"] = None

        # Scale parameter
        fit_result["alpha0"] = fit_result["x"][1 + nmu + ntrend_loc + nind_loc]
        if npsi > 0:
            fit_result["alpha"] = fit_result["x"][
                2 + nmu + ntrend_loc + nind_loc : 2 + nmu + ntrend_loc + nind_loc + npsi
            ]
        else:
            fit_result["alpha"] = None
        if ntrend_sc > 0:
            fit_result["alphaT"] = fit_result["x"][
                2 + nmu + ntrend_loc + nind_loc + npsi
            ]
        else:
            fit_result["alphaT"] = None
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
            fit_result["alpha_cov"] = None

        # Shape parameter
        if self.ngamma0 == 1:
            fit_result["gamma0"] = fit_result["x"][
                2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + nind_sc
            ]
        else:
            fit_result["gamma0"] = None
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
            fit_result["gamma"] = None
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
            fit_result["gammaT"] = None
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
            fit_result["gamma_cov"] = None

    def _auxmin_loglikelihood(
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
    ) -> float:
        """
        Function used for minimizing in the 'self._optimize_parameters' where the Negative loglikelihood of the GEV will be minimized
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
                covariates_loc,
                x[1 + nmu + ntrend_loc : 1 + nmu + ntrend_loc + nind_loc],
            )  # beta0, beta, beta_cov
        elif ntrend_loc != 0 and nind_loc == 0:
            mut1 = self._parametro(
                x[0], x[1 : 1 + nmu], np.array(x[1 + nmu])
            )  # beta0, beta, betaT
        else:
            mut1 = self._parametro(
                x[0],
                x[1 : 1 + nmu],
                np.array(x[1 + nmu]),
                covariates_loc,
                x[1 + nmu + ntrend_loc : 1 + nmu + ntrend_loc + nind_loc],
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
                    covariates_sc,
                    x[
                        2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                    ],
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
                    np.array(x[2 + nmu + ntrend_loc + nind_loc + npsi]),
                )
            )  # alpha0, alpha, betaT2
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
                    np.array(x[2 + nmu + ntrend_loc + nind_loc + npsi]),
                    covariates_sc,
                    x[
                        2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                    ],
                )
            )  # alpha0, alpha, betaT2, alpha_cov

        # Evaluate the shape parameter at each time t as a function of the actual values of the parameters given by x
        # TODO: Añadir gammaT
        if self.ngamma0 == 1:
            if self.ngamma == 0 and nind_sh == 0:
                epst = self._parametro(
                    x[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + nind_sc],
                    None,
                )  # gamma0
            elif self.ngamma != 0 and nind_sh == 0:
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
            elif self.ngamma == 0 and nind_sh != 0:
                epst = self._parametro(
                    x[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + nind_sc],
                    None,
                    None,
                    covariates_sh,
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
                )  # gamma0, gamma_cov
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
                    None,
                    covariates_sh,
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
                )  # gamma0, gamma, gamma_cov
        else:
            if self.ngamma == 0 and nind_sh == 0:
                epst = 0 * mut1  # Gumbel
            elif self.ngamma != 0 and nind_sh == 0:
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
            elif self.ngamma == 0 and nind_sh != 0:
                epst = self._parametro(
                    None,
                    None,
                    None,
                    covariates_sh,
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
                )  # gamma_cov
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
                    None,
                    covariates_sh,
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
                )  # gamma, gamma_cov

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
        TODO: ELIMINAR ESTO Y USAR EL GRADIENTE NUMERICO
        Function used for minimizing in the 'self._optimize_parameters' where the Negative loglikelihood of the GEV will be minimized
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
                covariates_loc,
                x[1 + nmu + ntrend_loc : 1 + nmu + ntrend_loc + nind_loc],
            )  # beta0, beta, varphi
        elif ntrend_loc != 0 and nind_loc == 0:
            mut1 = self._parametro(
                x[0], x[1 : 1 + nmu], np.array(x[1 + nmu])
            )  # beta0, beta, betaT
        else:
            mut1 = self._parametro(
                x[0],
                x[1 : 1 + nmu],
                np.array(x[1 + nmu]),
                covariates_loc,
                x[1 + nmu + ntrend_loc : 1 + nmu + ntrend_loc + nind_loc],
            )  # beta0, beta, betaT, varphi

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
                    covariates_sc,
                    x[
                        2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                    ],
                )
            )  # alpha0, alpha, varphi2
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
                    np.array(x[2 + nmu + ntrend_loc + nind_loc + npsi]),
                )
            )  # alpha0, alpha, betaT2
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
                    np.array(x[2 + nmu + ntrend_loc + nind_loc + npsi]),
                    covariates_sc,
                    x[
                        2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc : 2
                        + nmu
                        + ntrend_loc
                        + nind_loc
                        + npsi
                        + ntrend_sc
                        + nind_sc
                    ],
                )
            )  # alpha0, alpha, betaT2, varphi2

        # Evaluate the shape parameter at each time t as a function of the actual values of the parameters given by x
        if self.ngamma0 == 1:
            if self.ngamma == 0 and nind_sh == 0:
                epst = self._parametro(
                    x[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + nind_sc],
                    None,
                )  # gamma0
            elif self.ngamma != 0 and nind_sh == 0:
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
            elif self.ngamma == 0 and nind_sh != 0:
                epst = self._parametro(
                    x[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + nind_sc],
                    None,
                    None,
                    covariates_sh,
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
                )  # gamma0, varphi3
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
                    None,
                    covariates_sh,
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
                )  # gamma0, gamma, varphi3
        else:
            if self.ngamma == 0 and nind_sh == 0:
                epst = 0 * mut1  # Gumbel
            elif self.ngamma != 0 and nind_sh == 0:
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
            elif self.ngamma == 0 and nind_sh != 0:
                epst = self._parametro(
                    None,
                    None,
                    None,
                    covariates_sh,
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
                )  # varphi3
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
                    None,
                    covariates_sh,
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
                )  # gamma, varphi3

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

        # Evaluate the loglikelihood function with the sign changed, not that the general and Gumbel expressions are used
        # f = np.sum(-np.log(self.kt[pos]) + np.log(psit[pos]) + (1+1/epst[pos])*np.log(z[pos])+self.kt[pos]*zn[pos]) + \
        #    np.sum(-np.log(self.kt[posG]) +np.log(psit[posG]) + xn[posG] + self.kt[posG]*np.exp(-xn[posG]))

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

        # Jacobian elements related to the location parameters betaT, varphi (equation A.9)
        if ntrend_loc > 0:
            Jx[1 + nmu] = np.sum(Dmut * self.t * Dmutastmut)  # betaT
        if nind_loc > 0:
            for i in range(nind_loc):
                Jx[1 + nmu + ntrend_loc + i] = np.sum(
                    Dmut * covariates_loc[:, i] * Dmutastmut
                )  # varphi_i

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
        # Jacobian elements related to the scale parameters betaT2 and varphi (equation A.10)
        if ntrend_sc > 0:
            Jx[2 + nmu + ntrend_loc + nind_loc + npsi] = np.sum(
                (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit) * self.t * psit1
            )  # betaT2
        if nind_sc > 0:
            for i in range(nind_sc):
                Jx[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i] = np.sum(
                    (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit)
                    * covariates_sc[:, i]
                    * psit1
                )  # varphi2

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

        # Jacobian elements related to the shape parameters varphi3 (defined by Victor)
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
                    + i
                ] = np.sum(Depst * covariates_sh[:, i])

        # Change the Jacobian sign since the numerical problem is a minimization problem
        Jx = -Jx

        return Jx

    def fit(
        self,
        nmu,
        npsi,
        ngamma,
        ntrend_loc=0,
        list_loc=[],
        ntrend_sc=0,
        list_sc=[],
        ntrend_sh=0,
        list_sh=[],
    ) -> dict:
        """
        Function to determine the optimal parameters of given Non-Stationary GEV considering all the covariates

        TODO: QUIZAS QUITAR ESTA FUNCION Y DEJAR SOLO LA _fit PORQUE AL FINAL HACEN LO MISMO O AÑADIR
        ALGO PARA AJUSTAR USANDO TODAS LAS COVARIABLES DE FORMA MAS FACIL (PONER COMO DEFAULT)

        Parameters
        ----------
        nmu : int
            Number of harmonics to be included in the location parameter
        npsi : int
            Number of harmonics to be included in the scale parameter
        neps : int
            Number of harmonics to be included in the shape parameter
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

        Returns
        ----------
        fit_result : dict
            Dictionary with the optimal parameters and other information about the fit.
            The keys of the dictionary are:
            - beta0, beta, betaT, beta_cov: Location parameters (intercept, harmonic, trend, covariates)
            - alpha0, alpha, alphaT, alpha_cov: Scale parameters (intercept, harmonic, trend, covariates)
            - gamma0, gamma, gammaT, gamma_cov: Shape parameters (intercept, harmonic, trend, covariates)
            - negloglikelihood: Negative log-likelihood value at the optimal solution
            - hessian: Hessian matrix of the log-likelihood function at the optimal solution
            - AIC: Akaike Information Criterion value at the optimal solution
        """
        fit_result = self._fit(
            nmu,
            npsi,
            ngamma,
            list_loc,
            ntrend_loc,
            list_sc,
            ntrend_sc,
            list_sh,
            ntrend_sh,
        )

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
        ----------
        aic : float
            AIC value
        """
        aic = -2 * loglike + 2 * nparam
        return aic

    def _loglikelihood(
        self,
        beta0: float=None,
        beta: np.ndarray=None,
        betaT: float=None,
        beta_cov: np.ndarray=None,
        alpha0: float=None,
        alpha: np.ndarray=None,
        alphaT: float=None,
        alpha_cov: np.ndarray=None,
        gamma0: float=None,
        gamma: np.ndarray=None,
        gammaT: float=None,
        gamma_cov: np.ndarray=None,
        list_loc: list=[],
        list_sc: list=[],
        list_sh: list=[],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Function to calculate the loglikelihood function, the Jacobian and the Hessian for a given parameterization

        Parameters
        ----------
        beta0 : float, default=None
            Optimal constant parameter related to location
        beta : np.ndarray, default=None
            Optimal harmonic vector associated with location
        betaT : float, default=None
            Optimal location trend parameter
        beta_cov : np.ndarray, default=None
            Optimal location covariate vector
        alpha0 : float, default=None
            Optimal constant parameter related to scale
        alpha : np.ndarray, default=None
            Optimal harmonic vector associated with scale
        alphaT : float, default=None
            Optimal scale trend parameter
        alpha_cov : np.ndarray, default=None
            Optimal scale covariate vector
        gamma0 : float, default=None
            Optimal constant parameter related to shape
        gamma : np.ndarray, default=None
            Optimal harmonic vector associated with shape
        gammaT : float, default=None
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
        ----------
        f : np.ndarray
            Optimal loglikelihood function
        Jx : np.ndarray
            Gradient of the log-likelihood function at the optimal solution
        Hxx : np.ndarray
            Hessian of the log-likelihood function at the optimal solution
        """

        # Location
        if beta0 is None:
            beta0 = np.empty(0)
        if beta is None:
            beta = np.empty(0)
        if betaT is None:
            betaT = np.empty(0)
            ntrend_loc = 0
        else:
            ntrend_loc = 1

        # Scale
        if alpha0 is None:
            alpha0 = np.empty(0)
        if alpha is None:
            alpha = np.empty(0)
        if alphaT is None:
            alphaT = np.empty(0)
            ntrend_sc = 0
        else:
            ntrend_sc = 1

        # Shape
        if gamma0 is None:
            gamma0 = np.empty(0)
        if gamma is None:
            gamma = np.empty(0)
        if gammaT is None:
            gammaT = np.empty(0)
            ntrend_sh = 0
        else:
            ntrend_sh = 1

        if beta_cov is None:
            beta_cov = np.empty(0)
        if alpha_cov is None:
            alpha_cov = np.empty(0)
        if gamma_cov is None:
            gamma_cov = np.empty(0)

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

        nmu = beta.size
        npsi = alpha.size
        ngamma = gamma.size
        nind_loc = beta_cov.size
        nind_sc = alpha_cov.size
        nind_sh = gamma_cov.size

        # Evaluate the parameters
        # TODO: COMPROBAR LOS INPUTS
        mut1, psit1, epst = self._evaluate_params(
            beta0,
            beta,
            alpha0,
            alpha,
            gamma0,
            gamma,
            betaT,
            beta_cov,
            alphaT,
            alpha_cov,
            gammaT,
            gamma_cov,
            covariates_loc,
            covariates_sc,
            covariates_sh,
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

        # Jacobian elements related to the location parameters betaT, varphi (equation A.9)
        if ntrend_loc > 0:
            Jx[1 + nmu] = np.sum(Dmut * self.t * Dmutastmut)  # betaT
        if nind_loc > 0:
            for i in range(nind_loc):
                Jx[1 + nmu + ntrend_loc + i] = np.sum(
                    Dmut * covariates_loc[:, i] * Dmutastmut
                )  # varphi_i

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
        # Jacobian elements related to the scale parameters betaT2 and varphi (equation A.10)
        if ntrend_sc > 0:
            Jx[2 + nmu + ntrend_loc + nind_loc + npsi] = np.sum(
                (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit) * self.t * psit1
            )  # betaT2
        if nind_sc > 0:
            for i in range(nind_sc):
                Jx[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i] = np.sum(
                    (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit)
                    * covariates_sc[:, i]
                    * psit1
                )  # varphi2

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
        # Jacobian elements related to the shape parameters varphi3 (defined by Victor)
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
                + (z / (epst * epst))
                * np.log(z)
                * (
                    2 * epst * (-xn * (1 + epst) - 1 + z ** (1 + 1 / epst))
                    + z * np.log(z)
                )
            )
            / (epst * epst * z**2)
        )
        Dmutpsit = -(1 + epst - (1 - xn) * zn) / ((z * psit) ** 2)
        Dmutepst = (
            -zn
            * (
                epst * (-(1 + epst) * xn - epst * (1 - xn) * z ** (1 / epst))
                + z * np.log(z)
            )
            / (epst * epst * psit * z**2)
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
        # Sub-block number 3, varphi_i*varphi_j
        if nind_loc > 0:
            for i in range(nind_loc):
                for j in range(i + 1):
                    Hxx[1 + nmu + ntrend_loc + i, 1 + nmu + ntrend_loc + j] = np.sum(
                        D2mut * covariates_loc[:, i] * covariates_loc[:, j]
                    )
        # Sub-block number 4, betaT2^2
        if ntrend_sc > 0:
            Hxx[
                2 + nmu + npsi + ntrend_loc + nind_loc,
                2 + nmu + npsi + ntrend_loc + nind_loc,
            ] = np.sum((D2psit * psit + Dpsit) * psit * (self.t**2))
        # Sub-block number 5, varphi2_i*varphi2_j
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
        # Sub-block added by Victor, varphi3_i*varphi3_j
        if nind_sh > 0:
            for i in range(nind_sh):
                for j in range(i + 1):
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
                            + j,
                        ] = np.sum(D2epst * covariates_sh[:, i] * covariates_sh[:, j])

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
        # Sub-block number 12 (Scale exponential involved), beta0*betaT2
        if ntrend_sc > 0:
            Hxx[2 + nmu + ntrend_loc + nind_loc + npsi, 0] = np.sum(
                Dmutpsit * self.t * psit
            )
        # Sub-block number 52 (Scale exponential involved), betaT2*alpha0
        if ntrend_sc > 0:
            Hxx[
                2 + nmu + ntrend_loc + nind_loc + npsi, 1 + nmu + ntrend_loc + nind_loc
            ] = np.sum((D2psit * psit + Dpsit) * self.t * psit)
        # Sub-block number 48 (Scale exponential involved), betaT*betaT2
        if ntrend_loc > 0 and ntrend_sc > 0:
            Hxx[2 + nmu + ntrend_loc + nind_loc + npsi, 1 + nmu] = np.sum(
                Dmutpsit * self.t * self.t * psit
            )
        # Sub-block number 13, beta0*varphi_i
        if nind_loc > 0:
            for i in range(nind_loc):
                Hxx[1 + nmu + ntrend_loc + i, 0] = np.sum(D2mut * covariates_loc[:, i])
        # Sub-block number 14 (Scale exponential involved), beta0*varphi2_i
        if nind_sc > 0:
            for i in range(nind_sc):
                Hxx[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i, 0] = np.sum(
                    Dmutpsit * covariates_sc[:, i] * psit
                )
        # Sub-block number 53 (Scale exponential involved), alpha0*varphi2_i
        if nind_sc > 0:
            for i in range(nind_sc):
                Hxx[
                    2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i,
                    1 + nmu + ntrend_loc + nind_loc,
                ] = np.sum((D2psit * psit + Dpsit) * covariates_sc[:, i] * psit)
        # Sub-block number 49 (Scale exponential involved), betaT*varphi2_i
        if ntrend_loc > 0 and nind_sc > 0:
            for i in range(nind_sc):
                Hxx[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i, 1 + nmu] = (
                    np.sum(Dmutpsit * self.t * covariates_sc[:, i] * psit)
                )
        # Sub-block number 15, betaT*varphi_i
        if nind_loc > 0 and ntrend_loc > 0:
            for i in range(nind_loc):
                Hxx[1 + nmu + ntrend_loc + i, 1 + nmu] = np.sum(
                    D2mut * self.t * covariates_loc[:, i]
                )
        # Sub-block number 16, betaT2*varphi2_i
        if nind_sc > 0 and ntrend_sc > 0:
            for i in range(nind_sc):
                Hxx[
                    2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i,
                    2 + nmu + ntrend_loc + nind_loc + npsi,
                ] = np.sum(
                    (D2psit * psit + Dpsit) * self.t * covariates_sc[:, i] * psit
                )
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
        # Sub-block number 19 (Scale exponential involved), gamma0*betaT2
        if ntrend_sc > 0 and self.ngamma0 == 1:
            Hxx[
                2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                2 + nmu + ntrend_loc + nind_loc + npsi,
            ] = np.sum(Dpsitepst * self.t * psit)
        # Sub-block number 20 (Scale exponential involved), alpha0*varphi_i
        if nind_loc > 0:
            for i in range(nind_loc):
                Hxx[1 + nmu + ntrend_loc + nind_loc, 1 + nmu + ntrend_loc + i] = np.sum(
                    Dmutpsit * covariates_loc[:, i] * psit
                )
        # Sub-block number 21, gamma0*varphi_i
        if nind_loc > 0 and self.ngamma0 == 1:
            for i in range(nind_loc):
                Hxx[
                    2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                    1 + nmu + ntrend_loc + i,
                ] = np.sum(Dmutepst * covariates_loc[:, i])
        # Sub-block number 22 (Scale exponential involved), gamma0*varphi2_i
        if nind_sc > 0 and self.ngamma0 == 1:
            for i in range(nind_sc):
                Hxx[
                    2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + nind_sc,
                    2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + i,
                ] = np.sum(Dpsitepst * covariates_sc[:, i] * psit)
        # Sub-block added by Victor, gamma0*varphi3_i
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
                    # Sub-block number 46 (Scale exponential involved), betaT2*beta_i
                    Hxx[2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc, 1 + i] = aux
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
                        # Sub-block number 28, beta_i*varphi_j
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
                        # Sub-block number 47 (Scale exponential involved), beta_i*varphi2_j
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
                        # Sub-block added by Victor, beta_j*varphi3_i
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
                        # Sub-block number 34 (Scale exponential involved), alpha_i*varphi_j
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
                        # Sub-block added by Victor (scale exponential involved), alpha_i*varphi3_j
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
                            + j,
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
                    # Sub-block number 44 (Scale exponential involved), gamma_i*betaT2
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
                        # Sub-block number 40, gamma_i*varphi_j
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
                        # Sub-block number 45 (Scale exponential involved), gamma_i*varphi2_j
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
                        # Sub-block added by Victor, gamma_i*varphi3_j
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
                # Sub-block number 50 (Scale exponential involved), varphi_i*betaT2
                Hxx[
                    2 + nmu + ntrend_loc + nind_loc + npsi, 1 + nmu + ntrend_loc + i
                ] = aux
        if nind_loc > 0 and nind_sc > 0:
            for i in range(nind_loc):
                for j in range(nind_sc):
                    # Sub-block number 51 (Scale exponential involved), varphi_i*varphi2_j
                    Hxx[
                        2 + nmu + ntrend_loc + nind_loc + npsi + ntrend_sc + j,
                        1 + nmu + ntrend_loc + i,
                    ] = np.sum(
                        Dmutpsit * covariates_sc[:, j] * covariates_loc[:, i] * psit
                    )
        if nind_loc > 0 and nind_sh > 0:
            for i in range(nind_loc):
                for j in range(nind_sh):
                    # Sub-block added by Victor, varphi_i*varphi3_j
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
                        + j,
                        1 + nmu + ntrend_loc + i,
                    ] = np.sum(Dmutepst * covariates_loc[:, i] * covariates_sh[:, j])
        if nind_sc > 0 and nind_sh > 0:
            for i in range(nind_sc):
                for j in range(nind_sh):
                    # Sub-block added by Victor (scale exponential involved), varphi2_i*varphi3_j
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
                        + j,
                        2 + nmu + npsi + ntrend_loc + nind_loc + ntrend_sc + i,
                    ] = np.sum(
                        Dmutepst * covariates_sc[:, i] * covariates_sh[:, j] * psit
                    )
        if nind_sh > 0 and ntrend_loc > 0:
            for i in range(nind_sh):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dmutepst[k] * tt * covariates_sh[k, i]
                # Sub-block added by Victor, betaT*varphi3_i
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
                    + i,
                    1 + nmu,
                ] = aux
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
                # Sub-block number 54 (Scale exponential involved), alpha_i*betaT2
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
                    # Sub-block number 55 (Scale exponential involved), alpha_i*varphi2_j
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
                # Sub-block added by Victor, beta0*varphi3_i
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
                    + i,
                    0,
                ] = np.sum(Dmutepst * covariates_sh[:, i])
                # Sub-block added by Victor (scale exponential involved), alpha0*varphi3_i
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
                    + i,
                    1 + nmu + ntrend_loc + nind_loc,
                ] = np.sum(Dpsitepst * psit * covariates_sh[:, i])

                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dpsitepst[k] * tt * covariates_sh[k, i] * psit[k]
                # Sub-bloc added by Victor (scale exponential involved), betaT2*varphi3_i
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
                    + i,
                    1 + nmu + npsi + ntrend_loc + nind_loc,
                ] = aux

        # Simmetric part of the Hessian
        Hxx = Hxx + np.tril(Hxx, -1).T

        return f, Jx, Hxx

    def _update_params(self, **kwargs) -> None:
        self.beta0 = kwargs.get("beta0")
        self.beta = kwargs.get("beta")
        self.betaT = kwargs.get("betaT")
        self.beta_cov = kwargs.get("beta_cov")

        self.alpha0 = kwargs.get("alpha0")
        self.alpha = kwargs.get("alpha")
        self.alphaT = kwargs.get("alphaT")
        self.alpha_cov = kwargs.get("alpha_cov")

        self.ngamma0 = kwargs.get("ngamma0")
        self.gamma0 = kwargs.get("gamma0")
        self.gamma = kwargs.get("gamma")
        self.gammaT = kwargs.get("gammaT")
        self.gamma_cov = kwargs.get("gamma_cov")

        self.popt = kwargs.get("popt")

    def _parametro(
        self,
        alpha0=None,
        alpha=None,
        betaT=None,
        covariates=None,
        varphi=None,
        indicesint=None,
        times=None,
        x=None,
    ) -> np.ndarray:
        """
        This function computes the location, scale and shape parameters for given parameters. Expressions by (2)-(3) in the paper

        Inputs:
        - alpha0: Value of the intercept
        - alpha: Value of the harmonics terms
        - betaT: tendency for the location parameter, just need to include a number in order to consider it, otherwise leave it empty
        - covariates: Covariates data related to the location parameter, a matrix including the data at time t for each covariate
        - varphi: covariate parameter vector associated with the location parameter, just need to include a list of numbers of length the number of covariates, otherwise leave it empty
        - indicesint: covariate mean values in the integral interval
        - times: times when covariates are known, nearest value (function _search)

        Output:
        - y: values of the parameter
        """

        if alpha is None:
            alpha = np.empty(0)
        if betaT is None or betaT.size == 0:
            betaT = np.empty(0)
            ntend = 0
        else:
            ntend = 1
        if covariates is None:
            covariates = np.empty((0, 0))
        if varphi is None:
            varphi = np.empty(0)
        if indicesint is None:
            indicesint = np.empty((0, 0))
        if times is None:
            times = np.empty((0, 0))
        if x is not None:
            t = np.array([x])
        else:
            t = self.t

        m = len(t)

        na, nind = covariates.shape
        nparam = len(alpha)
        # Chek if the number of parameters is even
        if nparam % 2 != 0:
            raise ValueError("Parameter number must be even")

        # Adding the intercept term
        if alpha0 is not None and np.asarray(alpha0).size > 0:
            y = alpha0 * np.ones(m)
        else:
            y = np.zeros(m)

        # Adding the harmonic part
        if nparam > 0:
            for i in range(nparam // 2):
                y += alpha[2 * i] * np.cos((i + 1) * 2 * np.pi * t) + alpha[
                    2 * i + 1
                ] * np.sin((i + 1) * 2 * np.pi * t)

        # Adding the tendency part
        if ntend > 0:
            y += betaT * t

        # Adding the covariate part
        if nind > 0:
            if indicesint.shape[1] > 0:
                if times.shape[1] == 0:
                    for i in range(nind):
                        y += varphi[i] * indicesint[i]
                else:
                    for i in range(nind):
                        indicesintaux = self._search(times, covariates[:, i])
                        y += varphi[i] * indicesintaux
            else:
                for i in range(nind):
                    y += varphi[i] * covariates[:, i]

        return y

    def _search(self, times, values):
        """
        Function to search the nearest value of certain time to use in self._parametro function
        """
        n = times.shape[0]
        yin = np.zeros_like(self.t)
        pos = 0
        for j in range(len(self.t)):
            found = 0
            while found == 0 and pos <= n:
                if self.t[j] < times[pos]:
                    yin[j] = values[pos]
                    found = 1
                else:
                    pos += 1

        return yin

    def _evaluate_params(
        self,
        beta0=None,
        beta=None,
        alpha0=None,
        alpha=None,
        gamma0=None,
        gamma=None,
        betaT=None,
        varphi=None,
        betaT2=None,
        varphi2=None,
        varphi3=None,
        covariates_loc=None,
        covariates_sc=None,
        covariates_sh=None,
    ):
        """
        Function to evaluate the parameters in the corresponding values
        """
        # Evaluate the location parameter at each time t as function of the actual values of the parameters given by p
        mut1 = self._parametro(beta0, beta, betaT, covariates_loc, varphi)
        # Evaluate the scale parameter at each time t as function of the actual values of the parameters given by p
        psit1 = np.exp(self._parametro(alpha0, alpha, betaT2, covariates_sc, varphi2))
        # Evaluate the shape parameter at each time t as function of the actual values of the parameters given by p
        epst = self._parametro(gamma0, gamma, None, covariates_sh, varphi3)

        return mut1, psit1, epst

    @staticmethod
    def _Dparam(t, i):
        """
        Derivative of the location, scale and shape fucntions with respect to harmonic parameters. It corresponds to the rhs in equation (A.11) of the paper
        Input:
        -t: time in yearly scale
        -i: harmonic number

        Output:
        -dp: Corresponding derivative
        """

        if i % 2 == 0:
            dp = np.sin(i / 2 * 2 * np.pi * t)
        else:
            dp = np.cos((i + 1) / 2 * 2 * np.pi * t)
        return dp

    def _quantile(self, harm=False):
        """
        Calculates the quantile q associated with a given parameterization, the main input is quanval introduced in __init__ (default 0.95)
        """

        if harm:
            betaT = None
            betaT2 = None
            cov_loc = None
            varphi = None
            cov_sc = None
            varphi2 = None
            cov_sh = None
            varphi3 = None
        else:
            betaT = self.betaT
            betaT2 = self.betaT2
            cov_loc = self.covariates[:, self.list_loc]
            varphi = self.varphi
            cov_sc = self.covariates[:, self.list_sc]
            varphi2 = self.varphi2
            cov_sh = self.covariates[:, self.list_sh]
            varphi3 = self.varphi3

        Q = np.zeros(len(self.xt))

        # Evaluate the parameters
        mut1, psit1, epst = self._evaluate_params(
            self.beta0,
            self.beta,
            self.alpha0,
            self.alpha,
            self.gamma0,
            self.gamma,
            betaT,
            varphi,
            betaT2,
            varphi2,
            varphi3,
            cov_loc,
            cov_sc,
            cov_sh,
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
            - (1 - (-np.log(self.quanval) / self.kt[pos]) ** (-epst[pos]))
            * psit[pos]
            / epst[pos]
        )
        Q[posG] = mut[posG] - psit[posG] * np.log(-np.log(self.quanval) / self.kt[posG])

        return Q

    def plot(self, return_plot=True):
        """
        Plot the location, scale and shape parameters, also the PP plot and QQ plot
        """

        # Parameter Evaluation
        mut1, psit1, epst = self._evaluate_params(
            self.beta0,
            self.beta,
            self.alpha0,
            self.alpha,
            self.gamma0,
            self.gamma,
            self.betaT,
            self.varphi,
            self.betaT2,
            self.varphi2,
            self.varphi3,
            self.covariates[:, self.list_loc],
            self.covariates[:, self.list_sc],
            self.covariates[:, self.list_sh],
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
                        : 1 + 2 * self.nmu + self.ntend_loc + self.nind_loc,
                        : 1 + 2 * self.nmu + self.ntend_loc + self.nind_loc,
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
                        1 + 2 * self.nmu + self.ntend_loc + self.nind_loc : 2
                        + 2 * self.nmu
                        + self.ntend_loc
                        + self.nind_loc
                        + 2 * self.npsi
                        + self.ntend_sc
                        + self.nind_sc,
                        1 + 2 * self.nmu + self.ntend_loc + self.nind_loc : 2
                        + 2 * self.nmu
                        + self.ntend_loc
                        + self.nind_loc
                        + 2 * self.npsi
                        + self.ntend_sc
                        + self.nind_sc,
                    ]
                )
                * Derpsit.T,
                axis=1,
            )
        )
        if self.neps0 == 1 or self.neps > 0 or self.nind_sh > 0:
            stdepst = np.sqrt(
                np.sum(
                    (
                        Derepst.T
                        @ self.invI0[
                            2
                            + 2 * self.nmu
                            + self.ntend_loc
                            + self.nind_loc
                            + 2 * self.npsi
                            + self.ntend_sc
                            + self.nind_sc : 2
                            + 2 * self.nmu
                            + self.ntend_loc
                            + self.nind_loc
                            + 2 * self.npsi
                            + self.ntend_sc
                            + self.nind_sc
                            + self.neps0
                            + 2 * self.neps
                            + self.nind_sh,
                            2
                            + 2 * self.nmu
                            + self.ntend_loc
                            + self.nind_loc
                            + 2 * self.npsi
                            + self.ntend_sc
                            + self.nind_sc : 2
                            + 2 * self.nmu
                            + self.ntend_loc
                            + self.nind_loc
                            + 2 * self.npsi
                            + self.ntend_sc
                            + self.nind_sc
                            + self.neps0
                            + 2 * self.neps
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

        if (
            self.betaT is None
            and self.betaT2 is None
            and self.varphi.size == 0
            and self.varphi2.size == 0
        ):
            t_ord = np.argsort(t_anual)
            fig, ax1 = plt.subplots(figsize=(10, 6))
            l0 = ax1.plot(
                t_anual[t_ord],
                self.xt[t_ord],
                marker="+",
                linestyle="None",
                color="black",
                markersize=5,
                label=r"$H_s^{max}$",
            )
            ax2 = ax1.twinx()
            l1 = ax1.plot(
                t_anual[t_ord],
                mut[t_ord],
                label=r"$\mu_t$",
                linewidth=2,
                color=self.colors[0],
                alpha=1,
            )
            ax1.fill_between(
                t_anual[t_ord],
                ci_low_mut[t_ord],
                ci_up_mut[t_ord],
                color=self.colors[0],
                alpha=0.3,
            )
            l2 = ax2.plot(
                t_anual[t_ord],
                psit[t_ord],
                label=r"$\psi_t$",
                linewidth=2,
                color=self.colors[1],
                alpha=1,
            )
            ax2.fill_between(
                t_anual[t_ord],
                ci_low_psit[t_ord],
                ci_up_psit[t_ord],
                color=self.colors[1],
                alpha=0.3,
            )
            l3 = ax1.plot(
                t_anual[t_ord],
                quan95[t_ord],
                linestyle="dashed",
                color=self.colors[2],
                markersize=5,
            )
        else:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            l0 = ax1.plot(
                self.t,
                self.xt,
                marker="+",
                linestyle="None",
                color="black",
                markersize=5,
                label=r"$H_s^{max}$",
            )
            ax2 = ax1.twinx()
            l1 = ax1.plot(
                self.t,
                mut,
                label=r"$\mu_t$",
                linewidth=2,
                color=self.colors[0],
                alpha=1,
            )
            ax1.fill_between(
                self.t, ci_low_mut, ci_up_mut, color=self.colors[0], alpha=0.3
            )
            l2 = ax2.plot(
                self.t,
                psit,
                label=r"$\psi_t$",
                linewidth=2,
                color=self.colors[1],
                alpha=1,
            )
            ax2.fill_between(
                self.t, ci_low_psit, ci_up_psit, color=self.colors[1], alpha=0.3
            )
            l3 = ax1.plot(
                self.t, quan95, linestyle="dashed", color=self.colors[2], markersize=5
            )

        ax1.set_xlabel("Time (yearly scale)")
        ax1.set_ylabel(r"$\mu_t$")
        ax2.set_ylabel(r"$\psi_t$")
        ax1.set_title(f"Location and Scale parameters ({self.example})")
        ax1.grid(True)
        handles = [
            art for art in l0 + l1 + l2 + l3 if not art.get_label().startswith("_")
        ]
        ax1.legend(handles=handles, loc="best")
        ax1.margins(x=0.01)
        plt.show()

        # mu, mu-phi, mu+phi, points
        fig, ax1 = plt.subplots(figsize=(10, 6))
        l0 = ax1.plot(
            self.t,
            self.xt,
            marker="+",
            linestyle="None",
            color="black",
            markersize=5,
            label=r"$H_s^{max}$",
        )
        l1 = ax1.plot(
            self.t, mut, label=r"$\mu_t$", linewidth=2, color=self.colors[0], alpha=1
        )
        ax1.fill_between(
            self.t, mut - psit, mut + psit, color=self.colors[1], alpha=0.3
        )
        l3 = ax1.plot(
            self.t, quan95, linestyle="dashed", color=self.colors[2], linewidth=2
        )
        ax1.set_xlabel("Time (yearly scale)")
        ax1.set_ylabel(r"$\mu_t$")
        ax1.set_title(f"Location parameter ({self.example})")
        ax1.grid(True)
        handles = [art for art in l0 + l1 + l3 if not art.get_label().startswith("_")]
        ax1.legend(handles=handles, loc="best")
        ax1.margins(x=0.01)
        plt.show()

        month_initials = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        month_positions = [(i + 0.5) / 12 for i in range(12)]

        #### Creating the first year plot
        mask_year = (self.t >= 0) & (self.t <= 1)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        l0 = ax1.plot(
            self.t[mask_year],
            self.xt[mask_year],
            marker="+",
            linestyle="None",
            color="black",
            markersize=5,
            label=r"$H_s^{max}$",
        )
        # ax2 = ax1.twinx()
        l1 = ax1.plot(
            self.t[mask_year],
            mut[mask_year],
            label=r"$\mu_t$",
            linewidth=2,
            color=self.colors[0],
            alpha=1,
        )
        # ax1.fill_between(self.t[mask_year], ci_low_mut[mask_year], ci_up_mut[mask_year], color=self.colors[0], alpha=0.3)
        uppermaxs = np.maximum(
            mut[mask_year] - psit[mask_year], mut[mask_year] + psit[mask_year]
        )
        lowermins = np.minimum(
            mut[mask_year] - psit[mask_year], mut[mask_year] + psit[mask_year]
        )
        ax1.fill_between(
            self.t[mask_year], lowermins, uppermaxs, color=self.colors[0], alpha=0.3
        )
        # l2 = ax2.plot(self.t[mask_year], psit[mask_year], label=r'$\psi_t$', linewidth=2, color=self.colors[1], alpha=1)
        # ax2.fill_between(self.t[mask_year], ci_low_psit[mask_year], ci_up_psit[mask_year], color=self.colors[1], alpha=0.3)
        l3 = ax1.plot(
            self.t[mask_year],
            quan95[mask_year],
            linestyle="dashed",
            color=self.colors[2],
            label="95th Quantile",
        )
        ax1.set_title(f"Location and Scale Parameters (First Year) ({self.example})")
        ax1.set_xlabel("Time")
        ax1.set_ylabel(r"$\mu_t$")
        # ax2.set_ylabel(r'$\psi_t$')
        ax1.grid(True)
        # handles = [art for art in l0 + l1 + l2 + l3 if not art.get_label().startswith('_')]
        handles = [art for art in l0 + l1 + l3 if not art.get_label().startswith("_")]
        ax1.legend(handles=handles, loc="best")
        ax1.margins(x=0.01)
        plt.xticks(month_positions, month_initials)
        plt.show()

        # Creating the first monthly plot if not monthly or annual data
        mask_month = (self.t >= 0) & (self.t <= 1 / 12)
        if sum(mask_month) > 1:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            l0 = ax1.plot(
                self.t[mask_month],
                self.xt[mask_month],
                marker="+",
                linestyle="None",
                color="black",
                markersize=5,
                label=r"$H_s^{max}$",
            )
            # ax2 = ax1.twinx()
            l1 = ax1.plot(
                self.t[mask_month],
                mut[mask_month],
                label=r"$\mu_t$",
                linewidth=2,
                color=self.colors[0],
                alpha=1,
            )
            uppermaxs = np.maximum(
                mut[mask_month] - psit[mask_month], mut[mask_month] + psit[mask_month]
            )
            lowermins = np.minimum(
                mut[mask_month] - psit[mask_month], mut[mask_month] + psit[mask_month]
            )
            ax1.fill_between(
                self.t[mask_month],
                lowermins,
                uppermaxs,
                color=self.colors[0],
                alpha=0.3,
            )
            # l2 = ax2.plot(self.t[mask_month], psit[mask_month], label=r'$\psi_t$', linewidth=2, color=self.colors[1], alpha=1)
            # ax2.fill_between(self.t[mask_month], ci_low_psit[mask_month], ci_up_psit[mask_month], color=self.colors[1],
            #                 alpha=0.3)
            l3 = ax1.plot(
                self.t[mask_month],
                quan95[mask_month],
                linestyle="dashed",
                color=self.colors[2],
                label="95th Quantile",
            )
            ax1.set_title(
                f"Location and Scale Parameters (First Month) ({self.example})"
            )
            ax1.set_xlabel("Time (yearly scale)")
            ax1.set_ylabel(r"$\mu_t$")
            # ax2.set_ylabel(r'$\psi_t$')
            ax1.grid(True)
            # handles = [art for art in l0 + l1 + l2 + l3 if not art.get_label().startswith('_')]
            handles = [
                art for art in l0 + l1 + l3 if not art.get_label().startswith("_")
            ]
            ax1.legend(handles=handles, loc="best")
            ax1.margins(x=0.01)
            plt.show()

        ### Shape parameter plot
        if self.neps > 0:
            t_ord = np.argsort(t_anual)

            # Confidence interval for epst
            ci_up = (
                epst + norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdepst
            )
            ci_low = (
                epst - norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdepst
            )

            plt.figure(figsize=(10, 6))
            plt.plot(t_anual[t_ord], epst[t_ord], color=self.colors[0])
            plt.fill_between(
                t_anual[t_ord],
                ci_low[t_ord],
                ci_up[t_ord],
                color=self.colors[0],
                alpha=0.3,
                label=r"$\xi_t$ Confidence Interval",
            )
            plt.title(f"Shape parameter ({self.example})")
            plt.xlabel("Time (yearly scale)")
            plt.ylabel(r"$\xi_t$")
            plt.xticks(month_positions, month_initials)
            plt.grid(True)
            plt.show()

        ### Harmonic Location parameter plot
        if self.nmu > 0:
            t_ord = np.argsort(t_anual)
            quan95_2 = self._quantile(harm=True)

            mut2 = self._parametro(self.beta0, self.beta)
            # Confidence interval for mut
            ci_up = mut2 + norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdmut
            ci_low = (
                mut2 - norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdmut
            )

            plt.figure(figsize=(10, 6))
            plt.plot(
                t_anual[t_ord],
                self.xt[t_ord],
                marker="+",
                linestyle="None",
                color="black",
                markersize=5,
                label=r"$H_s^{max}$",
            )
            plt.plot(
                t_anual[t_ord],
                mut2[t_ord],
                label=r"$\mu_t$",
                linewidth=2,
                color=self.colors[0],
            )
            plt.fill_between(
                t_anual[t_ord],
                ci_low[t_ord],
                ci_up[t_ord],
                color=self.colors[0],
                alpha=0.3,
            )
            # Confidence interval for the quantile
            ci_up = (
                quan95_2 + norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdDq
            )
            ci_low = (
                quan95_2 - norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdDq
            )
            plt.plot(
                t_anual[t_ord],
                quan95_2[t_ord],
                linestyle="dashed",
                color=self.colors[1],
                markersize=5,
                label=rf"$q_{self.quanval}$",
            )
            plt.fill_between(
                t_anual[t_ord],
                ci_low[t_ord],
                ci_up[t_ord],
                color=self.colors[1],
                alpha=0.3,
            )
            plt.title(f"Harmonic part of Location parameter ({self.example})")
            plt.xlabel("Time (yearly scale)")
            plt.ylabel(r"$\mu_t$")
            plt.xticks(month_positions, month_initials)
            plt.legend(loc="best")
            plt.grid(True)
            plt.show()

        ### Scale parameter plot
        if self.npsi > 0:
            t_ord = np.argsort(t_anual)

            psit2 = np.exp(self._parametro(self.alpha0, self.alpha))
            # Confidence interval for psit
            ci_up = (
                psit2 + norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdpsit
            )
            ci_low = (
                psit2 - norm.ppf(1 - (1 - self.quanval) / 2, loc=0, scale=1) * stdpsit
            )

            plt.figure(figsize=(10, 6))
            plt.plot(
                t_anual[t_ord],
                psit2[t_ord],
                label=r"$\psi_t$",
                linewidth=2,
                color=self.colors[0],
            )
            plt.fill_between(
                t_anual[t_ord],
                ci_low[t_ord],
                ci_up[t_ord],
                color=self.colors[0],
                alpha=0.3,
                label=r"$\psi_t$ Confidence Interval",
            )
            # plt.plot(t_anual[t_ord], quan95[t_ord], linestyle='dashed', color=self.colors[2], markersize=5, label=fr"$q_{self.quanval}$")
            plt.title(f"Harmonic part of Scale parameter ({self.example})")
            plt.xlabel("Time (yearly scale)")
            plt.xticks(month_positions, month_initials)
            plt.ylabel(r"$\psi_t$")
            plt.grid(True)
            plt.show()

        #### PP Plot
        self.PPplot()

        #### QQ plot
        self.QQplot()

        #### Return periods
        if (
            self.ntend_loc == 0
            and self.ntend_sc == 0
            and self.nind_loc == 0
            and self.nind_sc == 0
            and self.nind_sh == 0
        ) and return_plot:
            self.ReturnPeriodPlot()

    def QQplot(self):
        """
        Print QQ plot
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
            and self.ntend_loc == 0
            and self.ntend_sc == 0
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
        plt.title(f"Best model QQ plot ({self.example})")
        plt.xlabel("Empirical")
        plt.ylabel("Fitted")
        plt.axis("square")
        plt.grid(True)
        plt.margins(x=0.1)
        plt.show()

    def _Zstandardt(self):
        """
        Calculates the standardized variable corresponding to the given parameters
        """

        Zt = np.zeros(len(self.xt))

        # Evaluate the parameters
        mut1, psit1, epst = self._evaluate_params(
            self.beta0,
            self.beta,
            self.alpha0,
            self.alpha,
            self.gamma0,
            self.gamma,
            self.betaT,
            self.varphi,
            self.betaT2,
            self.varphi2,
            self.varphi3,
            self.covariates[:, self.list_loc],
            self.covariates[:, self.list_sc],
            self.covariates[:, self.list_sh],
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

    def _Dzweibull(self):
        """
        Calculates the derivatives of the standardized maximum with respect to parameters
        """

        nd = len(self.t)

        # Evaluate the parameters
        mut1, psit1, epst = self._evaluate_params(
            self.beta0,
            self.beta,
            self.alpha0,
            self.alpha,
            self.gamma0,
            self.gamma,
            self.betaT,
            self.varphi,
            self.betaT2,
            self.varphi2,
            self.varphi3,
            self.covariates[:, self.list_loc],
            self.covariates[:, self.list_sc],
            self.covariates[:, self.list_sh],
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
        zn = z ** (-1 / epst)

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
                + self.neps0
                + 2 * self.nmu
                + 2 * self.npsi
                + 2 * self.neps
                + self.ntend_loc
                + self.nind_loc
                + self.ntend_sc
                + self.nind_sc
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

        # Jacobian elements related to the location parameters betaT, varphi (equation A.9)
        if self.ntend_loc > 0:
            Dq[1 + 2 * self.nmu, :] = Dmut * self.t * Dmutastmut  # betaT
        if self.nind_loc > 0:
            for i in range(self.nind_loc):
                Dq[1 + 2 * self.nmu + self.ntend_loc + i, :] = (
                    Dmut * self.covariates[:, self.list_loc[i]] * Dmutastmut
                )  # varphi_i

        # Jacobian elements related to the scale parameters alpha0, alpha (equation A.7)
        Dq[1 + 2 * self.nmu + self.ntend_loc + self.nind_loc, :] = psit1 * (
            Dpsit * Dpsitastpsit + Dmut * Dmutastpsit
        )  # alpha0
        # If scale harmonic are included
        if self.npsi > 0:
            for i in range(2 * self.npsi):
                for k in range(len(self.t)):
                    Dq[2 + 2 * self.nmu + self.ntend_loc + self.nind_loc + i, k] = (
                        self._Dparam(self.t[k], i + 1)
                        * psit1[k]
                        * (Dpsit[k] * Dpsitastpsit[k] + Dmut[k] * Dmutastpsit[k])
                    )  # alpha
        # Jacobian elements related to the scale parameters betaT2 and varphi (equation A.10)
        if self.ntend_sc > 0:
            Dq[2 + 2 * self.nmu + self.ntend_loc + self.nind_loc + 2 * self.npsi, :] = (
                (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit) * self.t * psit1
            )  # betaT2
        if self.nind_sc > 0:
            for i in range(self.nind_sc):
                Dq[
                    2
                    + 2 * self.nmu
                    + self.ntend_loc
                    + self.nind_loc
                    + 2 * self.npsi
                    + self.ntend_sc
                    + i,
                    :,
                ] = (
                    (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit)
                    * self.covariates[:, self.list_sc[i]]
                    * psit1
                )  # varphi2

        # Jacobian elements related to the shape parameters gamma0 and gamma (equation A.10)
        if self.neps0 == 1:
            Dq[
                2
                + 2 * self.nmu
                + self.ntend_loc
                + self.nind_loc
                + 2 * self.npsi
                + self.ntend_sc
                + self.nind_sc,
                :,
            ] = Depst + Dpsit * Dpsitastepst + Dmut * Dmutastepst
        # If shape harmonics are included
        if self.neps > 0:
            for i in range(self.neps):
                for k in range(len(self.t)):
                    Dq[
                        2
                        + self.neps0
                        + 2 * self.nmu
                        + self.ntend_loc
                        + self.nind_loc
                        + 2 * self.npsi
                        + self.ntend_sc
                        + self.nind_sc
                        + i,
                        k,
                    ] = (
                        Depst[k] + Dpsit[k] * Dpsitastepst[k] + Dmut[k] * Dmutastepst[k]
                    ) * self._Dparam(self.t[k], i + 1)
        # If shape covariates are included
        if self.nind_sh > 0:
            for i in range(self.nind_sh):
                Dq[
                    2
                    + self.neps0
                    + 2 * self.nmu
                    + self.ntend_loc
                    + self.nind_loc
                    + 2 * self.npsi
                    + self.ntend_sc
                    + self.nind_sc
                    + 2 * self.neps
                    + i,
                    :,
                ] = (
                    Depst + Dpsit * Dpsitastepst + Dmut * Dmutastepst
                ) * self.covariates[:, self.list_sh[i]]  # varphi3

        return Dq

    def _Dmupsiepst(self):
        """
        Calculates the derivatives of the standardized maximum with respect to parameters
        """

        t = self.t % 1
        nd = len(self.t)

        # Evaluate the parameters
        mut1, psit1, epst = self._evaluate_params(
            self.beta0,
            self.beta,
            self.alpha0,
            self.alpha,
            self.gamma0,
            self.gamma,
            self.betaT,
            self.varphi,
            self.betaT2,
            self.varphi2,
            self.varphi3,
            self.covariates[:, self.list_loc],
            self.covariates[:, self.list_sc],
            self.covariates[:, self.list_sh],
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
        Dermut = np.zeros((1 + 2 * self.nmu + self.ntend_loc + self.nind_loc, nd))
        Derpsit = np.zeros((1 + 2 * self.npsi + self.ntend_sc + self.nind_sc, nd))
        Derepst = np.zeros((self.neps0 + 2 * self.neps + self.nind_sh, nd))
        # Jacobian elements related to the location parameters beta0 and beta
        Dermut[0, :] = Dmut * Dmutastmut

        # If location harmonics are included
        if self.nmu > 0:
            for i in range(2 * self.nmu):
                for k in range(len(self.t)):
                    Dermut[i + 1, k] = (
                        Dmut[k] * Dmutastmut[k] * self._Dparam(t[k], i + 1)
                    )

        # Jacobian elements related to the location parameters betaT, varphi (equation A.9)
        if self.ntend_loc > 0:
            Dermut[1 + 2 * self.nmu, :] = Dmut * t * Dmutastmut  # betaT
        if self.nind_loc > 0:
            for i in range(self.nind_loc):
                Dermut[1 + 2 * self.nmu + self.ntend_loc + i, :] = (
                    Dmut * self.covariates[:, self.list_loc[i]] * Dmutastmut
                )  # varphi_i

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
        # Jacobian elements related to the scale parameters betaT2 and varphi (equation A.10)
        if self.ntend_sc > 0:
            Derpsit[1 + 2 * self.npsi, :] = (
                (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit) * t * psit1
            )  # betaT2
        if self.nind_sc > 0:
            for i in range(self.nind_sc):
                Derpsit[1 + 2 * self.npsi + self.ntend_sc + i, :] = (
                    (Dpsit * Dpsitastpsit + Dmut * Dmutastpsit)
                    * self.covariates[:, self.list_sc[i]]
                    * psit1
                )  # varphi2

        # Jacobian elements related to the shape parameters gamma0 and gamma (equation A.10)
        if self.neps0 == 1:
            Derepst[0, :] = Depst + Dpsit * Dpsitastepst + Dmut * Dmutastepst
        # If shape harmonics are included
        if self.neps > 0:
            for i in range(2 * self.neps):
                for k in range(len(self.t)):
                    Derepst[self.neps0 + i, k] = (
                        Depst[k] + Dpsit[k] * Dpsitastepst[k] + Dmut[k] * Dmutastepst[k]
                    ) * self._Dparam(t[k], i + 1)
        # If shape covariates are included
        if self.nind_sh > 0:
            for i in range(self.nind_sh):
                Derpsit[self.neps0 + 2 * self.neps + 1, :] = (
                    Depst + Dpsit * Dpsitastepst + Dmut * Dmutastepst
                ) * self.covariates[:, self.list_sh[i]]  # varphi3

        return Dermut, Derpsit, Derepst

    def _DQuantile(self):
        """
        Calculates the quantile derivative associated with a given parameterization with respect model parameters
        """

        t = self.t % 1
        nd = len(self.t)

        # Evaluate the parameters
        mut1, psit1, epst = self._evaluate_params(
            self.beta0,
            self.beta,
            self.alpha0,
            self.alpha,
            self.gamma0,
            self.gamma,
            self.betaT,
            self.varphi,
            self.betaT2,
            self.varphi2,
            self.varphi3,
            self.covariates[:, self.list_loc],
            self.covariates[:, self.list_sc],
            self.covariates[:, self.list_sh],
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
                + self.neps0
                + 2 * self.nmu
                + 2 * self.npsi
                + 2 * self.neps
                + self.ntend_loc
                + self.nind_loc
                + self.ntend_sc
                + self.nind_sc
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

        # Jacobian elements related to the location parameters betaT, varphi (equation A.9)
        if self.ntend_loc > 0:
            Dq[1 + 2 * self.nmu, :] = Dqmut * t * Dmutastmut  # betaT
        if self.nind_loc > 0:
            for i in range(self.nind_loc):
                Dq[1 + 2 * self.nmu + self.ntend_loc + i, :] = (
                    Dqmut * self.covariates[:, self.list_loc[i]] * Dmutastmut
                )  # varphi_i

        # Jacobian elements related to the scale parameters alpha0, alpha (equation A.7)
        Dq[1 + 2 * self.nmu + self.ntend_loc + self.nind_loc, :] = psit1 * (
            Dqpsit * Dpsitastpsit + Dqmut * Dmutastpsit
        )  # alpha0
        # If scale harmonic are included
        if self.npsi > 0:
            for i in range(2 * self.npsi):
                for k in range(len(self.t)):
                    Dq[2 + 2 * self.nmu + self.ntend_loc + self.nind_loc + i, k] = (
                        self._Dparam(t[k], i + 1)
                        * psit1[k]
                        * (Dqpsit[k] * Dpsitastpsit[k] + Dqmut[k] * Dmutastpsit[k])
                    )  # alpha
        # Jacobian elements related to the scale parameters betaT2 and varphi (equation A.10)
        if self.ntend_sc > 0:
            Dq[2 + 2 * self.nmu + self.ntend_loc + self.nind_loc + 2 * self.npsi, :] = (
                (Dqpsit * Dpsitastpsit + Dqmut * Dmutastpsit) * t * psit1
            )  # betaT2
        if self.nind_sc > 0:
            for i in range(self.nind_sc):
                Dq[
                    2
                    + 2 * self.nmu
                    + self.ntend_loc
                    + self.nind_loc
                    + 2 * self.npsi
                    + self.ntend_sc
                    + i,
                    :,
                ] = (
                    (Dqpsit * Dpsitastpsit + Dqmut * Dmutastpsit)
                    * self.covariates[:, self.list_sc[i]]
                    * psit1
                )  # varphi2

        # Jacobian elements related to the shape parameters gamma0 and gamma (equation A.10)
        if self.neps0 == 1:
            Dq[
                2
                + 2 * self.nmu
                + self.ntend_loc
                + self.nind_loc
                + 2 * self.npsi
                + self.ntend_sc
                + self.nind_sc,
                :,
            ] = Dqepst + Dqpsit * Dpsitastepst + Dqmut * Dmutastepst
        # If shape harmonics are included
        if self.neps > 0:
            for i in range(self.neps):
                for k in range(len(self.t)):
                    Dq[
                        2
                        + self.neps0
                        + 2 * self.nmu
                        + self.ntend_loc
                        + self.nind_loc
                        + 2 * self.npsi
                        + self.ntend_sc
                        + self.nind_sc
                        + i,
                        k,
                    ] = (
                        Dqepst[k]
                        + Dqpsit[k] * Dpsitastepst[k]
                        + Dqmut[k] * Dmutastepst[k]
                    ) * self._Dparam(t[k], i + 1)
        # If shape covariates are included
        if self.nind_sh > 0:
            for i in range(self.nind_sh):
                Dq[
                    2
                    + self.neps0
                    + 2 * self.nmu
                    + self.ntend_loc
                    + self.nind_loc
                    + 2 * self.npsi
                    + self.ntend_sc
                    + self.nind_sc
                    + 2 * self.neps
                    + i,
                    :,
                ] = (
                    Dqepst + Dqpsit * Dpsitastepst + Dqmut * Dmutastepst
                ) * self.covariates[:, self.list_sc[i]]  # varphi3

        return Dq

    def PPplot(self):
        """
        PP plot
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
            and self.ntend_loc == 0
            and self.ntend_sc == 0
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
        plt.title(f"Best model PP plot ({self.example})")
        plt.xlabel("Empirical")
        plt.ylabel("Fitted")
        plt.grid(True)
        plt.axis("square")
        plt.margins(x=0.1)
        plt.show()

    def _CDFGEVt(self):
        """
        Calculates the GEV distribution function corresponding to the given parameters
        """

        F = np.zeros(len(self.xt))

        # Evaluate the parameters
        mut1, psit1, epst = self._evaluate_params(
            self.beta0,
            self.beta,
            self.alpha0,
            self.alpha,
            self.gamma0,
            self.gamma,
            self.betaT,
            self.varphi,
            self.betaT2,
            self.varphi2,
            self.varphi3,
            self.covariates[:, self.list_loc],
            self.covariates[:, self.list_sc],
            self.covariates[:, self.list_sh],
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

    def ReturnPeriodPlot(self, annualplot=True):
        """
        Funtion to plot the Aggregated Return period plot for each month and if annualplot, the annual Return period (default True)
        """

        # Ts = np.array([2, 5, 10, 20, 25, 50, 75, 100, 200, 300, 400, 500])
        Ts = np.concatenate(
            (np.arange(2, 10, 1), np.arange(10, 100, 10), np.arange(100, 501, 100))
        )

        nts = len(Ts)
        quanaggrA = np.zeros(nts)
        quanaggr = np.zeros((12, nts))
        stdDqX = np.zeros((12, nts))
        for i in range(12):
            for j in range(nts):
                quanaggr[i, j] = self._aggquantile(1 - 1 / Ts[j], i / 12, (i + 1) / 12)[
                    0
                ]
                stdQuan = self._ConfidInterQuanAggregate(
                    1 - 1 / Ts[j], i / 12, (i + 1) / 12
                )
                stdDqX[i, j] = stdQuan * norm.ppf(
                    1 - (1 - self.quanval) / 2, loc=0, scale=1
                )

        # If annual data has to be plotted
        if annualplot:
            for j in range(nts):
                quanaggrA[j] = self._aggquantile(1 - 1 / Ts[j], 0, 1)[0]
            # Confidence intervals
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
        datemax_mod = self.t % 1
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
        plt.figure(figsize=(10, 6))
        for i in range(12):
            plt.semilogx(
                Ts,
                quanaggr[i, :],
                color=colors[i],
                linestyle="-",
                linewidth=1.2,
                label=labels[i],
            )

        # Anual return periods
        if annualplot:
            plt.semilogx(Ts, quanaggrA, color="black", linewidth=2, label="Annual")
            ny = int(np.ceil(self.t[-1]))
            hmax1 = np.zeros(ny)
            for j in range(ny):
                hmax1[j] = np.max(
                    self.xt[np.where((self.t >= j) & (self.t < j + 1))[0]]
                )
            hmaxsort = np.sort(hmax1)
            ProHsmaxsort = np.arange(1, len(hmaxsort) + 1) / (len(hmaxsort) + 1)
            Tapprox = 1 / (1 - ProHsmaxsort)
            idx = np.where(Tapprox >= 2)[0]
            plt.semilogx(Tapprox[idx], hmaxsort[idx], "ok", markersize=1.6)
            plt.semilogx(Ts, stdlo, "--k", linewidth=1.1)
            plt.semilogx(Ts, stdup, "--k", linewidth=1.1)
        plt.xlabel("Return Period (years)")
        plt.ylabel(r"$H_s^{max}$")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.xticks([1, 2, 5, 10, 20, 50, 100, 250, 500])
        plt.xlim(left=1.8, right=Ts[-1] + 50)
        plt.ylim(bottom=0)
        plt.title(f"Aggregate Quantiles ({self.example})")
        plt.grid(True)
        plt.margins(x=0.1)
        plt.show()

        ### plt.figure(figsize=(10,6))
        ### for i in range(nts):
        ###     col = np.random.uniform(0, 1, 3)
        ###     plt.plot(np.arange(1, ny + 1) - 0.5, quanaggr[i,:], color=col)
        ### plt.xlabel("Return Period (years)")
        ### plt.ylabel(r"$H_s^{max}$")
        ### plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ### plt.xticks([1, 2, 5, 10, 20, 50, 100, 250, 500])
        ### plt.xlim(left=1.8,right=Ts[-1]+50)
        ### plt.ylim(bottom=0)
        ### plt.title(f"Aggregate Quantiles ({self.example})")
        ### plt.grid(True)
        ### plt.margins(x=0.1)
        ### plt.show()

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
        betaT2=None,
        varphi=None,
        varphi2=None,
    ):
        """
        Function to compute the aggregated quantile for certain parameters
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
        if betaT2 is None:
            betaT2 = self.betaT2
        if varphi is None:
            varphi = self.varphi
        if varphi2 is None:
            varphi2 = self.varphi2

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
            cov_locint = np.zeros(len(varphi))
            cov_scint = np.zeros(len(varphi2))
            if len(pos) > 0:
                for i in range(len(varphi)):
                    cov_locint[i] = np.mean(self.covariates[pos, self.list_loc[i]])
                for i in range(len(varphi2)):
                    cov_scint[i] = np.mean(self.covariates[pos, self.list_sc[i]])
        else:
            cov_locint = None
            cov_scint = None

        # Require quantile
        zqout = np.zeros(m)

        media = quad(
            lambda x: self._parametro(
                beta0,
                beta,
                betaT,
                self.covariates[:, self.list_loc],
                varphi,
                cov_locint,
                self.t,
                x,
            ),
            0,
            1,
        )
        std = quad(
            lambda x: np.exp(
                self._parametro(
                    alpha0,
                    alpha,
                    betaT2,
                    self.covariates[:, self.list_sc],
                    varphi2,
                    cov_scint,
                    self.t,
                    x,
                )
            ),
            0,
            1,
        )

        for il in range(m):
            # for jl in range(n)
            zq = media
            err: float = 1
            iter1: int = 1
            integ: float = 0
            while err > 1e-4 and iter1 < 1000:
                zqold = zq
                integ = quad(
                    lambda x: self._fzeroquanint(
                        x,
                        zqold,
                        q[il],
                        cov_locint,
                        cov_scint,
                        beta0,
                        beta,
                        alpha0,
                        alpha,
                        gamma0,
                        gamma,
                        betaT,
                        betaT2,
                        varphi,
                        varphi2,
                    ),
                    float(t0[il]),
                    float(t1[il]),
                )[0]
                integ += np.log(q[il]) / 12
                dint = quad(
                    lambda x: self._fzeroderiquanint(
                        x,
                        zqold,
                        q[il],
                        cov_locint,
                        cov_scint,
                        beta0,
                        beta,
                        alpha0,
                        alpha,
                        gamma0,
                        gamma,
                        betaT,
                        betaT2,
                        varphi,
                        varphi2,
                    ),
                    float(t0[il]),
                    float(t1[il]),
                )[0]
                zq += -integ / dint
                if np.abs(zq) > 1e-5:
                    err = np.abs((zq - zqold) / zqold)
                else:
                    err = np.abs(zq - zqold)
                iter1 += 1
            if iter1 == 1000:
                zq = np.nan
                Warning("Maximum number of Newton iterations")
            if integ > 1e-2:
                zq = np.nan
                Warning("False zero, check it")
            zqout[il] = zq

        return zqout

    def _fzeroquanint(
        self,
        t,
        zq,
        q,
        indicesint,
        indices2int,
        beta0,
        beta,
        alpha0,
        alpha,
        gamma0,
        gamma,
        betaT,
        betaT2,
        varphi,
        varphi2,
    ):
        """
        Function to solve the quantile
        """

        # Evaluate the location parameter at each time t as a function of the actual values of the parameters given by p
        mut1 = self._parametro(
            beta0,
            beta,
            betaT,
            self.covariates[:, self.list_loc],
            varphi,
            indicesint,
            self.t,
            t,
        )
        # Evaluate the scale parameter at each time t as a function of the actual values of the parameters given by p
        psit1 = np.exp(
            self._parametro(
                alpha0,
                alpha,
                betaT2,
                self.covariates[:, self.list_sc],
                varphi2,
                indices2int,
                self.t,
                t,
            )
        )
        # Evaluate the sahpe parameter at each time t as a function of the actual values of the parameters given by p
        epst = self._parametro(alpha0=gamma0, alpha=gamma, x=t)

        # The values whose shape parameter is almost cero corresponds to the GUMBEL distribution, locate their positions if they exist
        posG = np.where(np.abs(epst) <= 1e-8)[0]
        # The remaining values correspond to WEIBULL or FRECHET
        pos = np.where(np.abs(epst) > 1e-8)[0]
        # The corresponding GUMBEl values are set to 1 to avoid numerical problems, note that for those cases the GUMBEL expressions are used
        epst[posG] = 1

        # TODO: AÑadir
        #### if times is not None:
        ####    kt2 = spline

        mut = mut1
        psit = psit1
        mut[pos] = mut1[pos] + psit1[pos] * (self.kt[pos] ** epst[pos] - 1) / epst[pos]
        psit[pos] = psit1[pos] * self.kt[pos] ** epst[pos]
        # Modify the parameters to include Gumbel
        mut[posG] += psit[posG] * np.log(self.kt[posG])

        # Evaluate the auxiliary variable
        xn = (zq - mut) / psit
        z = 1 + epst * xn
        # Since the z-values must be greater than zero in order to avoid numerical problems their values are set to be greater than 1e-4
        z = np.maximum(1e-8, z)
        zn = z ** (-1 / epst)
        # GUMBEL case
        zn[posG] = np.exp(-xn[posG])

        return zn

    def _fzeroderiquanint(
        self,
        t,
        zq,
        q,
        indicesint,
        indices2int,
        beta0,
        beta,
        alpha0,
        alpha,
        gamma0,
        gamma,
        betaT,
        betaT2,
        varphi,
        varphi2,
    ):
        """
        Function to solve the quantile
        """
        # Evaluate the location parameter at each time t as a function of the actual values of the parameters given by p
        mut1 = self._parametro(
            beta0,
            beta,
            betaT,
            self.covariates[:, self.list_loc],
            varphi,
            indicesint,
            self.t,
            t,
        )
        # Evaluate the scale parameter at each time t as a function of the actual values of the parameters given by p
        psit1 = np.exp(
            self._parametro(
                alpha0,
                alpha,
                betaT2,
                self.covariates[:, self.list_sc],
                varphi2,
                indices2int,
                self.t,
                t,
            )
        )
        # Evaluate the sahpe parameter at each time t as a function of the actual values of the parameters given by p
        epst = self._parametro(alpha0=gamma0, alpha=gamma, x=t)

        # The values whose shape parameter is almost cero corresponds to the GUMBEL distribution, locate their positions if they exist
        posG = np.where(np.abs(epst) <= 1e-8)[0]
        # The remaining values correspond to WEIBULL or FRECHET
        pos = np.where(np.abs(epst) > 1e-8)[0]
        # The corresponding GUMBEl values are set to 1 to avoid numerical problems, note that for those cases the GUMBEL expressions are used
        epst[posG] = 1

        # TODO: AÑadir
        #### if times is not None:
        ####    kt2 = spline

        mut = mut1
        psit = psit1
        mut[pos] = mut1[pos] + psit1[pos] * (self.kt[pos] ** epst[pos] - 1) / epst[pos]
        psit[pos] = psit1[pos] * self.kt[pos] ** epst[pos]
        # Modify the parameters to include Gumbel
        mut[posG] += psit[posG] * np.log(self.kt[posG])

        # Evaluate the auxiliary variable
        xn = (zq - mut) / psit
        z = 1 + epst * xn
        # Since the z-values must be greater than zero in order to avoid numerical problems their values are set to be greater than 1e-4
        z = np.maximum(1e-8, z)
        zn = z ** (-1 - 1 / epst) / psit
        # GUMBEL case
        zn[posG] = -np.exp(-xn[posG]) / psit[posG]

        return zn

    def _ConfidInterQuanAggregate(self, q, t0, t1):
        """
        Auxiliar function to compute the std for the aggregated quantiles
        """
        # Total length of the data
        n = (
            2
            + self.neps0
            + 2 * self.nmu
            + 2 * self.npsi
            + 2 * self.neps
            + self.ntend_loc
            + self.nind_loc
            + self.ntend_sc
            + self.nind_sc
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
        if self.ntend_loc != 0:
            aux += 1
            jacob[aux] = (
                self._aggquantile(q, t0, t1, betaT=self.betaT * (1 + epsi))[0]
                - self._aggquantile(q, t0, t1, betaT=self.betaT * (1 - epsi))[0]
            ) / (2 * self.betaT * epsi)

        # varphi derivative
        if self.nind_loc != 0:
            for i in range(self.nind_loc):
                aux += 1
                if self.varphi[i] != 0:
                    varphilb = self.varphi
                    varphiub = self.varphi
                    varphilb[i] = self.varphi[i] * (1 + epsi)
                    varphiub[i] = self.varphi[i] * (1 - epsi)
                    jacob[aux] = (
                        self._aggquantile(q, t0, t1, varphi=varphilb[i])[0]
                        - self._aggquantile(q, t0, t1, varphi=varphilb[i])[0]
                    ) / (2 * self.varphi[i] * epsi)
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

        # betaT derivative
        if self.ntend_sc != 0:
            aux += 1
            jacob[aux] = (
                self._aggquantile(q, t0, t1, betaT2=self.betaT2 * (1 + epsi))[0]
                - self._aggquantile(q, t0, t1, betaT2=self.betaT2 * (1 - epsi))[0]
            ) / (2 * self.betaT2 * epsi)

        # varphi derivative
        if self.nind_sc != 0:
            for i in range(self.nind_sc):
                aux += 1
                if self.varphi2[i] != 0:
                    varphilb2 = self.varphi2
                    varphiub2 = self.varphi2
                    varphilb2[i] = self.varphi2[i] * (1 + epsi)
                    varphiub2[i] = self.varphi2[i] * (1 - epsi)
                    jacob[aux] = (
                        self._aggquantile(q, t0, t1, varphi2=varphilb2[i])[0]
                        - self._aggquantile(q, t0, t1, varphi2=varphilb2[i])[0]
                    ) / (2 * self.varphi2[i] * epsi)
                else:
                    jacob[aux] = 0

        # gamma0 derivative
        if self.neps0 != 0:
            aux += 1
            jacob[aux] = (
                self._aggquantile(q, t0, t1, gamma0=self.gamma0 * (1 + epsi))[0]
                - self._aggquantile(q, t0, t1, gamma0=self.gamma0 * (1 - epsi))[0]
            ) / (2 * self.gamma0 * epsi)

        if self.neps != 0:
            for i in range(2 * self.neps):
                aux += 1
                gamma1 = self.gamma
                gamma2 = self.gamma
                gamma2[i] = self.gamma[i] * (1 + epsi)
                gamma1[i] = self.gamma[i] * (1 - epsi)
                jacob[aux] = (
                    self._aggquantile(q, t0, t1, gamma=gamma2)[0]
                    - self._aggquantile(q, t0, t1, gamma=gamma1)[0]
                ) / (2 * self.gamma[i] * epsi)

        # Computing the standard deviations for the quantiles
        stdQuan = np.sqrt(jacob.T @ self.invI0 @ jacob)

        return stdQuan
