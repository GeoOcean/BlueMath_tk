import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np
import pandas as pd

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
            self.loglike_iter[iter] = -fit_result["loglikelihood"]
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
            self.AIC_iter[iter] = self._AIC(-fit_result["loglikelihood"], n_params)
            
            ### Step 4: Sensitivity of optimal loglikelihood respect to possible additional harmonics 
            # for the location, scale and shape parameters. 
            # Note that the new parameter values are set to zero since derivatives do not depend on them
            fit_result_aux = fit_result.copy()
            # Location
            if fit_result['beta'] is not None:
                fit_result_aux['beta'] = np.concatenate((fit_result['beta'] , [0, 0]))
            else:
                fit_result_aux['beta'] = np.array([0, 0])
            # Scale
            if fit_result['alpha'] is not None:
                fit_result_aux['alpha'] = np.concatenate((fit_result['alpha'] , [0, 0]))
            else:
                fit_result_aux['alpha'] = np.array([0, 0])
            # Shape
            if fit_result['gamma'] is not None:
                fit_result_aux['gamma'] = np.concatenate((fit_result['gamma'] , [0, 0]))
            else:
                fit_result_aux['gamma'] = np.array([0, 0])
            
            auxf, auxJx, auxHxx = self._loglikelihood(**fit_result_aux) # TODO: ADD THE PARAMETERS

            # Inverse of the Information Matrix (auxHxx) 
            auxI0 = np.linalg.inv(-auxHxx)

            # Updating the best model
            if iter > 0:
                # TODO: Implement another criterias (Proflike)
                if self.AIC_iter[iter] < self.AIC_iter[iter-1]:
                    modelant = np.array([nmu, npsi, ngamma])
            else:
                modelant = np.array([nmu, npsi, ngamma])
            
            ### Step 5: Compute maximum perturbation
            pos = 1
            max_val = np.abs(auxJx[2*nmu:2*nmu+2].T @ auxI0[2*nmu:2*nmu+2, 2*nmu:2*nmu+2] @ auxJx[2*nmu:2*nmu+2])
            # TODO: AÑADIR BIEN LAS COVARIABLES EN EL PARAMETRO DE FORMA
            auxmax = abs(auxJx[2 + 2*nmu + ntrend_loc + nind_loc + 2*npsi + 2:2 + 2*nmu + ntrend_loc + nind_loc + 2*npsi + 2+2].T @ auxI0[2 + 2*nmu + ntrend_loc + nind_loc + 2*npsi + 2:2 + 2*nmu + ntrend_loc + nind_loc + 2*npsi + 2+2, 2 + 2*nmu + ntrend_loc + nind_loc + 2*npsi + 2:2 + 2*nmu + ntrend_loc + nind_loc + 2*npsi + 2+2] @ auxJx[2 + 2*nmu + ntrend_loc + nind_loc + 2*npsi + 2:2 + 2*nmu + ntrend_loc + nind_loc + 2*npsi + 2+2])
            if auxmax > max_val:
                max_val = auxmax
                pos = 2

            # TODO: AÑADIR BIEN LAS COVARIABLES EN EL PARAMETRO DE FORMA
            auxmax = abs(auxJx[2 + self.ngamma0 + 2*nmu + ntrend_loc + nind_loc + ntrend_sc + nind_sc + 2*npsi + 2*ngamma + 4 : 2 + self.ngamma0 + 2*nmu + ntrend_loc + nind_loc + ntrend_sc + nind_sc + 2*npsi + 2*ngamma + 4+2].T @ auxI0[2 + self.ngamma0 + 2*nmu + ntrend_loc + nind_loc + ntrend_sc + nind_sc + 2*npsi + 2*ngamma + 4 : 2 + self.ngamma0 + 2*nmu + ntrend_loc + nind_loc + ntrend_sc + nind_sc + 2*npsi + 2*ngamma + 4+2, 2 + self.ngamma0 + 2*nmu + ntrend_loc + nind_loc + ntrend_sc + nind_sc + 2*npsi + 2*ngamma + 4 : 2 + self.ngamma0 + 2*nmu + ntrend_loc + nind_loc + ntrend_sc + nind_sc + 2*npsi + 2*ngamma + 4+2] @ auxJx[2 + self.ngamma0 + 2*nmu + ntrend_loc + nind_loc + ntrend_sc + nind_sc + 2*npsi + 2*ngamma + 4 : 2 + self.ngamma0 + 2*nmu + ntrend_loc + nind_loc + ntrend_sc + nind_sc + 2*npsi + 2*ngamma + 4+2])
            if auxmax>max_val:
                max_val=auxmax
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
                if self.AIC_iter[iter] >= self.AIC_iter[iter-1]:
                    model = modelant
                    self.AICini = self.AIC_iter[iter-1]
                    loglikeobjITini = self.loglike_iter[iter-1]
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
        self.ngamma0 = 0 # Force the elimination of the constant shape parameter (gamma0)
        fit_result = self._fit(nmu,npsi,ngamma)

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
        self.AIC_iter[self.niter_harm + 1] = self._AIC(-fit_result["loglikelihood"], n_params)
        self.loglike_iter[self.niter_harm + 1] = -fit_result["loglikelihood"]

        if self.AICini < self.AIC_iter[self.niter_harm + 1]:
            # The constant shape parameter (gamma0) is significative
            self.ngamma0 = 1
            fit_result = self._fit(nmu,npsi,ngamma)
        
        print("Harmonic AIC:", self.AICini, "\n")

        self._update_params(fit_result)
        self.nmu = nmu
        self.npsi = npsi
        self.ngamma = ngamma

        ######### COVARIATES Iterative process #########
        nrows, nind_cov = self.covariates.shape # Number of covariates

        # Auxiliar variables related to location parameter
        beta_cov = np.asarray([])
        list_loc = []   # List of covariates for location
        nind_loc = 0    # TODO: IT IS ZERO BY DEFAULT, CHECK IF IT POSSIBLE TO REMOVE THIS LINE
        auxcov_loc = self.covariates.iloc[:, list_loc].values
        # Auxiliar variables related to scale parameter
        alpha_cov = np.asarray([])
        list_sc = []
        nind_sc = 0     # TODO: IT IS ZERO BY DEFAULT, CHECK IF IT POSSIBLE TO REMOVE THIS LINE
        auxcov_sc = self.covariates.iloc[:, list_sc].values
        # Auxiliar variables related to shape parameter
        gamma_cov = np.asarray([])
        list_sh = []
        nind_sh = 0     # TODO: IT IS ZERO BY DEFAULT, CHECK IF IT POSSIBLE TO REMOVE THIS LINE
        auxcov_sh = self.covariates.iloc[:, list_sh].values

        if self.covariates is None:
            print("No covariates provided, skipping Covariates iterative process")
        else: 
            print("Starting Covariates iterative process")
            for iter in range(self.niter_harm + 1, self.max_iter):
                
                self.ngamma0 = 1

                ### Step 9: Calculate the sensitivities of the optimal log-likelihood objective function with respect to possible 
                # additional covariates for the location and  scale parameters
                auxf, auxJx, auxHxx = self._loglikelihood() # TODO: ADD THE PARAMETERS

                # Step 10: Include in the parameter vector the corresponding covariate
                auxI0 = np.linalg.inv(-auxHxx)
                values1 = np.abs(auxJx[1+2*nmu+ntrend_loc : 1+2*nmu+ntrend_loc+nind_cov]**2 / np.diag(auxI0[1+2*nmu+ntrend_loc : 1+2*nmu+ntrend_loc+nind_cov, 1+2*nmu+ntrend_loc: 1+2*nmu+ntrend_loc+nind_cov]))
                maximo_loc, pos_loc = np.max(values1), np.argmax(values1)

                values2 = np.abs(auxJx[2+2*nmu+ntrend_loc+nind_cov+2*npsi+ntrend_sc : 2+2*nmu+ntrend_loc+nind_cov+2*npsi+ntrend_sc+nind_cov]**2 / np.diag(auxI0[2+2*nmu+ntrend_loc+nind_cov+2*npsi+ntrend_sc : 2+2*nmu+ntrend_loc+nind_cov+2*npsi+ntrend_sc+nind_cov, 2+2*nmu+ntrend_loc+nind_cov+2*npsi+ntrend_sc : 2+2*nmu+ntrend_loc+nind_cov+2*npsi+ntrend_sc+nind_cov]))
                maximo_sc, pos_sc = np.max(values2), np.argmax(values2)

                values3 = np.abs(auxJx[2+self.ngamma0+2*nmu+ntrend_loc+nind_cov+2*npsi+ntrend_sc+nind_cov+2*ngamma : 2+self.ngamma0+2*nmu+ntrend_loc+nind_cov+2*npsi+ntrend_sc+nind_cov+2*ngamma+nind_cov]**2 / np.diag(auxI0[2+self.ngamma0+2*nmu+ntrend_loc+nind_cov+2*npsi+ntrend_sc+nind_cov+2*ngamma : 2+self.ngamma0+2*nmu+ntrend_loc+nind_cov+2*npsi+ntrend_sc+nind_cov+2*ngamma+nind_cov, 2+self.ngamma0+2*nmu+ntrend_loc+nind_cov+2*npsi+ntrend_sc+nind_cov+2*ngamma : 2+self.ngamma0+2*nmu+ntrend_loc+nind_cov+2*npsi+ntrend_sc+nind_cov+2*ngamma+nind_cov]))
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
                    beta_cov_init = np.append(beta_cov_init, [0])   # Initialize the new covariate as zero
                elif posmaxparam == 1:
                    # Add covariate to scale
                    nind_sc += 1
                    list_sc.append(int(pos_sc))
                    alpha_cov_init = np.append(alpha_cov_init, [0])   # Initialize the new covariate as zero
                elif posmaxparam == 2:
                    # Add covariate to shape
                    nind_sh += 1
                    list_sh.append(int(pos_sh))
                    gamma_cov_init = np.append(gamma_cov_init, [0])   # Initialize the new covariate as zero

                # Update auxiliar covariates
                auxcov_loc = self.covariates.iloc[:, list_loc].values
                auxcov_sc = self.covariates.iloc[:, list_sc].values
                auxcov_sh = self.covariates.iloc[:, list_sh].values

                ### Step 11: Obtain the maximum-likelihood estimators for the selected parameters and 
                # calculate the Akaike Information criterion objective function AIC
                # TODO: DEFINE PROPERLY
                # concatvalues = [popt[0:1 + 2 * nmu], varphiini_loc, popt[1 + 2 * nmu : 2 + 2 * nmu + 2 * npsi], varphiini_sc, np.zeros(self.neps0), np.zeros(2 * neps), varphiini_sh]
                # pini = np.concatenate([np.asarray(v) for v in concatvalues if v is not None])
                fit_result = self._fit(self.nmu, self.npsi, self.ngamma)

                # Check if model is Gumbel
                # self.ngamma0 = 
                n_params = (2+self.ngamma0+2*nmu+2*npsi+2*ngamma+ntrend_loc+nind_loc+ntrend_sc+nind_sc+ntrend_sh+nind_sh)
                self.AIC_iter[iter] = self._AIC(-fit_result["loglikelihood"], n_params)
                self.loglike_iter[iter] = -fit_result["loglikelihood"]

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

            # concatvalues = [popt[0:1 + 2 * nmu], np.zeros(ntend_loc), varphiini_loc, popt[1 + 2 * nmu : 2 + 2 * nmu + 2 * npsi], varphiini_sc, np.zeros(self.neps0), np.zeros(2 * neps), varphiini_sh]
            # pini = np.concatenate([np.asarray(v) for v in concatvalues if v is not None])
            fit_result = self._fit()

            n_params = (2+self.ngamma0+2*nmu+2*npsi+2*ngamma+ntrend_loc+nind_loc+ntrend_sc+nind_sc+ntrend_sh+nind_sh)
            self.AIC_iter[self.nit + 1] = self._AIC(-fit_result["loglikelihood"], n_params)
            self.loglike_iter[self.nit + 1] = -fit_result["loglikelihood"]

            if self.AIC_iter[self.nit + 1] < self.AIC_iter[self.nit]:
                self.AICini = self.AIC_iter[self.nit+1]
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

            # concatvalues = [popt[0:1 + 2 * nmu], np.zeros(ntend_loc), varphiini_loc, popt[1 + 2 * nmu : 2 + 2 * nmu + 2 * npsi], varphiini_sc, np.zeros(self.neps0), np.zeros(2 * neps), varphiini_sh]
            # pini = np.concatenate([np.asarray(v) for v in concatvalues if v is not None])
            fit_result = self._fit()

            n_params = (2+self.ngamma0+2*nmu+2*npsi+2*ngamma+self.ntrend_loc+nind_loc+ntrend_sc+nind_sc+ntrend_sh+nind_sh)
            self.AIC_iter[self.nit + 2] = self._AIC(-fit_result["loglikelihood"], n_params)
            self.loglike_iter[self.nit + 2] = -fit_result["loglikelihood"]

            if self.AIC_iter[self.nit + 2] < self.AIC_iter[self.nit]:
                self.AICini = self.AIC_iter[self.nit+2]
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

            # concatvalues = [popt[0:1 + 2 * nmu], np.zeros(ntend_loc), varphiini_loc, popt[1 + 2 * nmu : 2 + 2 * nmu + 2 * npsi], varphiini_sc, np.zeros(self.neps0), np.zeros(2 * neps), varphiini_sh]
            # pini = np.concatenate([np.asarray(v) for v in concatvalues if v is not None])
            fit_result = self._fit()

            n_params = (2+self.ngamma0+2*nmu+2*npsi+2*ngamma+self.ntrend_loc+nind_loc+self.ntrend_sc+nind_sc+ntrend_sh+nind_sh)
            self.AIC_iter[self.nit + 3] = self._AIC(-fit_result["loglikelihood"], n_params)
            self.loglike_iter[self.nit + 3] = -fit_result["loglikelihood"]

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









            




                


                




    def _fit(
        self,
        nmu,
        mpsi,
        ngamma,
        nind_loc,
        ntrend_loc,
        nind_sc,
        ntrend_sc,
        nind_sh,
        ntrend_sh,
    ) -> dict:
        """
        Auxiliar function to determine the optimal parameters of given Non-Stationary GEV
        """
        # TODO
        ...



    def fit(self) -> dict:
        """
        Function to determine the optimal parameters of given Non-Stationary GEV considering all the covariates
        """
        # TODO
        ...


    def _AIC(self) -> float: 
        # TODO
        ...


    def _loglikelihood(self, beta0=None,beta=None,alpha0=None,alpha=None,gamma0=None,gamma=None,betaT=None,varphi=None,betaT2=None,varphi2=None,varphi3=None,covariates_loc=None,covariates_sc=None,covariates_sh=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
        """
        Function to calculate the loglikelihood function, the Jacobian and the Hessian for a given parameterization

        Parameters
        ----------
        beta0 : 
            Optimal constant parameter related to location
        beta : 
            Optimal harmonic vector associated with location
        alpha0 : 
            Optimal constant parameter related to scale
        alpha : 
            Optimal harmonic vector associated with scale
        gamma0 : 
            Optimal constant parameter related to shape
        gamma : 
            Optimal harmonic vector associated with shape
        betaT : 
            Optimal location trend parameter
        varphi : 
            Optimal location covariate vector
        betaT2 : 
            Optimal scale trend parameter
        varphi2 : 
            Optimal scale covariate vector
        varphi3 : 
            Optimal shape covariate vector
        covariates_loc : 
            covariates data related to the location parameter, a matrix including the data at time t for each covariate
        covariates_sc : 
            covariates data related to the scale parameter, a matrix including the data at time t for each covariate
        covariates_sh : 
            covariates data related to the shape parameter, a matrix including the data at time t for each covariate

        Returuns
        ----------
        f : np.ndarray
            Optimal loglikelihood function
        Jx : np.ndarray
            Gradient of the log-likelihood function at the optimal solution
        Hxx : np.ndarray
            Hessian of the log-likelihood function at the optimal solution 
        """

        if beta0 is None:
            beta0 = np.empty(0)
        if beta is None:
            beta = np.empty(0)
        if alpha0 is None:
            alpha0 = np.empty(0)
        if alpha is None:
            alpha = np.empty(0)
        if gamma0 is None:
            gamma0 = np.empty(0)
        if gamma is None:
            gamma = np.empty(0)
        if betaT is None or betaT.size == 0:
            betaT = np.empty(0)
            ntend_loc = 0
        else:
            ntend_loc = 1
        if varphi is None:
            varphi = np.empty(0)
        if betaT2 is None or betaT2.size == 0:
            betaT2 = np.empty(0)
            ntend_sc = 0
        else:
            ntend_sc = 1
        if varphi2 is None:
            varphi2 = np.empty(0)
        if varphi3 is None:
            varphi3 = np.empty(0)
        if covariates_loc is None:
            covariates_loc = np.empty((0,0))
        if covariates_sc is None:
            covariates_sc = np.empty((0,0))
        if covariates_sh is None:
            covariates_sh = np.empty((0,0))
        
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


        
        nmu = len(beta)
        npsi = len(alpha)
        neps = len(gamma)
        #ntend_loc = len(betaT)
        nind_loc = len(varphi)
        #ntend_sc = len(betaT2)
        nind_sc = len(varphi2)
        nind_sh = len(varphi3)

        # Evaluate the parameters
        mut1, psit1, epst = self._evaluate_params(beta0,beta,alpha0,alpha,gamma0,gamma,betaT,varphi,betaT2,varphi2,varphi3,covariates_loc,covariates_sc,covariates_sh)


        # The values whose shape parameter is almost 0 correspond to the Gumbel distribution
        posG = np.where(np.abs(epst) <= 1e-8)[0]
        # The remaining values correspond to Weibull or Frechet
        pos = np.where(np.abs(epst) > 1e-8)[0]

        # The corresponding Gumbel values are set to 1 to avoid numerical problems, note that in this case, the Gumbel expressions are used
        epst[posG] = 1

        # Modify the parameters to include the length of the data
        mut = mut1
        psit = psit1
        mut[pos] = mut1[pos]+psit1[pos]*(self.kt[pos]**epst[pos]-1)/epst[pos]
        psit[pos] = psit1[pos]*self.kt[pos]**epst[pos]
        # Modify the parameters to include the length of the data in Gumbel
        mut[posG] = mut[posG] + psit[posG]*np.log(self.kt[posG])

        # Evaluate auxiliarly variables
        xn = (self.xt-mut)/psit
        z = 1+epst*xn
        
        # Since z-values must be greater than 0 in order to avoid numerical problems, their values are set to be greater than 1e-8
        z = np.maximum(1e-8, z)
        zn = z**(-1/epst)

        # Evaluate the loglikelihood function, not that the general and Gumbel expressions are used
        f = - np.sum(-np.log(self.kt[pos]) + np.log(psit[pos]) + (1+1/epst[pos])*np.log(z[pos])+self.kt[pos]*zn[pos]) - \
            np.sum(-np.log(self.kt[posG]) + np.log(psit[posG]) + xn[posG] + self.kt[posG]*np.exp(-xn[posG]))
        

        ### Gradient of the loglikelihood
        # Derivatives given by equations (A.1)-(A.3) in the paper
        Dmut = (1+epst-self.kt*zn) / (psit*z)
        Dpsit = -(1-xn*(1-self.kt*zn)) / (psit*z)
        Depst = zn * (xn*(self.kt-(1+epst)/zn)+z*(-self.kt+1/zn)*np.log(z)/epst) / (epst*z)

        # Gumbel derivatives given by equations (A.4)-(A.5) in the paper
        Dmut[posG] = (1-self.kt[posG]*np.exp(-xn[posG])) / psit[posG]
        Dpsit[posG] = (xn[posG]-1-self.kt[posG]*xn[posG]*np.exp(-xn[posG])) / (psit[posG])
        Depst[posG] = 0


        ## New Derivatives
        Dmutastmut = np.ones_like(self.kt)
        Dmutastpsit = (-1 + self.kt ** epst) / epst
        Dmutastepst = psit1 * (1 + (self.kt ** epst) * (epst * np.log(self.kt) - 1)) / (epst ** 2)

        Dpsitastpsit = self.kt ** epst
        Dpsitastepst = np.log(self.kt) * psit1 * (self.kt ** epst)

        Dmutastpsit[posG] = np.log(self.kt[posG])
        Dmutastepst[posG] = 0

        Dpsitastpsit[posG] = 1
        Dpsitastepst[posG] = 0

        # Set the Jacobian to zero
        Jx = np.zeros(2 + self.neps0 + nmu + npsi + neps + ntend_loc + nind_loc + ntend_sc + nind_sc + nind_sh)
        # Jacobian elements related to the location parameters beta0 and beta, equation (A.6) in the paper
        Jx[0] = np.dot(Dmut,Dmutastmut)

        # If location harmonics are included
        if nmu > 0:
            for i in range(nmu):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dmut[k]*Dmutastmut[k]*self._Dparam(tt, i+1)
                Jx[i+1] = aux
        
        # Jacobian elements related to the location parameters betaT, varphi (equation A.9)
        if ntend_loc > 0:
            Jx[1+nmu] = np.sum(Dmut*self.t*Dmutastmut)  # betaT
        if nind_loc > 0:
            for i in range(nind_loc):
                Jx[1+nmu+ntend_loc+i] = np.sum(Dmut*covariates_loc[:,i]*Dmutastmut) # varphi_i
        
        # Jacobian elements related to the scale parameters alpha0, alpha (equation A.7)
        Jx[1+nmu+ntend_loc+nind_loc] = np.sum(psit1*(Dpsit*Dpsitastpsit+Dmut*Dmutastpsit))  # alpha0
        # If scale harmonic are included
        if npsi > 0:
            for i in range(npsi):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += self._Dparam(tt, i+1)*psit1[k]*(Dpsit[k]*Dpsitastpsit[k]+Dmut[k]*Dmutastpsit[k])
                Jx[2+nmu+ntend_loc+nind_loc+i] = aux    # alpha
        # Jacobian elements related to the scale parameters betaT2 and varphi (equation A.10)
        if ntend_sc > 0:
            Jx[2+nmu+ntend_loc+nind_loc+npsi] = np.sum((Dpsit*Dpsitastpsit+Dmut*Dmutastpsit)*self.t*psit1)  # betaT2
        if nind_sc > 0:
            for i in range(nind_sc):
                Jx[2+nmu+ntend_loc+nind_loc+npsi+ntend_sc+i] = np.sum((Dpsit*Dpsitastpsit+Dmut*Dmutastpsit)*covariates_sc[:,i]*psit1) # varphi2

        # Jacobian elements related to the shape parameters gamma0 and gamma (equation A.10)
        if self.neps0 == 1:
            Jx[2+nmu+ntend_loc+nind_loc+npsi+ntend_sc+nind_sc] = np.sum(Depst+Dpsit*Dpsitastepst+Dmut*Dmutastepst)
        # If shape harmonics are included
        if neps > 0:
            for i in range(neps):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += (Depst[k]+Dpsit[k]*Dpsitastepst[k]+Dmut[k]*Dmutastepst[k])*self._Dparam(tt,i+1)
                Jx[2+self.neps0+nmu+ntend_loc+nind_loc+npsi+ntend_sc+nind_sc + i] = aux

        # Jacobian elements related to the shape parameters varphi3 (defined by Victor)
        if nind_sh > 0:
            for i in range(nind_sh):
                Jx[2 + self.neps0 + nmu + ntend_loc + nind_loc + npsi + ntend_sc + nind_sc + neps + i] = np.sum(
                    Depst * covariates_sh[:, i])


        ### Hessian matrix
        # Derivatives given by equations (A.13)-(A.17) in the paper
        D2mut = (1+epst)*zn*(-1+epst*z**(1/epst))/((z*psit)**2)
        D2psit = (-zn*xn*((1-epst)*xn-2)+((1-2*xn)-epst*(xn**2)))/((z*psit)**2)
        D2epst = -zn*(xn*(xn*(1+3*epst)+2+(-2-epst*(3+epst)*xn)*z**(1/epst)) + (z/(epst*epst))*np.log(z)*(2*epst*(-xn*(1+epst)-1+z**(1+1/epst))+z*np.log(z)))/(epst*epst*z**2)
        Dmutpsit = -(1+epst-(1-xn)*zn)/((z*psit)**2)
        Dmutepst = -zn*(epst*(-(1+epst)*xn-epst*(1-xn)*z**(1/epst))+z*np.log(z))/(epst*epst*psit*z**2)
        Dpsitepst = xn*Dmutepst

        # Corresponding Gumbel derivatives given by equations (A.18)-(A.20)
        D2mut[posG] = -(np.exp(-xn[posG])) / (psit[posG] ** 2)
        D2psit[posG] = ((1 - 2 * xn[posG]) + np.exp(-xn[posG]) * (2 - xn[posG]) * xn[posG]) / (psit[posG] ** 2)
        D2epst[posG] = 0  
        Dmutpsit[posG] = (-1 + np.exp(-xn[posG]) * (1 - xn[posG])) / (psit[posG] ** 2)
        Dmutepst[posG] = 0  
        Dpsitepst[posG] = 0

        # Initialize the Hessian matrix
        Hxx = np.zeros((2 + self.neps0 + nmu + npsi + neps + ntend_loc + nind_loc + ntend_sc + nind_sc + nind_sh,
                        2 + self.neps0 + nmu + npsi + neps + ntend_loc + nind_loc + ntend_sc + nind_sc + nind_sh))
        # Elements of the Hessian matrix
        # Sub-blocks following the order shown in Table 4 of the paper

        ## DIAGONAL SUB-BLOCKS
        # Sub-block number 1, beta0^2
        Hxx[0,0] = np.sum(D2mut)
        # Sub-block number 2, betaT^2
        if ntend_loc > 0:
            Hxx[1+nmu,1+nmu] = np.sum(D2mut*(self.t**2))
        # Sub-block number 3, varphi_i*varphi_j
        if nind_loc > 0:
            for i in range(nind_loc):
                for j in range(i+1):
                    Hxx[1+nmu+ntend_loc+i,1+nmu+ntend_loc+j] = np.sum(D2mut*covariates_loc[:,i]*covariates_loc[:,j])
        # Sub-block number 4, betaT2^2
        if ntend_sc > 0:
            Hxx[2+nmu+npsi+ntend_loc+nind_loc, 2+nmu+npsi+ntend_loc+nind_loc] = np.sum((D2psit*psit+Dpsit)*psit*(self.t**2))
        # Sub-block number 5, varphi2_i*varphi2_j
        if nind_sc > 0:
            for i in range(nind_sc):
                for j in range(i+1):
                    Hxx[2+nmu+npsi+ntend_loc+nind_loc+ntend_sc+i, 2+nmu+npsi+ntend_loc+nind_loc+ntend_sc+j] = np.sum((D2psit*psit+Dpsit)*psit*covariates_sc[:,i]*covariates_sc[:,j])
        # Sub-block number 6, alpha0^2
        Hxx[1+nmu+ntend_loc+nind_loc,1+nmu+ntend_loc+nind_loc] = np.sum((D2psit*psit+Dpsit)*psit)
        # Sub-block number 7, gamma0^2
        if self.neps0 == 1:
            # If the shape parameter is added but later the result is GUMBEL
            if len(posG) == len(self.xt):
                Hxx[2+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc, 2+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc] = -1
            else:
                Hxx[2+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc, 2+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc] = np.sum(D2epst)
        # Sub-block added by Victor, varphi3_i*varphi3_j
        if nind_sh > 0:
            for i in range(nind_sh):
                for j in range(i+1):
                    if len(posG) == len(self.xt) and i == j:
                        Hxx[2+self.neps0+nmu+npsi+neps+ntend_loc+nind_loc+ntend_sc+nind_sc+i, 2+self.neps0+nmu+npsi+neps+ntend_loc+nind_loc+ntend_sc+nind_sc+j] = -1
                    else:
                        Hxx[2+self.neps0+nmu+npsi+neps+ntend_loc+nind_loc+ntend_sc+nind_sc+i, 2+self.neps0+nmu+npsi+neps+ntend_loc+nind_loc+ntend_sc+nind_sc+j] = np.sum(D2epst*covariates_sh[:,i]*covariates_sh[:,j])

        # Sub-block number 8 (Scale exponential involved), beta0*alpha0
        Hxx[1+nmu+ntend_loc+nind_loc, 0] = np.sum(Dmutpsit*psit)

        if self.neps0 == 1:
            # Sub-block number 9, beta0*gamma0
            Hxx[2+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc,0] = np.sum(Dmutepst)
            # Sub-block number 10 (Scale exponential involved), alpha0*gamma0
            Hxx[2+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc, 1+nmu+ntend_loc+nind_loc] = np.sum(Dpsitepst*psit)
        # Sub-block number 11, beta0*betaT
        if ntend_loc > 0:
            Hxx[1+nmu, 0] = np.sum(D2mut*self.t)
        # Sub-block number 12 (Scale exponential involved), beta0*betaT2
        if ntend_sc > 0:
            Hxx[2+nmu+ntend_loc+nind_loc+npsi,0] = np.sum(Dmutpsit*self.t*psit)
        # Sub-block number 52 (Scale exponential involved), betaT2*alpha0
        if ntend_sc > 0:
            Hxx[2+nmu+ntend_loc+nind_loc+npsi,1+nmu+ntend_loc+nind_loc] = np.sum((D2psit*psit+Dpsit)*self.t*psit)
        # Sub-block number 48 (Scale exponential involved), betaT*betaT2
        if ntend_loc > 0 and ntend_sc > 0:
            Hxx[2+nmu+ntend_loc+nind_loc+npsi,1+nmu] = np.sum(Dmutpsit*self.t*self.t*psit)
        # Sub-block number 13, beta0*varphi_i
        if nind_loc > 0:
            for i in range(nind_loc):
                Hxx[1+nmu+ntend_loc+i,0] = np.sum(D2mut*covariates_loc[:,i])
        # Sub-block number 14 (Scale exponential involved), beta0*varphi2_i
        if nind_sc > 0:
            for i in range(nind_sc):
                Hxx[2+nmu+ntend_loc+nind_loc+npsi+ntend_sc+i,0] = np.sum(Dmutpsit*covariates_sc[:,i]*psit)
        # Sub-block number 53 (Scale exponential involved), alpha0*varphi2_i
        if nind_sc > 0:
            for i in range(nind_sc):
                Hxx[2+nmu+ntend_loc+nind_loc+npsi+ntend_sc+i,1+nmu+ntend_loc+nind_loc] = np.sum((D2psit*psit+Dpsit)*covariates_sc[:,i]*psit)
        # Sub-block number 49 (Scale exponential involved), betaT*varphi2_i
        if ntend_loc > 0 and nind_sc > 0:
            for i in range(nind_sc):
                Hxx[2+nmu+ntend_loc+nind_loc+npsi+ntend_sc+i,1+nmu] = np.sum(Dmutpsit * self.t * covariates_sc[:, i] * psit)
        # Sub-block number 15, betaT*varphi_i
        if nind_loc > 0 and ntend_loc > 0:
            for i in range(nind_loc):
                Hxx[1+nmu+ntend_loc+i, 1+nmu] = np.sum(D2mut * self.t * covariates_loc[:, i])
        # Sub-block number 16, betaT2*varphi2_i
        if nind_sc > 0 and ntend_sc > 0:
            for i in range(nind_sc):
                Hxx[2+nmu+ntend_loc+nind_loc+npsi+ntend_sc+i, 2+nmu+ntend_loc+nind_loc+npsi] = np.sum((D2psit * psit + Dpsit) * self.t * covariates_sc[:, i] * psit)
        # Sub-block number 17, alpha0*betaT
        if ntend_loc > 0:
            Hxx[1+nmu+ntend_loc+nind_loc, 1+nmu] = np.sum(Dmutpsit * self.t * psit)
        # Sub-block number 18, gamma0*betaT
        if ntend_loc > 0 and self.neps0 == 1:
            Hxx[2+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc, 1+nmu] = np.sum(Dmutepst * self.t)
        # Sub-block number 19 (Scale exponential involved), gamma0*betaT2
        if ntend_sc > 0 and self.neps0 == 1:
            Hxx[2+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc, 2+nmu+ntend_loc+nind_loc+npsi] = np.sum(Dpsitepst * self.t * psit)
        # Sub-block number 20 (Scale exponential involved), alpha0*varphi_i
        if nind_loc > 0:
            for i in range(nind_loc):
                Hxx[1+nmu+ntend_loc+nind_loc, 1+nmu+ntend_loc+i] = np.sum(Dmutpsit * covariates_loc[:, i] * psit)
        # Sub-block number 21, gamma0*varphi_i
        if nind_loc > 0 and self.neps0 == 1:
            for i in range(nind_loc):
                Hxx[2+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc, 1+nmu+ntend_loc+i] = np.sum(Dmutepst * covariates_loc[:, i])
        # Sub-block number 22 (Scale exponential involved), gamma0*varphi2_i
        if nind_sc > 0 and self.neps0 == 1:
            for i in range(nind_sc):
                Hxx[2+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc, 2+nmu+ntend_loc+nind_loc+npsi+ntend_sc+i] = np.sum(Dpsitepst * covariates_sc[:, i] * psit)
        # Sub-block added by Victor, gamma0*varphi3_i
        if nind_sh > 0 and self.neps0 == 1:
            for i in range(nind_sh):
                Hxx[2+self.neps0+nmu+npsi+neps+ntend_loc+nind_loc+ntend_sc+nind_sc+i,2+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc] = np.sum(D2epst*covariates_sh[:,i])

        
        if nmu > 0: 
            for i in range(nmu):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += D2mut[k]*self._Dparam(tt,i+1)
                # Sub-block number 23, beta_i*beta0
                Hxx[1+i,0] = aux
                for j in range(i+1):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += D2mut[k]*self._Dparam(tt,i+1)*self._Dparam(tt,j+1)
                    # Sub-block number 24, beta_i,beta_j
                    Hxx[1+i,1+j] = aux

            for i in range(nmu):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dmutpsit[k]*self._Dparam(tt,i+1)*psit[k]
                # Sub-block number 25 (Scale exponential involved), beta_i*alpha0
                Hxx[1+nmu+ntend_loc+nind_loc, 1+i] = aux
            
            if self.neps0 == 1:
                for i in range(nmu):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dmutepst[k]*self._Dparam(tt, i+1)
                    # Sub-block number 26 (Scale exponential involved), beta_i*gamma0
                    Hxx[2+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc,1+i] = aux
            if ntend_loc > 0:
                for i in range(nmu):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += D2mut[k]*tt*self._Dparam(tt, i+1)
                    # Sub-block number 27, betaT*beta_i
                    Hxx[1+nmu,1+i] = aux
                
            if ntend_sc > 0:
                for i in range(nmu):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dmutpsit[k]*tt*self._Dparam(tt, i+1)*psit[k]
                    # Sub-block number 46 (Scale exponential involved), betaT2*beta_i
                    Hxx[2+nmu+ntend_loc+nind_loc+npsi+ntend_sc,1+i] = aux 
            if nind_loc > 0:
                for i in range(nmu):
                    for j in range(nind_loc):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += D2mut[k]*covariates_loc[k,j]*self._Dparam(tt,i+1)
                        # Sub-block number 28, beta_i*varphi_j
                        Hxx[1+nmu+ntend_loc+j, 1+i] = aux
            if nind_sc > 0:
                for i in range(nmu):
                    for j in range(nind_sc):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += Dmutpsit[k]*covariates_sc[k,j]*self._Dparam(tt,i+1)*psit[k]
                        # Sub-block number 47 (Scale exponential involved), beta_i*varphi2_j
                        Hxx[2+nmu+ntend_loc+nind_loc+npsi+ntend_sc+j,1+i] = aux
            if nind_sh > 0:
                for i in range(nmu):
                    for j in range(nind_sh):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += Dmutepst[k]*covariates_sh[k,j]*self._Dparam(tt, i+1)
                        # Sub-block added by Victor, beta_j*varphi3_i
                        Hxx[2+self.neps0+nmu+npsi+neps+ntend_loc+nind_loc+ntend_sc+nind_sc+j,1+i] = aux
        if npsi > 0:
            for i in range(npsi):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += (D2psit[k]*psit[k]+Dpsit[k])*self._Dparam(tt,i+1)*psit[k]
                # Sub-block number 29 (Scale exponential involved), alpha_i*alpha_0
                Hxx[2+nmu+ntend_loc+nind_loc+i,1+ntend_loc+nind_loc+nmu] = aux
                for j in range(i+1):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += (D2psit[k]*psit[k]+Dpsit[k])*self._Dparam(tt, i+1)*self._Dparam(tt,j+1)*psit[k]
                    # Sub-block 30 (Scale exponential involved), alpha_i*alpha_j
                    Hxx[2+nmu+ntend_loc+nind_loc+i,2+nmu+ntend_loc+nind_loc+j] = aux
            if self.neps0 == 1:
                for i in range(npsi):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dpsitepst[k]*self._Dparam(tt,i+1)*psit[k]
                    # Sub-block number 31 (Scale exponential involved), alpha_i*gamma0
                    Hxx[2+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc, 2+nmu+ntend_loc+nind_loc+i] = aux
            for i in range(npsi):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dmutpsit[k]*self._Dparam(tt, i+1) * psit[k]
                # Sub-block number 32 (Scale exponential involved), beta0*alpha_i
                Hxx[2+nmu+ntend_loc+nind_loc+i,0] = aux
            if ntend_loc > 0:
                for i in range(npsi):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dmutpsit[k]*tt*self._Dparam(tt,i+1)*psit[k]
                    # Sub-block number 33 (Scale exponential involved), alpha_i*betaT
                    Hxx[2+nmu+ntend_loc+nind_loc+i,1+nmu] = aux
            if nind_loc > 0:
                for i in range(npsi):
                    for j in range(nind_loc):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += Dmutpsit[k]*covariates_loc[k,j]*self._Dparam(tt,i+1)*psit[k]
                        # Sub-block number 34 (Scale exponential involved), alpha_i*varphi_j
                        Hxx[2+nmu+ntend_loc+nind_loc+i,1+nmu+ntend_loc+j] = aux
            if nind_sh > 0:
                for i in range(npsi):
                    for j in range(nind_sh):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += Dpsitepst[k]*covariates_sh[k,j]*self._Dparam(tt,i+1)*psit[k]
                        # Sub-block added by Victor (scale exponential involved), alpha_i*varphi3_j
                        Hxx[2+self.neps0+nmu+npsi+neps+ntend_loc+nind_loc+ntend_sc+nind_sc+j,2+nmu+ntend_loc+nind_loc+i]=aux
        if neps > 0:
            for i in range(neps):
                # First element associated to the constant value (first column)
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += D2epst[k]*self._Dparam(tt,i+1)
                # Sub-block number 35, gamma_i*gamma0
                Hxx[2+self.neps0+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc+i,2+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc] = aux
                for j in range(i+1):
                    # If shape parameters included but later everything is GUMBEL
                    if j==i and len(posG) == len(self.xt):
                        Hxx[2+self.neps0+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc+i,2+self.neps0+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc+j] = -1
                    else:
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += D2epst[k]*self._Dparam(tt,i+1)*self._Dparam(tt,j+1)
                        # Sub-block number 36, gamma_i*gamma_j
                        Hxx[2+self.neps0+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc+i,2+self.neps0+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc+j] = aux
            for i in range(neps):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dpsitepst[k]*self._Dparam(tt,i+1)*psit[k]
                # Sub-block number 37 (Scale exponential involved) gamma_i*alpha0
                Hxx[2+self.neps0+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc+i,1+nmu+ntend_loc+nind_loc] = aux
            for i in range(neps):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dmutepst[k]*self._Dparam(tt,i+1)
                # Sub-block number 38, gamma_i*beta0
                Hxx[2+self.neps0+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc+i,0] = aux
            if ntend_loc > 0:
                for i in range(neps):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dmutepst[k]*tt*self._Dparam(tt,i+1)
                    # Sub-block number 39, gamma_i*betaT
                    Hxx[2+self.neps0+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc+i,1+nmu] = aux
            if ntend_sc > 0:
                for i in range(neps):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dpsitepst[k]*tt*self._Dparam(tt,i+1)*psit[k]
                    # Sub-block number 44 (Scale exponential involved), gamma_i*betaT2
                    Hxx[2+self.neps0+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc+i,2+nmu+npsi+ntend_loc+nind_loc] = aux
            if nind_loc > 0:
                for i in range(neps):
                    for j in range(nind_loc):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += Dmutepst[k]*covariates_loc[k,j]*self._Dparam(tt,i+1)
                        # Sub-block number 40, gamma_i*varphi_j
                        Hxx[2+self.neps0+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc+i,1+nmu+ntend_loc+j] = aux
            if nind_sc > 0:
                for i in range(neps):
                    for j in range(nind_sc):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += Dpsitepst[k]*covariates_sc[k,j]*self._Dparam(tt,i+1)*psit[k]
                        # Sub-block number 45 (Scale exponential involved), gamma_i*varphi2_j
                        Hxx[2+self.neps0+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc+i,2+nmu+npsi+ntend_loc+nind_loc+ntend_sc+j] = aux
            if nind_sh > 0:
                for i in range(neps):
                    for j in range(nind_sh):
                        aux = 0
                        for k, tt in enumerate(self.t):
                            aux += D2psit[k]*covariates_sh[k,j]*self._Dparam(tt,i+1)
                        # Sub-block added by Victor, gamma_i*varphi3_j
                        Hxx[2+self.neps0+nmu+npsi+neps+ntend_loc+nind_loc+ntend_sc+nind_sc+j,2+self.neps0+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc+i] = aux


        if nind_loc > 0 and ntend_sc > 0:
            for i in range(nind_loc):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dmutpsit[k]*tt*covariates_loc[k,i]*psit[k]
                # Sub-block number 50 (Scale exponential involved), varphi_i*betaT2
                Hxx[2+nmu+ntend_loc+nind_loc+npsi,1+nmu+ntend_loc+i] = aux
        if nind_loc > 0 and nind_sc > 0:
            for i in range(nind_loc):
                for j in range(nind_sc):
                    # Sub-block number 51 (Scale exponential involved), varphi_i*varphi2_j
                    Hxx[2+nmu+ntend_loc+nind_loc+npsi+ntend_sc+j,1+nmu+ntend_loc+i] = np.sum(Dmutpsit*covariates_sc[:,j]*covariates_loc[:,i]*psit)
        if nind_loc > 0 and nind_sh > 0:
            for i in range(nind_loc):
                for j in range(nind_sh):
                    # Sub-block added by Victor, varphi_i*varphi3_j
                    Hxx[2+self.neps0+nmu+npsi+neps+ntend_loc+nind_loc+ntend_sc+nind_sc+j, 1+nmu+ntend_loc+i] = np.sum(Dmutepst*covariates_loc[:,i]*covariates_sh[:,j])
        if nind_sc > 0 and nind_sh > 0:
            for i in range(nind_sc):
                for j in range(nind_sh):
                    # Sub-block added by Victor (scale exponential involved), varphi2_i*varphi3_j
                    Hxx[2+self.neps0+nmu+npsi+neps+ntend_loc+nind_loc+ntend_sc+nind_sc+j, 2+nmu+npsi+ntend_loc+nind_loc+ntend_sc+i] = np.sum(Dmutepst*covariates_sc[:,i]*covariates_sh[:,j]*psit)
        if nind_sh > 0 and ntend_loc >0:
            for i in range(nind_sh):
                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dmutepst[k]*tt*covariates_sh[k,i]
                # Sub-block added by Victor, betaT*varphi3_i
                Hxx[2+self.neps0+nmu+npsi+neps+ntend_loc+nind_loc+ntend_sc+nind_sc+i, 1+nmu] = aux
        if ntend_sc > 0:
            for i in range(npsi):
                aux = 0 
                for k, tt in enumerate(self.t):
                    aux += (D2psit[k]*psit[k]+Dpsit[k])*tt*self._Dparam(tt,i+1)*psit[k]
                # Sub-block number 54 (Scale exponential involved), alpha_i*betaT2
                Hxx[2+nmu+ntend_loc+nind_loc+npsi+ntend_sc,2+nmu+ntend_loc+nind_loc+i] = aux
        if nind_sc > 0:
            for i in range(npsi):
                for j in range(nind_sc):
                    aux = 0 
                    for k, tt in enumerate(self.t):
                        aux += (D2psit[k]*psit[k]+Dpsit[k])*covariates_sc[k,j]*self._Dparam(tt,i+1)*psit[k]
                    # Sub-block number 55 (Scale exponential involved), alpha_i*varphi2_j
                    Hxx[2+nmu+ntend_loc+nind_loc+npsi+ntend_sc+j,2+nmu+ntend_loc+nind_loc+i] = aux
        if nmu > 0 and npsi > 0:
            for j in range(nmu):
                for i in range(npsi):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dmutpsit[k]*self._Dparam(tt,i+1)*self._Dparam(tt,j+1)*psit[k]
                    # Sub-block number 41 (Scale exponential involved), beta_j*alpha_i
                    Hxx[2+nmu+ntend_loc+nind_loc+i,1+j] = aux
        if nmu > 0 and neps > 0:
            for j in range(nmu):
                for i in range(neps):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dmutepst[k]*self._Dparam(tt,i+1)*self._Dparam(tt,j+1)
                    # Sub-block number 42, beta_j*gamma_i
                    Hxx[2+self.neps0+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc+i,1+j] = aux
        if npsi > 0 and neps > 0:
            for j in range(npsi):
                for i in range(neps):
                    aux = 0
                    for k, tt in enumerate(self.t):
                        aux += Dpsitepst[k]*self._Dparam(tt,i+1)*self._Dparam(tt,j+1)*psit[k]
                    # Sub-block number 43 (Scale exponential involved), alpha_j*gamma_i
                    Hxx[2+self.neps0+nmu+npsi+ntend_loc+nind_loc+ntend_sc+nind_sc+i,2+nmu+ntend_loc+nind_loc+j] = aux

        if nind_sh > 0:
            for i in range(nind_sh):
                # Sub-block added by Victor, beta0*varphi3_i
                Hxx[2+self.neps0+nmu+npsi+neps+ntend_loc+nind_loc+ntend_sc+nind_sc+i, 0] = np.sum(Dmutepst*covariates_sh[:,i])
                # Sub-block added by Victor (scale exponential involved), alpha0*varphi3_i
                Hxx[2+self.neps0+nmu+npsi+neps+ntend_loc+nind_loc+ntend_sc+nind_sc+i, 1+nmu+ntend_loc+nind_loc] = np.sum(Dpsitepst*psit*covariates_sh[:,i])

                aux = 0
                for k, tt in enumerate(self.t):
                    aux += Dpsitepst[k]*tt*covariates_sh[k,i]*psit[k]
                # Sub-bloc added by Victor (scale exponential involved), betaT2*varphi3_i
                Hxx[2+self.neps0+nmu+npsi+neps+ntend_loc+nind_loc+ntend_sc+nind_sc+i, 1+nmu+npsi+ntend_loc+nind_loc]  = aux

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

