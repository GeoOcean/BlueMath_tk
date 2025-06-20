import matplotlib.pyplot as plt
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
        beta_cov = []
        list_loc = []   # List of covariates for location
        nind_loc = 0    # TODO: IT IS ZERO BY DEFAULT, CHECK IF IT POSSIBLE TO REMOVE THIS LINE
        auxcov_loc = self.covariates.iloc[:, list_loc].values
        # Auxiliar variables related to scale parameter
        alpha_cov = []
        list_sc = []
        nind_sc = 0     # TODO: IT IS ZERO BY DEFAULT, CHECK IF IT POSSIBLE TO REMOVE THIS LINE
        auxcov_sc = self.covariates.iloc[:, list_sc].values
        # Auxiliar variables related to shape parameter
        gamma_cov = []
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
                if any(beta_cov != 0):
                    beta_cov_init = beta_cov.copy()
                else: 
                    beta_cov_init = np.asarray([])
                if any(alpha_cov != 0):
                    alpha_cov_init = alpha_cov.copy()
                else: 
                    alpha_cov_init = np.asarray([])
                if any(gamma_cov != 0):
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


    def _loglikelihood(self) -> float: 
        # TODO
        ...



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

