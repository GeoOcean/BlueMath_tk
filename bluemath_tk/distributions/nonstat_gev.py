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
        self.xi0 = np.empty(0)  # Shape intercept
        self.xi = np.empty(0)  # Shape harmonic
        self.xiT = np.empty(0)  # Shape trend
        self.xi_cov = np.empty(0)  # Shape covariates

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
        self.nxi0 = 1  # 1 if shape parameter is included, defaul Weibull or Frechet
        self.nxi = 0  # Number of parameters of harmonic part of shape
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
            - xi0    -> Optimal constant parameter related to shape
            - xi     -> Optimal harmonic vector associated with shape
            - betaT     -> Optimal location trend parameter
            - auxvarphi -> Optimal location covariate vector
            - betaT     -> Optimal scale trend parameter
            - auxvarphi -> Optimal scale covariate vector
            - grad      -> Gradient of the log-likelihood function with the sign changed at the optimal solution
            - hessian   -> Hessian of the log-likelihood function with the sign changed at the optimal solution
            - popt      -> vector including the optimal values for the parameters in the following order:
                        beta0, beta, betaT, varphi, alpha0, alpha, betaT2, varphi2, xi0, xi
            - stdpara   -> vector including the standard deviation of the optimal values for the parameters in the following order:
                        beta0, beta, betaT, varphi, alpha0, alpha, betaT2, varphi2, xi0, xi
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
        nxi = 0  # Number of parameters of harmonic part of shape
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
            fit_result = self._fit(nmu, npsi, nxi)

            # Check if the model is Gumbel
            self.nxi0 = 1
            if fit_result["xi0"] is None:
                self.nxi0 = 0
            elif np.abs(fit_result["xi0"]) <= 1e-8:
                self.nxi0 = 0

            # Compute AIC and Loglikelihood
            self.loglike_iter[iter] = -fit_result["loglikelihood"]
            n_params = (
                2
                + self.nxi0
                + 2 * nmu
                + 2 * npsi
                + 2 * nxi
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
            if fit_result['xi'] is not None:
                fit_result_aux['xi'] = np.concatenate((fit_result['xi'] , [0, 0]))
            else:
                fit_result_aux['xi'] = np.array([0, 0])
            
            auxf, auxJx, auxHxx = self._loglikelihood(**fit_result_aux)

            # Inverse of the Information Matrix (auxHxx) 
            auxI0 = np.linalg.inv(-auxHxx)

            # Updating the best model
            if iter > 0:
                # TODO: Implement another criterias (Proflike)
                if self.AIC_iter[iter] < self.AIC_iter[iter-1]:
                    modelant = np.array([nmu, npsi, nxi])
            else:
                modelant = np.array([nmu, npsi, nxi])
            
            ### Step 5: Compute maximum perturbation
            pos = 1
            max_val = np.abs(auxJx[2*nmu:2*nmu+2].T @ auxI0[2*nmu:2*nmu+2, 2*nmu:2*nmu+2] @ auxJx[2*nmu:2*nmu+2])
            # TODO: AÑADIR BIEN LAS COVARIABLES EN EL PARAMETRO DE FORMA
            auxmax = abs(auxJx[2 + 2*nmu + ntrend_loc + nind_loc + 2*npsi + 2:2 + 2*nmu + ntrend_loc + nind_loc + 2*npsi + 2+2].T @ auxI0[2 + 2*nmu + ntrend_loc + nind_loc + 2*npsi + 2:2 + 2*nmu + ntrend_loc + nind_loc + 2*npsi + 2+2, 2 + 2*nmu + ntrend_loc + nind_loc + 2*npsi + 2:2 + 2*nmu + ntrend_loc + nind_loc + 2*npsi + 2+2] @ auxJx[2 + 2*nmu + ntrend_loc + nind_loc + 2*npsi + 2:2 + 2*nmu + ntrend_loc + nind_loc + 2*npsi + 2+2])
            if auxmax > max_val:
                max_val = auxmax
                pos = 2

            # TODO: AÑADIR BIEN LAS COVARIABLES EN EL PARAMETRO DE FORMA
            auxmax = abs(auxJx[2 + self.nxi0 + 2*nmu + ntrend_loc + nind_loc + ntrend_sc + nind_sc + 2*npsi + 2*nxi + 4 : 2 + self.nxi0 + 2*nmu + ntrend_loc + nind_loc + ntrend_sc + nind_sc + 2*npsi + 2*nxi + 4+2].T @ auxI0[2 + self.nxi0 + 2*nmu + ntrend_loc + nind_loc + ntrend_sc + nind_sc + 2*npsi + 2*nxi + 4 : 2 + self.nxi0 + 2*nmu + ntrend_loc + nind_loc + ntrend_sc + nind_sc + 2*npsi + 2*nxi + 4+2, 2 + self.nxi0 + 2*nmu + ntrend_loc + nind_loc + ntrend_sc + nind_sc + 2*npsi + 2*nxi + 4 : 2 + self.nxi0 + 2*nmu + ntrend_loc + nind_loc + ntrend_sc + nind_sc + 2*npsi + 2*nxi + 4+2] @ auxJx[2 + self.nxi0 + 2*nmu + ntrend_loc + nind_loc + ntrend_sc + nind_sc + 2*npsi + 2*nxi + 4 : 2 + self.nxi0 + 2*nmu + ntrend_loc + nind_loc + ntrend_sc + nind_sc + 2*npsi + 2*nxi + 4+2])
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
                nxi += 1

            if iter > 0:
                if self.AIC_iter[iter] >= self.AIC_iter[iter-1]:
                    model = modelant
                    self.AICini = self.AIC_iter[iter-1]
                    loglikeobjITini = self.loglike_iter[iter-1]
                    break
                else:
                    model = np.array([nmu, npsi, nxi])
            

        self.niter_harm = iter
        self.nit = iter
        print("End of the Harmonic iterative process")

        ######### End of the Harmonic iterative process       
        # Obtaining the MLE for the best model
        nmu = model[0]
        npsi = model[1]
        nxi = model[2]

        # CHECKING THE SHAPE PARAMETER
        self.nxi0 = 0 # Force the elimination of the constant shape parameter (xi0)
        fit_result = self._fit(nmu,npsi,nxi)

        n_params = (
                2
                + self.nxi0
                + 2 * nmu
                + 2 * npsi
                + 2 * nxi
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
            # The constant shape parameter (xi0) is significative
            self.nxi0 = 1
            fit_result = self._fit(nmu,npsi,nxi)
        
        print("Harmonic AIC:", self.AICini, "\n")

        self._update_params(fit_result)
        self.nmu = nmu
        self.npsi = npsi
        self.nxi = nxi

        ######### COVARIATES Iterative process #########
        # TODO



    def _fit(
        self,
        nmu,
        mpsi,
        nxi,
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
        self.nxi0 = kwargs.get("nxi0")

        self.beta0 = kwargs.get("beta0")
        self.beta = kwargs.get("beta")
        self.betaT = kwargs.get("betaT")
        self.beta_cov = kwargs.get("beta_cov")

        self.alpha0 = kwargs.get("alpha0")
        self.alpha = kwargs.get("alpha")
        self.alphaT = kwargs.get("alphaT")
        self.alpha_cov = kwargs.get("alpha_cov")

        self.xi0 = kwargs.get("xi0")
        self.xi = kwargs.get("xi")
        self.xiT = kwargs.get("xiT")
        self.xi_cov = kwargs.get("xi_cov")

        self.popt = kwargs.get("popt")

