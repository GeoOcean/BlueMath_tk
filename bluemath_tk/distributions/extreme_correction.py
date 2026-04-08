import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import xarray as xr
from statsmodels.distributions.empirical_distribution import ECDF

from ..core.io import BlueMathModel
from ..distributions.gev import GEV
from ..distributions.gpd import GPD
from ..distributions.pareto_poisson import GPDPoiss
from ..distributions.pot import OptimalThreshold
from ..distributions.utils.extr_corr_utils import (
    gev_ci_rp_bootstrap,
    gpdpoiss_ci_rp_bootstrap,
)


class ExtremeCorrection(BlueMathModel):
    """
    Extreme Correction class
    """

    def __init__(
        self,
        corr_config: dict,
        pot_config: dict = {},
        method: str = "pot",
        conf_level: float = 0.95,
        debug: bool = False,
    ):
        """
        upper-tail correction technique for synthetic datasets using
        Generalized Extreme Value (GEV) or Peaks Over Threshold (POT) approaches.

        This class applies upper-tail corrections to sampled datasets
        by fitting extreme value distributions to historical observations
        and adjusting the sampled extremes accordingly. See Collado et al. (2026) [1].

        Parameters
        ----------
        config : dict
            Dictionary containing the main configuration of the model.
            Required keys:
                - var : str
                    Variable to apply the correction.
                - time_var : str
                    Name of the time variable (datetime or timestamp).
                - yyyy_var : str
                    Name of the year variable.
                - freq : float or int
                    Frequency of observations per year
                    (e.g., 365.25 for daily data).
            Optional keys:
                - mm_var : str, default "mm"
                    Name of the month variable.
                - dd_var : str, default "dd"
                    Name of the day variable.
                - folder : str, default None
                    Path to a folder where diagnostic plots will be saved.
        pot_config : dict
            Dictionary containing the POT configuration.
            Keys:
                - n0 : int, default 10
                    Minimum number of exceedances required.
                - min_peak_distance : int, default 2
                    Minimum distance (in data points) between two peaks.
                - init_threshold : float, default 0.0
                    Initial threshold for peak extraction.
                - siglevel : float, default 0.05
                    Significance level for the Chi-squared test in
                    threshold optimization.
                - plot_flag : bool, default True
                    Whether to generate threshold selection plots.
        method : {"am", "pot"}, default "pot"
            Method for correction.
            - "am" : Annual Maxima using GEV distribution.
            - "pot" : Peaks Over Threshold using GPD distribution.
        conf_level : float, default=0.95
            Confidence level for return period confidence intervals.

        References
        ----------
        [1] Collado, V., Méndez, F.J. & Mínguez, R. Upper-tail correction of
            multivariate synthetic environmental series using annual maxima.
            Stoch Environ Res Risk Assess 40, 92 (2026).
            https://doi.org/10.1007/s00477-026-03215-0
        """
        super().__init__()

        self.set_logger_name(
            name=self.__class__.__name__, level="DEBUG" if debug else "INFO"
        )
        self.logger.info("Initializing Extreme Correction Procedure")

        # TODO: CAMBIAR EL CONFIG
        # Validate config input
        self.config = corr_config
        self._validate_config()

        # Method
        self.method = method.lower()
        if self.method == "pot":
            self.pot_config = pot_config
            self._validate_pot_config()

        # Initialize fitted parameters
        # If GEV (loc, scale, shape)
        # If GPD (threshold, scale, shape)
        self.parameters = np.empty(3)
        self.threshold = 1e10  # Fix threshold in case "am" method is used

        # Confidence level
        self.conf = conf_level

    def _validate_config(self) -> None:
        """
        Validate the configuration dictionary for extreme correction

        Raise
        -----
        KeyError
            If any required key is missing
        TypeError
            If type of any required key is wrong
        """
        # Required fields
        required_fields = {
            "var": str,
            # "time_var": str,
            # "yyyy_var": str,
            # "freq": float | int,
        }

        for key, exp_type in required_fields.items():
            if key not in self.config:
                if key not in self.config:
                    raise KeyError(
                        f"Configuration error: Key '{key}' is missing in the config dictionary."
                    )
            if not isinstance(self.config[key], exp_type):
                raise TypeError(
                    f"Configuration error: Key '{key}' must be of type {exp_type.__name__}."
                )

        # Optional fields with defaults
        optional_fields = {
            "mm_var": "mm",
            "dd_var": "dd",
            "bmus_var": None,
            "folder": None,
        }

        for key, default_value in optional_fields.items():
            self.config[key] = self.config.get(key, default_value)

        # Define the configuration in the class
        self.var = self.config.get("var")
        self.time_var = self.config.get("time_var")
        self.year_var = self.config.get("yyyy_var")
        self.month_var = self.config.get("mm_var")
        self.day_var = self.config.get("dd_var")
        self.freq = self.config.get("freq")
        # Weather Type variable in case we apply the correction by WT
        self.bmus_var = self.config["bmus_var"]

        if self.config["folder"] is not None:
            self.folder = self.config["folder"]
            os.makedirs(self.folder, exist_ok=True)

        # TODO: Corrección por WT
        # if self.config.get(self.bmus_var) is not None:
        #     self.n_wt = np.unique(self.data_hist[self.bmus_var])
        # else:
        #     self.n_wt = 1

    def _validate_pot_config(self) -> None:
        """
        Validate POT configuration dictionary for peaks extraction.
        """

        self.pot_config["n0"] = self.pot_config.get("n0", 10)
        self.pot_config["min_peak_distance"] = self.pot_config.get(
            "min_peak_distance", 2
        )
        self.pot_config["init_threshold"] = self.pot_config.get("init_threshold", 0.0)
        self.pot_config["sig_level"] = self.pot_config.get("siglevel", 0.05)
        self.pot_config["plot"] = self.pot_config.get("plot", False)

    def fit(
        self,
        data_hist: xr.Dataset,
        plot_diagnostic: bool = False,
    ) -> None:
        """
        Fit the historical data into GEV or GPD

        Parameters
        ----------
        data_hist : xr.Dataset
            Dataset with historical data
        bmus : list[bool, str], default=[False, ""]
            Whether to apply the correction by BMUS, if given the name of bmus variable should be given
        plot_diagnostic : bool, default=False
            Whether to plot the diagnostics plot of the fitted distribution
        """
        self.pit_data, self.am_data = self._preprocess_data(
            data_hist,
            var=self.config.get("var"),
            bmus=self.config.get("bmus", [False, ""]),
            sim=False,
            join_sims=self.config.get("join_sims", True),
        )
        self.n_year = self.am_data.size

        # If POT used in fitting step
        if self.method == "pot":
            opt_threshold = OptimalThreshold(
                data=self.pit_data,
                threshold=self.pot_config.get("init_threshold", 0.0),
                n0=self.pot_config.get("n0", 10),
                min_peak_distance=self.pot_config.get("min_peak_distance", 2),
                sig_level=self.pot_config.get("sig_level", 0.05),
                method=self.pot_config.get("method", "studentized"),
                plot=self.pot_config.get("plot", False),
                folder=self.pot_config.get("folder", False),
                display_flag=self.pot_config.get("display_flag", False),
            )
            self.threshold, self.pot_data, pot_idx = opt_threshold.fit()
            self.poiss_parameter = self.pot_data.size / self.am_data.size

            fit_result = GPD.fit(self.pot_data, threshold=self.threshold)

        # If Annual Maxima used in fitting step
        if self.method == "am":
            fit_result = GEV.fit(self.am_data)
            self.poiss_parameter = 1  # If GEV only 1 exceedance per year (AM)

        # [loc, scale, shape] if GEV or [threshold, scale, shape] if GPD
        self.parameters = fit_result.params

        # TODO: Ver que diagnostic devolver alomejor no hace falta todo
        if plot_diagnostic:
            fit_result.plot()

    def transform(
        self,
        data_sim: xr.Dataset,
        quantile: float = 0.9,
        prob: str = "unif",
        random_state: int = 0,
        siglevel: float = 0.05,
    ) -> xr.Dataset:
        """
        Apply the correction in the synthetic dataset

        Parameters
        ----------
        data_sim : xr.Dataset
            Dataset with synthetic data
        quantile : float, default=0.9
            Quantile to apply the correction. Only values above this quantile will be
            corrected.
            By default, the correction is applied to the upper 10% of the data
        prob : str, default="unif"
            Type of probabilities consider to random correct the AM
            If "unif", a sorted random uniform is considered
            If "ecdf", the ECDF is considered
        random_state : int, default=0
            Random state to generate the probabilities
        siglevel : float, default=0.05

        Returns
        -------
        sim_pit_data_corrected : xr.Dataset
            Point-in-time corrected data
        """
        np.random.seed(random_state)

        self.sim_pit_data, self.sim_am_data = self._preprocess_data(
            data_sim,
            var=self.config.get("var"),
            bmus=self.config.get("bmus", [False, ""]),
            sim=True,
            join_sims=self.config.get("join_sims", True),
        )
        self.sim_am_data_sorted = np.sort(self.sim_am_data)
        self.n_year_sim = self.sim_am_data.shape[0]

        # Avoid correct when AM is 0 or lower than the threshold or the quantile given
        self.am_idx_0 = np.argmin(
            (self.sim_am_data_sorted == 0)
            | (
                ECDF(self.sim_pit_data)(self.sim_am_data_sorted)
                < ECDF(self.pit_data)(self.threshold)
            )
        )

        # Test if the correction has to be applied
        test_result = self.test()
        self.p_value = test_result.get("P-value")
        if self.p_value > siglevel:
            self.logger.info(
                f"Synthetic data comes from fitted distribution (P-value: {self.p_value:.6f})"
            )
            self.sim_am_data_corr = self.sim_am_data
            self.sim_pit_data_corrected = self.sim_pit_data
            return

        else:
            self.logger.info(
                f"Synthetic data does not come from fitted distribution (P-value: {self.p_value:.6f})"
            )
            # Initialize sim_am_data_corr
            self.sim_am_data_corr = np.zeros(self.n_year_sim - self.am_idx_0)

            # Define probs using uniform or ECDF
            if prob == "unif":
                self.rprob_sim = np.sort(
                    np.random.uniform(
                        low=0,
                        high=1,
                        size=self.n_year_sim - self.am_idx_0,
                    )
                )
            else:
                self.rprob_sim = np.arange(1, self.n_year_sim + 1 - self.am_idx_0) / (
                    self.n_year_sim + 1 - self.am_idx_0
                )  # ECDF

            # Apply correction on AM
            if self.method == "pot":
                # TODO: Añadir funciones de POT
                self.sim_am_data_corr = GPDPoiss.qf(
                    self.rprob_sim,
                    threshold=self.parameters[0],
                    scale=self.parameters[1],
                    shape=self.parameters[2],
                    poisson=self.poiss_parameter,
                )
            elif self.method == "am":
                self.sim_am_data_corr = GEV.qf(
                    self.rprob_sim,
                    loc=self.parameters[0],
                    scale=self.parameters[1],
                    shape=self.parameters[2],
                )

            # Apply correction in pit data
            if self.n_year_sim > 1:
                q_threshold = np.quantile(self.sim_pit_data, quantile)

                self.sim_pit_data_corrected = np.interp(
                    self.sim_pit_data,  # x-coords to interpolate
                    np.append(
                        [min(self.sim_pit_data), q_threshold],
                        self.sim_am_data_sorted[self.am_idx_0 :],
                    ),  # x-coords of data points
                    np.append(
                        [min(self.sim_pit_data), q_threshold], self.sim_am_data_corr
                    ),  # y-coords of data points
                )

            self.sim_am_data_corr_aux = self.sim_am_data.copy()
            self.sim_am_data_corr_aux[self.am_idx_0 :] = self.sim_am_data_corr
            self.sim_am_data_corr = self.sim_am_data_corr_aux

        output = self._preprocess_output(data=data_sim)

        return output

    def fit_transform(
        self,
        data_hist: xr.Dataset,
        data_sim: xr.Dataset,
        quantile: float = 0.9,
        bmus: list[bool, str] = [False, ""],
        prob: str = "unif",
        plot_diagnostic: bool = False,
        random_state: int = 0,
    ) -> xr.Dataset:
        """
        Fit and apply the correction procedure

        See fit and transform for more information

        Parameters
        ----------
        data_hist : xr.Dataset
            Dataset with historical data
        data_sim : xr.Dataset
            Dataset with synthetic data
        bmus : list[bool, str], default=[False, ""]
            Whether to apply the correction by BMUS, if given the name of bmus variable
            should be given
        prob : str, default="unif"
            Type of probabilities consider to random correct the AM
            If "unif", a sorted random uniform is considered
            If "ecdf", the ECDF is considered
        plot_diagnostic : bool, default=False
            Whether to plot the diagnostics plot of the fitted distribution
        random_state : int, default=0
            Random state to generate the probabilities

        Returns
        -------
        sim_pit_data_corrected : xr.Dataset
            Point-in-time corrected data
        """
        self.fit(data_hist=data_hist, plot_diagnostic=plot_diagnostic)

        return self.transform(
            data_sim=data_sim, quantile=quantile, prob=prob, random_state=random_state
        )

    def _preprocess_data(
        self,
        data: xr.Dataset,
        var: list[str],
        bmus: list[bool, str] = [False, ""],
        sim: bool = True,
        join_sims: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data

        Parameters
        ----------
        data : xr.Dataset
            Data to apply correction
        var : list[str]
            List of variables to apply the correction technique.
            TODO: FUTURE WORK: INCLUDE MORE THAN ONE
        bmus : list[bool, str], default=[False, ""]
            List to decide if the correction must be applied by WT and if so name of the
            variable
        join_sims : bool, default=True
            Whether to joint all the simulations in one array

        Return
        ------
        pit_data : np.ndarray
            Point-in-time data
        am_data : np.ndarray
            Annual Maxima values

        """
        # dict_allowed_freq = {"D": 1, "h": 24, "m": 1440}

        # self.freqstr = data.indexes["time"].freqstr
        # self.freq = dict_allowed_freq.get(self.freqstr)

        if join_sims and sim:
            n_sims = data.get("n_sim").values
            pit_data = data.get(f"{var}").values.flatten()
            am_data = data.get(f"{var}").groupby("time.year").max().values.flatten()
        else:
            pit_data = data.get(f"{var}").values.flatten()
            am_data = data.get(f"{var}").groupby("time.year").max().values.flatten()

        return pit_data, am_data

    def _preprocess_output(self, data: xr.Dataset) -> xr.Dataset:
        """
        Preprocess the output dataset

        Parameters
        ----------
        data : xr.Dataset
            Data to add the corrected variable

        Returns
        -------
        data : xr.Dataset
            Data with added the corrected variable
        """
        n_sim = data.get("n_sim").values.shape[0]
        n_time = data.get("time").values.shape[0]
        sim_pit_data_corrected_reshaped = self.sim_pit_data_corrected.reshape(
            n_sim, n_time
        )

        data[f"{self.var}_corr"] = (
            data[f"{self.var}"].dims,
            sim_pit_data_corrected_reshaped,
        )

        return data

    def test(self) -> dict:
        """
        Bootstrap Cramer Von-Mises test to check the GOF of fitted distribution

        Test to check the Goodness-of-Fit of the historical fitted distribution with the
        synthetic data.
        Null Hypothesis: sampled AM comes from the fitted extreme distribution.

        Returns
        -------
        dict
            Statistic and p-value of the Cramer Von-Mises test

        Notes
        -----
        The test is applied in the AM since the correction procedure is applied in the AM
        """
        Bootstrap_test = 500
        # Historical parameters estimated
        u, sigma, xi = self.parameters
        mu = u + sigma/xi * (self.poiss_parameter ** xi - 1)
        psi = sigma * self.poiss_parameter ** xi

        CvM_statistic = (
            1/(12*self.n_year_sim) +
            np.sum(
                (np.sort(
                    GEV.cdf(self.sim_am_data, loc=mu, scale=psi, shape=xi)
                ) -
                (2 * np.arange(1, self.n_year_sim +1) - 1) /
                (2 * self.n_year_sim))**2
            )
        )

        # Bootstrap calibration
        boot_statistic = np.zeros(Bootstrap_test)

        for b in range(Bootstrap_test):
            # 1. Simulate AM of size n_hist from GEV with historical parameters
            uniform_random_hist = np.random.uniform(size=self.n_year)
            simulated_am_boot_hist = GEV.qf(
                uniform_random_hist,
                loc=mu,
                scale=psi,
                shape=xi
            )

            # 2. Fit GEV to simulated AM
            [mu_bootfit, psi_bootfit, xi_bootfit] = GEV.fit(
                simulated_am_boot_hist
            ).params

            # 3. Simulate AM of size n_sim from GEV with historical parameters
            uniform_random_sim = np.random.uniform(size=self.n_year_sim)
            simulated_am_boot_sim = GEV.qf(
                uniform_random_sim,
                loc=mu_bootfit,
                scale=psi_bootfit,
                shape=xi_bootfit
            )

            # 4. Compute bootstrap synthetic AM values using the bootfit
            u_bootfit = GEV.cdf(
                simulated_am_boot_sim,
                loc=mu_bootfit,
                scale=psi_bootfit,
                shape=xi_bootfit,
            )

            # 5. Compute CvM test for the bootstrap sample
            boot_statistic[b] = (
                1/(12*self.n_year_sim) +
                np.sum(
                    (np.sort(
                        u_bootfit
                    ) -
                    (2*np.arange(1, self.n_year_sim + 1) - 1) /
                    (2*self.n_year_sim))**2
                )
            )

        bootstrap_p_value = (
            (1 + np.sum(boot_statistic >= CvM_statistic)) /
            (1 + Bootstrap_test)
        )

        return {"Statistic": CvM_statistic, "P-value": bootstrap_p_value}

        # if self.method == "pot":
        #     gev_location = (
        #         self.parameters[0]
        #         + (
        #             self.parameters[1]
        #             * (1 - self.poiss_parameter ** self.parameters[2])
        #         )
        #         / self.parameters[2]
        #     )
        #     gev_scale = self.parameters[1] * self.poiss_parameter ** self.parameters[2]

        #     # POT test
        #     # res_test = stats.cramervonmises(self.sim_pot_data,
        #     #                                 cdf=stats.genpareto.cdf,
        #     #                                 args=(self.parameters[2], self.parameters[0], self.parameters[1])
        #     #                                 )

        #     # AM test to derived GEV from GPD-Poisson
        #     res_test = stats.cramervonmises(
        #         self.sim_am_data,
        #         cdf=stats.genextreme.cdf,
        #         args=(self.parameters[2], gev_location, gev_scale),
        #     )
        #     return {"Statistic": res_test.statistic, "P-value": res_test.pvalue}

        # elif self.method == "am":
        #     res_test = stats.cramervonmises(
        #         self.sim_am_data,
        #         cdf=stats.genextreme.cdf,
        #         args=(self.parameters[2], self.parameters[0], self.parameters[1]),
        #     )
        #     return {"Statistic": res_test.statistic, "P-value": res_test.pvalue}

    def plot(self) -> tuple[list[plt.Figure], list[plt.Axes]]:
        """
        Plot return periods
        """
        figs = []
        axes = []

        fig1, ax1 = self.hist_retper_plot()
        figs.append(fig1)
        axes.append(ax1)

        fig2, ax2 = self.sim_retper_plot()

        figs.append(fig2)
        axes.append(ax2)

        fig3, ax3 = self.ecdf_plot()
        figs.append(fig3)

        return figs, axes

    def hist_retper_plot(self) -> tuple[plt.Figure, plt.Axes]:
        """
        Historical Return Period plot

        Returns
        -------
        fig
            plt.Figure
        ax
            plt.Axes
        """

        ecdf_annmax_probs_hist = np.arange(1, self.n_year + 1) / (self.n_year + 1)
        self.T_annmax = 1 / (1 - ecdf_annmax_probs_hist)

        # Fitted Return Periods
        self.T_years = np.array(
            [
                1.001,
                1.01,
                1.1,
                1.2,
                1.4,
                1.6,
                2,
                2.5,
                3,
                3.5,
                4,
                4.5,
                5,
                7.5,
                10,
                12.5,
                15,
                17.5,
                20,
                25,
                30,
                35,
                40,
                45,
                50,
                60,
                70,
                80,
                90,
                100,
                150,
                200,
                500,
                1000,
                5000,
                10000,
            ]
        )
        if self.method == "pot":
            self.ret_levels = GPDPoiss.qf(
                1 - 1 / self.T_years,
                self.parameters[0],
                self.parameters[1],
                self.parameters[2],
                self.poiss_parameter,
            )
            self.lower_ci_rp, self.upper_ci_rp = gpdpoiss_ci_rp_bootstrap(
                pot_data=self.pot_data,
                years=self.T_years,
                threshold=self.threshold,
                poisson=self.poiss_parameter,
                B=1000,
                conf_level=0.95,
            )

            self.dist = "GPD-Poisson"
        else:
            self.ret_levels = GEV.qf(
                1 - 1 / self.T_years,
                self.parameters[0],
                self.parameters[1],
                self.parameters[2],
            )
            self.lower_ci_rp, self.upper_ci_rp = gev_ci_rp_bootstrap(
                am_data=self.am_data, years=self.T_years, B=1000, conf_level=0.95
            )

            self.dist = "GEV"

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        # Fitted distribution
        ax.semilogx(
            self.T_years,
            self.ret_levels,
            color="red",
            linestyle="dashed",
            linewidth=2.5,
            label=f"Fitted {self.dist}",
        )
        # Confidence interval for fitted Distribution
        ax.semilogx(
            self.T_years,
            self.upper_ci_rp,
            color="tab:gray",
            linestyle="dotted",
            label=f"{self.conf} Conf. Band",
        )
        ax.semilogx(
            self.T_years, self.lower_ci_rp, color="tab:gray", linestyle="dotted"
        )

        # Historical AM values
        ax.semilogx(
            self.T_annmax,
            np.sort(self.am_data),
            color="tab:blue",
            linewidth=0,
            marker="o",
            markersize=5,
            label="Historical Annual Maxima",
        )

        ax.set_xlabel("Return Periods (Years)")
        ax.set_ylabel(f"{self.var}")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 250, 1000, 10000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlim(left=0.9, right=self.n_year + 100)
        ax.set_ylim(bottom=0)
        ax.legend(loc="best")
        ax.grid()

        return fig, ax

    def sim_retper_plot(self) -> tuple[plt.Figure, plt.Axes]:
        """
        Corrected Sampled and Sampled Return Period plot

        Returns
        -------
        fig
            plt.Figure
        ax
            plt.Axes
        """

        ecdf_annmax_probs_sim = np.arange(1, self.n_year_sim + 1) / (
            self.n_year_sim + 1
        )
        self.T_annmax_sim = 1 / (1 - ecdf_annmax_probs_sim)

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        # Fitted distribution
        ax.semilogx(
            self.T_years,
            self.ret_levels,
            color="red",
            linestyle="dashed",
            linewidth=2.5,
            label=f"Fitted {self.dist}",
        )
        # Confidence interval for fitted Distribution
        ax.semilogx(
            self.T_years,
            self.upper_ci_rp,
            color="tab:gray",
            linestyle="dotted",
            label=f"{self.conf} Conf. Band",
        )
        ax.semilogx(
            self.T_years, self.lower_ci_rp, color="tab:gray", linestyle="dotted"
        )

        # Corrected Sampled AM values
        ax.semilogx(
            self.T_annmax_sim,
            np.sort(self.sim_am_data_corr),
            color="tab:red",
            linewidth=0,
            marker="D",
            markersize=5,
            alpha=0.8,
            label="Corrected Sampled Annual Maxima",
        )
        # Corrected Sampled AM values
        ax.semilogx(
            self.T_annmax_sim,
            np.sort(self.sim_am_data),
            color="tab:red",
            linewidth=0,
            marker="o",
            markersize=5,
            alpha=0.8,
            label="Sampled Annual Maxima",
        )

        # Historical AM values
        ax.semilogx(
            self.T_annmax,
            np.sort(self.am_data),
            color="tab:blue",
            linewidth=0,
            marker="o",
            markersize=5,
            alpha=0.8,
            label="Historical Annual Maxima",
        )

        ax.set_xlabel("Return Periods (Years)")
        ax.set_ylabel(f"{self.var}")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 250, 1000, 10000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlim(left=0.9, right=self.n_year_sim + 100)
        ax.set_ylim(bottom=0)
        ax.legend(loc="best")
        ax.grid()

        return fig, ax

    def ecdf_plot(self) -> tuple[plt.Figure, plt.Axes]:
        """
        Empirical Cumulative Distribution Function plot for historical and
        synthetic before and after the correction.

        Returns
        -------
        fig
            plt.Figure
        ax
            plt.Axes
        """
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        # Historical ECDF
        ax.ecdf(
            self.pit_data, label="Historical", alpha=0.9, linewidth=2.5, color="gray"
        )
        # Synthetic ECDF
        ax.ecdf(
            self.sim_pit_data, label="Before", alpha=0.9, linewidth=2.5, color="black"
        )
        ax.ecdf(
            self.sim_pit_data_corrected,
            label="After",
            alpha=0.9,
            linewidth=2.5,
            color="black",
            linestyle="dashed",
        )

        # ax.set_xlim(self.threshold, 12)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(f"{self.var}")
        ax.set_ylabel("Probability")
        ax.grid()
        ax.legend(loc="lower right")
        ax.tick_params(axis="both", which="major")
        ax.tick_params(axis="both", which="minor")

        return fig, ax

    def correlations(self) -> dict:
        """
        Rank based correlations between sampled and corrected sampled data

        Returns
        -------
        dict :
            Dictionary with Spearman, Kendall and Pearson correlation coefficients.
            Keys :
            - "Spearman" : Spearman correlation coefficient
            - "Kendall" : Kendall correlation coefficient
            - "Pearson" : Pearson correlation coefficient
        """

        spearman_corr, _ = stats.spearmanr(
            self.sim_pit_data, self.sim_pit_data_corrected
        )
        kendall_corr, _ = stats.kendalltau(
            self.sim_pit_data, self.sim_pit_data_corrected
        )
        pearson_corr, _ = stats.pearsonr(self.sim_pit_data, self.sim_pit_data_corrected)

        return {
            "Spearman": spearman_corr,
            "Kendall": kendall_corr,
            "Pearson": pearson_corr,
        }
