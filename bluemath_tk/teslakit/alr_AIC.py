from collections import OrderedDict
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.discrete.discrete_model as sm
import xarray as xr


class AutoLogisticRegression:
    """
    Auto-Regressive Logistic (ALR) model for categorical time series modeling.

    This implementation focuses on statsmodels with AIC-based covariate selection.
    The model simulates transitions between discrete states (weather types, climate
    patterns, etc.) accounting for:
    - Markov dependencies (previous states)
    - Seasonality
    - Long-term trends
    - External covariates

    Attributes
    ----------
    cluster_size : int
        Number of discrete states or clusters in the model
    model : statsmodels.MNLogit
        Fitted multinomial logistic regression model
    terms_fit : dict
        Dictionary of terms used for model fitting
    terms_fit_names : list
        Names of the terms used for model fitting
    """

    def __init__(self, cluster_size: int):
        """
        Initialize the ALR model.

        Parameters
        ----------
        cluster_size : int
            Number of discrete states/clusters in the model
        """

        self.cluster_size = cluster_size
        self.model = None
        self.terms_fit = {}
        self.terms_fit_names = []
        self.mk_order = 0
        self.cov_names = []

    def get_year_fraction(self, time_array):
        """
        Convert datetime array to fractional year representation.

        Parameters
        ----------
        time_array : array-like
            Array of datetime objects

        Returns
        -------
        np.ndarray
            Array of year fractions (0-1 for position within the year)
        """

        time_yfrac = np.zeros(len(time_array))
        for i, dt in enumerate(time_array):
            doy = dt.timetuple().tm_yday
            year_length = 366 if dt.year % 4 == 0 else 365
            time_yfrac[i] = dt.year + doy / year_length

        return time_yfrac

    def generate_terms(
        self,
        bmus: np.ndarray,
        time: np.ndarray,
        mk_order: int = 0,
        use_constant: bool = True,
        use_long_term: bool = False,
        seasonality: Tuple[bool, List[int]] = (False, []),
        covariates: Optional[xr.Dataset] = None,
        cov_seasonality: bool = False,
    ) -> Tuple[OrderedDict, List[str]]:
        """
        Generate design matrix terms for the ALR model.

        Parameters
        ----------
        bmus : np.ndarray
            Array of categorical states (1-indexed)
        time : np.ndarray
            Array of datetime objects
        mk_order : int, optional
            Markov chain order (number of previous states to include)
        use_constant : bool, optional
            Include intercept term
        use_long_term : bool, optional
            Include long-term trend
        seasonality : tuple(bool, list), optional
            Tuple of (use_seasonality, [phases]). Example: (True, [1, 2]) for annual
            and semi-annual cycles
        covariates : xr.Dataset, optional
            Dataset with covariates. Should have dimensions (time, cov_names) and variable cov_values
        cov_seasonality : bool, optional
            Include seasonal modulation of covariates

        Returns
        -------
        OrderedDict
            Dictionary of design matrix terms
        list
            List of term names
        """

        # Convert time to year fractions
        time_yfrac = self.get_year_fraction(time)

        # Initialize containers
        terms = OrderedDict()
        terms_names = []

        # Constant term (intercept)
        if use_constant:
            terms["constant"] = np.ones((bmus.size, 1))
            terms_names.append("intercept")

        # Long-term trend
        if use_long_term:
            terms["long_term"] = np.ones((bmus.size, 1))
            terms["long_term"][:, 0] = time_yfrac
            terms_names.append("long_term")

        # Seasonality terms (harmonic functions)
        if seasonality[0]:
            phases = seasonality[1]
            temp_seas = np.zeros((len(time_yfrac), 2 * len(phases)))

            for i, harmonic in enumerate(phases):
                # Add cosine and sine components for each harmonic
                temp_seas[:, 2 * i] = np.cos(harmonic * 2 * np.pi * (time_yfrac % 1))
                temp_seas[:, 2 * i + 1] = np.sin(
                    harmonic * 2 * np.pi * (time_yfrac % 1)
                )

                terms_names.append(f"cos_{harmonic}")
                terms_names.append(f"sin_{harmonic}")

            terms["seasonality"] = temp_seas

        # External covariates
        if covariates is not None:
            cov_values = covariates.cov_values.values
            cov_names = covariates.cov_names.values
            self.cov_names = cov_names

            # Normalize covariates
            cov_norm = (cov_values - cov_values.mean(axis=0)) / cov_values.std(axis=0)

            # Add each covariate as a separate term
            for i, name in enumerate(cov_names):
                terms[name] = np.reshape(cov_norm[:, i], (cov_norm.shape[0], 1))
                terms_names.append(name)

                # Add seasonal modulation of covariates if requested
                if cov_seasonality:
                    cos_term = terms[name] * np.cos(
                        2 * np.pi * (time_yfrac % 1)
                    ).reshape(-1, 1)
                    sin_term = terms[name] * np.sin(
                        2 * np.pi * (time_yfrac % 1)
                    ).reshape(-1, 1)

                    terms[f"{name}_cos"] = cos_term
                    terms[f"{name}_sin"] = sin_term

                    terms_names.append(f"{name}_cos")
                    terms_names.append(f"{name}_sin")

        # Markov terms (previous states)
        if mk_order > 0:
            # Define Helmert contrasts for categorical variables
            def helmert_coding(k):
                """Generate Helmert contrast matrix for k categories"""
                H = np.zeros((k, k - 1))
                for i in range(k - 1):
                    H[i, i] = (k - i - 1) / (k - i)
                    H[(i + 1) :, i] = -1.0 / (k - i)
                return H

            # Generate contrast matrix
            contrast_matrix = helmert_coding(self.cluster_size)

            # Add terms for each lag in the Markov order
            for i in range(mk_order):
                # Initialize terms for this lag
                Z = np.zeros((bmus.size, self.cluster_size - 1))

                # Set values using the contrast matrix
                for j in range(bmus.size - i - 1):
                    bmu_idx = int(bmus[j]) - 1  # Convert 1-indexed to 0-indexed
                    Z[j + i + 1, :] = contrast_matrix[bmu_idx, :]

                # Add to terms
                terms[f"markov_{i + 1}"] = Z

                # Add term names
                for c in range(self.cluster_size - 1):
                    terms_names.append(f"mk{i + 1}_{c + 1}")

        return terms, terms_names

    def fit(
        self,
        bmus: np.ndarray,
        time: np.ndarray,
        mk_order: int = 0,
        use_constant: bool = True,
        use_long_term: bool = False,
        seasonality: Tuple[bool, List[int]] = (False, []),
        covariates: Optional[xr.Dataset] = None,
        cov_seasonality: bool = False,
        max_iter: int = 1000,
    ) -> None:
        """
        Fit the ALR model to data.

        Parameters
        ----------
        bmus : np.ndarray
            Array of categorical states (1-indexed)
        time : np.ndarray
            Array of datetime objects
        mk_order : int, optional
            Markov chain order (number of previous states to include)
        use_constant : bool, optional
            Include intercept term
        use_long_term : bool, optional
            Include long-term trend
        seasonality : tuple(bool, list), optional
            Tuple of (use_seasonality, [phases])
        covariates : xr.Dataset, optional
            Dataset with covariates
        cov_seasonality : bool, optional
            Include seasonal modulation of covariates
        max_iter : int, optional
            Maximum iterations for model fitting
        """

        self.mk_order = mk_order

        # Generate design matrix terms
        self.terms_fit, self.terms_fit_names = self.generate_terms(
            bmus=bmus,
            time=time,
            mk_order=mk_order,
            use_constant=use_constant,
            use_long_term=use_long_term,
            seasonality=seasonality,
            covariates=covariates,
            cov_seasonality=cov_seasonality,
        )

        # Combine terms into design matrix
        X = np.concatenate(list(self.terms_fit.values()), axis=1)
        y = bmus

        # Convert to pandas for statsmodels
        X_df = pd.DataFrame(X, columns=self.terms_fit_names)
        y_df = pd.DataFrame(y, columns=["bmus"])

        print(
            f"Fitting multinomial logistic regression with {len(self.terms_fit_names)} terms..."
        )

        # Fit the model
        self.model = sm.MNLogit(y_df, X_df).fit(
            method="lbfgs", maxiter=max_iter, disp=True
        )

        print("Model fitting complete.")

    def select_covariates_by_aic(
        self,
        bmus: np.ndarray,
        time: np.ndarray,
        covariates: xr.Dataset,
        base_config: dict = None,
    ) -> Tuple[List[str], float]:
        """
        Select optimal covariates using AIC criterion.

        Parameters
        ----------
        bmus : np.ndarray
            Array of categorical states (1-indexed)
        time : np.ndarray
            Array of datetime objects
        covariates : xr.Dataset
            Dataset with all potential covariates
        base_config : dict, optional
            Base configuration for other terms (mk_order, seasonality, etc.)

        Returns
        -------
        list
            List of selected covariate names
        float
            AIC of the best model
        """

        # Default base configuration
        if base_config is None:
            base_config = {
                "mk_order": 1,
                "use_constant": True,
                "use_long_term": False,
                "seasonality": (True, [1]),
                "cov_seasonality": False,
            }

        cov_names = covariates.cov_names.values
        n_covs = len(cov_names)

        print(f"Starting covariate selection with {n_covs} potential covariates...")

        # Initialize
        best_aic = float("inf")
        best_covs = []
        all_results = []

        # Test each covariate individually first
        for i, cov in enumerate(cov_names):
            # Create subset of covariates with just this one
            cov_subset = xr.Dataset(
                data_vars={
                    "cov_values": (
                        ("time", "cov_names"),
                        covariates.cov_values.values[:, i : i + 1],
                    ),
                },
                coords={"time": covariates.time, "cov_names": [cov]},
            )

            # Fit model
            try:
                self.fit(
                    bmus=bmus,
                    time=time,
                    mk_order=base_config["mk_order"],
                    use_constant=base_config["use_constant"],
                    use_long_term=base_config["use_long_term"],
                    seasonality=base_config["seasonality"],
                    covariates=cov_subset,
                    cov_seasonality=base_config["cov_seasonality"],
                )

                # Get AIC
                aic = self.model.aic
                all_results.append((aic, [cov]))

                print(f"Covariate {cov}: AIC = {aic:.2f}")

                # Update best model if improved
                if aic < best_aic:
                    best_aic = aic
                    best_covs = [cov]

            except Exception as e:
                print(f"Error fitting model with covariate {cov}: {str(e)}")

        # Forward selection - iteratively add variables
        current_covs = best_covs.copy()
        current_aic = best_aic

        improved = True
        while improved and len(current_covs) < n_covs:
            improved = False
            best_new_aic = float("inf")
            best_new_cov = None

            # Try adding each remaining covariate
            for cov in cov_names:
                if cov not in current_covs:
                    test_covs = current_covs + [cov]

                    # Create subset of covariates
                    cov_indices = [list(cov_names).index(c) for c in test_covs]
                    cov_subset = xr.Dataset(
                        data_vars={
                            "cov_values": (
                                ("time", "cov_names"),
                                covariates.cov_values.values[:, cov_indices],
                            ),
                        },
                        coords={"time": covariates.time, "cov_names": test_covs},
                    )

                    # Fit model
                    try:
                        self.fit(
                            bmus=bmus,
                            time=time,
                            mk_order=base_config["mk_order"],
                            use_constant=base_config["use_constant"],
                            use_long_term=base_config["use_long_term"],
                            seasonality=base_config["seasonality"],
                            covariates=cov_subset,
                            cov_seasonality=base_config["cov_seasonality"],
                        )

                        # Get AIC
                        aic = self.model.aic
                        all_results.append((aic, test_covs))

                        print(f"Covariates {test_covs}: AIC = {aic:.2f}")

                        # Update best candidate
                        if aic < best_new_aic:
                            best_new_aic = aic
                            best_new_cov = cov

                    except Exception as e:
                        print(
                            f"Error fitting model with covariates {test_covs}: {str(e)}"
                        )

            # Check if adding the best new covariate improves AIC
            if best_new_aic < current_aic:
                improved = True
                current_aic = best_new_aic
                current_covs.append(best_new_cov)

                # Update global best if improved
                if current_aic < best_aic:
                    best_aic = current_aic
                    best_covs = current_covs.copy()

                print(f"Added covariate {best_new_cov}, new AIC = {current_aic:.2f}")

        # Final model with best covariates
        if best_covs:
            print(f"\nBest model has AIC = {best_aic:.2f} with covariates: {best_covs}")

            # Create final covariate subset
            cov_indices = [list(cov_names).index(c) for c in best_covs]
            best_cov_subset = xr.Dataset(
                data_vars={
                    "cov_values": (
                        ("time", "cov_names"),
                        covariates.cov_values.values[:, cov_indices],
                    ),
                },
                coords={"time": covariates.time, "cov_names": best_covs},
            )

            # Fit final model
            self.fit(
                bmus=bmus,
                time=time,
                mk_order=base_config["mk_order"],
                use_constant=base_config["use_constant"],
                use_long_term=base_config["use_long_term"],
                seasonality=base_config["seasonality"],
                covariates=best_cov_subset,
                cov_seasonality=base_config["cov_seasonality"],
            )

        return best_covs, best_aic

    def simulate(
        self,
        time_sim: np.ndarray,
        num_sims: int = 1,
        covariates_sim: Optional[xr.Dataset] = None,
    ) -> np.ndarray:
        """
        Generate simulations from the fitted ALR model.

        Parameters
        ----------
        time_sim : np.ndarray
            Array of datetime objects for simulation
        num_sims : int, optional
            Number of simulations to generate
        covariates_sim : xr.Dataset, optional
            Covariates for simulation period

        Returns
        -------
        np.ndarray
            Array of simulated states with shape (time, num_sims)
        """

        if self.model is None:
            raise ValueError("Model must be fitted before simulation")

        # Get basic parameters
        mk_order = self.mk_order

        # Initialize output array
        sim_bmus = np.zeros((len(time_sim), num_sims))

        # Initial values (random start)
        for n in range(num_sims):
            # Initialize with random values
            init_bmus = np.random.randint(1, self.cluster_size + 1, mk_order)
            sim_bmus[:mk_order, n] = init_bmus

        # Main simulation loop
        print(f"Simulating {num_sims} time series...")

        for n in range(num_sims):
            for i in range(mk_order, len(time_sim)):
                # Extract current simulation history
                hist_bmus = sim_bmus[i - mk_order : i, n]

                # Generate terms for this step
                terms_i, _ = self.generate_terms(
                    bmus=np.append(hist_bmus, 0),
                    time=time_sim[i - mk_order : i + 1],
                    mk_order=mk_order,
                    use_constant=True,  # Assuming constant is always used
                    use_long_term="long_term" in self.terms_fit,
                    seasonality=(
                        "seasonality" in self.terms_fit,
                        [1] if "seasonality" in self.terms_fit else [],
                    ),
                    covariates=covariates_sim.sel(time=time_sim[i : i + 1])
                    if covariates_sim is not None
                    else None,
                    cov_seasonality=any(
                        k.endswith("_cos") for k in self.terms_fit.keys()
                    ),
                )

                # Prepare prediction matrix
                X = np.concatenate(list(terms_i.values()), axis=1)

                # Get transition probabilities
                probs = self.model.predict(X)
                cum_probs = np.cumsum(probs[-1, :])

                # Generate random state based on probabilities
                rnd_val = np.random.rand()
                new_bmu = np.where(cum_probs > rnd_val)[0][0] + 1

                # Store result
                sim_bmus[i, n] = new_bmu

        return sim_bmus

    def get_summary(self) -> pd.DataFrame:
        """
        Get model summary statistics.

        Returns
        -------
        pd.DataFrame
            DataFrame with parameter estimates, p-values, etc.
        """

        if self.model is None:
            raise ValueError("Model must be fitted first")

        # Extract key information
        params = self.model.params.transpose()
        pvalues = self.model.pvalues.transpose()
        conf_int = self.model.conf_int()

        # Organize into a DataFrame
        summary = pd.DataFrame()

        # for cluster in range(2, self.cluster_size + 1):
        #     cluster_params = params.iloc[cluster - 1].reset_index()
        #     cluster_params.columns = ["term", "coefficient"]

        #     cluster_pvals = pvalues.iloc[cluster - 1].reset_index()
        #     cluster_pvals.columns = ["term", "p_value"]

        # # Get confidence intervals
        # ci_lower = conf_int.xs(cluster - 1, level=1)[0].reset_index()
        # ci_upper = conf_int.xs(cluster - 1, level=1)[1].reset_index()

        # ci_lower.columns = ["term", "ci_lower"]
        # ci_upper.columns = ["term", "ci_upper"]

        # # Combine all info
        # cluster_info = cluster_params.merge(cluster_pvals, on="term")
        # cluster_info = cluster_info.merge(ci_lower, on="term")
        # cluster_info = cluster_info.merge(ci_upper, on="term")

        # # Add cluster identifier
        # cluster_info["cluster"] = cluster

        # # Add significance stars
        # cluster_info["significance"] = ""
        # cluster_info.loc[cluster_info["p_value"] < 0.05, "significance"] = "*"
        # cluster_info.loc[cluster_info["p_value"] < 0.01, "significance"] = "**"
        # cluster_info.loc[cluster_info["p_value"] < 0.001, "significance"] = "***"

        # # Append to summary
        # summary = pd.concat([summary, cluster_info])

        # Add model metrics
        print("Model Information:")
        print(f"AIC: {self.model.aic:.2f}")
        print(f"BIC: {self.model.bic:.2f}")
        print(f"Log-Likelihood: {self.model.llf:.2f}")
        print(f"Pseudo R-squared: {self.model.prsquared:.4f}")

        # return summary.sort_values(["cluster", "term"]).reset_index(drop=True)


if __name__ == "__main__":
    from datetime import datetime, timedelta

    import numpy as np
    import pandas as pd
    import xarray as xr

    # Create sample data
    np.random.seed(42)

    # Create time array
    start_date = datetime(2000, 1, 1)
    days = 1000
    time_array = np.array([start_date + timedelta(days=i) for i in range(days)])

    # Generate weather states (1-6)
    bmus = np.random.randint(1, 7, size=days)

    # Add some seasonality to states
    for i in range(days):
        day_of_year = time_array[i].timetuple().tm_yday
        if day_of_year < 100:  # Winter
            bmus[i] = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        elif day_of_year > 250:  # Fall
            bmus[i] = np.random.choice([4, 5, 6], p=[0.5, 0.3, 0.2])

    # Create some covariates
    cov1 = np.sin(np.linspace(0, 10 * np.pi, days)) + np.random.normal(
        0, 0.2, days
    )  # Oscillating
    cov2 = np.linspace(-1, 1, days) + np.random.normal(0, 0.1, days)  # Trend
    cov3 = np.random.normal(0, 1, days)  # Random noise

    # Create dataset with covariates
    covariates = xr.Dataset(
        data_vars={
            "cov_values": (("time", "cov_names"), np.column_stack([cov1, cov2, cov3])),
        },
        coords={"time": time_array, "cov_names": ["oscillation", "trend", "noise"]},
    )

    # Initialize and fit the ALR model
    alr_model = AutoLogisticRegression(cluster_size=6)

    # Select best covariates using AIC
    best_covs, best_aic = alr_model.select_covariates_by_aic(
        bmus=bmus,
        time=time_array,
        covariates=covariates,
        base_config={
            "mk_order": 1,
            "use_constant": True,
            "use_long_term": True,
            "seasonality": (True, [1, 2]),
            "cov_seasonality": True,
        },
    )

    # Get model summary
    summary = alr_model.get_summary()
    print("\nModel Parameter Summary:")
    print(summary)

    # Generate future simulation
    future_days = 100
    future_time = np.array(
        [start_date + timedelta(days=days + i) for i in range(future_days)]
    )

    # Create future covariates
    future_cov1 = np.sin(
        np.linspace(10 * np.pi, 11 * np.pi, future_days)
    ) + np.random.normal(0, 0.2, future_days)
    future_cov2 = np.linspace(1, 1.2, future_days) + np.random.normal(
        0, 0.1, future_days
    )
    future_cov3 = np.random.normal(0, 1, future_days)

    # Create dataset with future covariates (using only selected covariates)
    cov_indices = [list(covariates.cov_names.values).index(c) for c in best_covs]
    future_covs = np.column_stack(
        [[future_cov1, future_cov2, future_cov3][i] for i in cov_indices]
    )

    future_covariates = xr.Dataset(
        data_vars={
            "cov_values": (("time", "cov_names"), future_covs),
        },
        coords={"time": future_time, "cov_names": best_covs},
    )

    # Generate simulations
    simulations = alr_model.simulate(
        time_sim=future_time, num_sims=5, covariates_sim=future_covariates
    )

    print("\nSimulation Results (first 10 time steps):")
    print(simulations[:10, :])
