import numpy as np

from ..gev import GEV
from ..gpd import GPD
from ..pareto_poisson import GPDPoiss


def gpdpoiss_ci_rp_bootstrap(
    pot_data: np.ndarray,
    years: np.ndarray,
    threshold: float,
    poisson: float,
    B: int = 1000,
    conf_level: float = 0.95,
):
    """
    Compute the Confidence intervals for return periods of GPD-Poisson based on Bootstrap method.

    Parameters
    ----------
    B : int, default=1000
        Number of bootstrap samples.
    """
    probs_ci = 1 - 1 / years  # Convert to exceedance probabilities

    # Generate all bootstrap samples at once
    boot_samples = np.random.choice(pot_data, size=(B, pot_data.size), replace=True)

    # Vectorized parameter fitting
    boot_params = np.zeros((B, 3))
    for i in range(B):
        fit_result = GPD.fit(boot_samples[i], threshold=threshold)
        boot_params[i, :] = np.asarray(fit_result.params)

    return_periods = np.array(
        [
            GPDPoiss.qf(probs_ci, threshold, params[1], params[2], poisson)
            for params in boot_params
        ]
    )

    lower_ci_rp = np.quantile(return_periods, (1 - conf_level) / 2, axis=0)
    upper_ci_rp = np.quantile(return_periods, 1 - (1 - conf_level) / 2, axis=0)

    return lower_ci_rp, upper_ci_rp


def gev_ci_rp_bootstrap(
    am_data: np.ndarray, years: np.ndarray, B: int = 1000, conf_level: float = 0.95
):
    """
    Compute the Confidence intervals for return periods of GEV based on Bootstrap method.

    Parameters
    ----------
    B : int, default=1000
        Number of bootstrap samples.
    """
    probs_ci = 1 - 1 / years  # Convert to exceedance probabilities

    # Generate all bootstrap samples at once
    boot_samples = np.random.choice(am_data, size=(B, am_data.size), replace=True)

    # Vectorized parameter fitting
    boot_params = np.zeros((B, 3))
    for i in range(B):
        fit_result = GEV.fit(boot_samples[i])
        boot_params[i, :] = fit_result.params

    return_periods = np.array(
        [GEV.qf(probs_ci, params[0], params[1], params[2]) for params in boot_params]
    )

    lower_ci_rp = np.quantile(return_periods, (1 - conf_level) / 2, axis=0)
    upper_ci_rp = np.quantile(return_periods, 1 - (1 - conf_level) / 2, axis=0)

    return lower_ci_rp, upper_ci_rp
