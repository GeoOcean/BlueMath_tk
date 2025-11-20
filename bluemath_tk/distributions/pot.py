import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import chi2, norm, pearsonr

from ..core.io import BlueMathModel
from .utils.pot_utils import RWLSfit, threshold_search


def block_maxima(
    x: np.ndarray,
    block_size: int | float = 365.25,
    min_sep: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to obtain the Block Maxima of given size taking into account
    minimum independence hypothesis

    Parameters
    ----------
    x : np.ndarray
        Data used to compute the Block Maxima
    block_size : int, default=5
        Size of BM in index units (if daily data, nº of days), by default 5
    min_sep : int, optional
        Minimum separation between maximas in index units, by default 2

    Returns
    -------
    idx : np.ndarray
        Indices of BM
    bmaxs : np.ndarray
        BM values

    Raises
    ------
    ValueError
        Minimum separation must be smaller than (block_size+1)/2

    Example
    -------
    >>> # 1-year of daily values
    >>> x = np.random.lognormal(1, 1.2, size=365)

    >>> # 5-day Block Maxima with 72h of independency
    >>> idx, bmaxs = block_maxima(
    >>>     x,
    >>>     block_size=5,
    >>>     min_sep=3
    >>> )
    """
    block_size = int(np.ceil(block_size))

    if min_sep > (block_size + 1) / 2:
        raise ValueError("min_sep must be smaller than (block_size+1)/2")

    x = np.asarray(x)
    n = x.size

    # Partition into non-overlapping blocks
    n_blocks = int(np.ceil(n / block_size))
    segments_idx = []
    segments = []
    blocks = []
    # For each block, keep a *ranked* list of (idx, value) candidates (desc by value)
    for b in range(n_blocks):
        start = b * block_size
        stop = min((b + 1) * block_size, n)
        segment = x[start:stop]

        candidates_idx = np.argsort(segment)[::-1]

        segments_idx.append(np.arange(start, stop))
        segments.append(segment)
        blocks.append(candidates_idx)

    def violates(i_left, i_right):
        if i_left is None or i_right is None:
            return False
        return i_right - i_left < min_sep

    changed = True
    while changed:
        changed = False

        for b in range(n_blocks - 1):
            idx_left = segments_idx[b][blocks[b][0]]
            idx_right = segments_idx[b + 1][blocks[b + 1][0]]
            if not violates(idx_left, idx_right):
                continue
            else:
                idx_block_adjust = (
                    0
                    if segments[b][blocks[b][0]] < segments[b + 1][blocks[b + 1][0]]
                    else 1
                )
                blocks[b + idx_block_adjust] = np.delete(
                    blocks[b + idx_block_adjust], 0
                )
                changed = True
                break

    bmaxs = np.asarray([segments[b][blocks[b][0]] for b in range(n_blocks)])
    idx = np.asarray([segments_idx[b][blocks[b][0]] for b in range(n_blocks)])

    return idx, bmaxs


def pot(
    data: np.ndarray,
    threshold: float = 0.0,
    n0: int = 10,
    min_peak_distance: int = 2,
    sig_level: float = 0.05,
):
    """
    Function to identiy POT
    This function identifies peaks in a dataset that exceed a specified
    threshold and computes statistics such as mean exceedances, variances,
    and weights for valid unique peaks. Peaks are considered independent if
    they are separated by a minimum distance.

    Parameters
    ----------
    data : np.ndarray(n,)
        Input time series or data array
    threshold : float, default=0
        Threshold above which peaks are extracted
    n0 : int, default=10
        Minimum number of exceedances required for valid computation
    min_peak_distance : int, default = 2
        Minimum distance between two peaks (in data points)
    sig_level : float, default=0.05
        Significance level for Chi-squared test

    Returns
    -------
    pks_unicos_valid : np.ndarray
        Valid unique peaks after removing NaN values
    excedencias_mean_valid : np.ndarray
        Mean exceedances for valid peaks
    excedencias_weight_valid : np.ndarray
        Weights based on exceedance variance for valid peaks
    pks : np.ndarray
        All detected peaks
    locs : np.ndarray
        Indices of the detected peaks in the data
    autocorrelations : np.ndarray(n, 3)
        Lags, correlations and pvalues to check the independence assumption
    """
    # Find peaks exceeding the threshold with specified min distance
    adjusted_data = np.maximum(data - threshold, 0)

    # Usamos la librería detecta que tiene el mismo funcionamiento que la función de matlab findpeaks
    locs, _ = find_peaks(adjusted_data, distance=min_peak_distance + 1)
    # Con scipy
    # locs, _ = find_peaks(adjusted_data, distance=min_peak_distance)

    pks = data[locs]

    # Calculate autocorrelation for lags 1 to 5 (if enough peaks)
    num_lags = 5
    if len(pks) > num_lags:
        autocorrelations = np.zeros((num_lags, 3), dtype=float)
        for i in range(num_lags):
            lag = i + 1
            r, p_value = pearsonr(pks[:-lag], pks[lag:])  # Test corr != 0
            autocorrelations[i, 0] = int(lag)
            autocorrelations[i, 1] = r
            autocorrelations[i, 2] = p_value

            if p_value < sig_level:
                Warning(
                    f"Lag {int(lag)} significant, consider increase the number of min_peak_distance"
                )
    else:
        # Not enough peaks for autocorrelation analysis
        autocorrelations = np.array([])

    # Unique peaks (pks_unicos), ignoring duplicates
    pks_unicos = np.unique(pks)

    # Allocate arrays to store mean exceedances, variances, and weights
    excedencias_mean = np.zeros(len(pks_unicos), dtype=float)
    excedencias_var = np.zeros(len(pks_unicos), dtype=float)
    excedencias_weight = np.zeros(len(pks_unicos), dtype=float)

    # Loop through each unique peak and calculate mean exceedances, variances, and weights
    for i in range(len(pks_unicos)):
        # Define the current unique peak
        pico_actual = pks_unicos[i]

        # Calculate the exceedances for peaks greater than the current unique peak
        excedencias = pks[pks > pico_actual]

        # If there are enough exceedances (greater than or equal to n0)
        if len(excedencias) >= n0:
            # Compute the mean exceedance (adjusted by the current peak)
            excedencias_mean[i] = np.mean(excedencias) - pico_actual
            # Compute the variance of the exceedances (ddof=1 to use the same variance as matlab)
            excedencias_var[i] = np.var(excedencias, ddof=1)
            # Compute the weight as the number of exceedances divided by the variance
            # Weight = number of exceedances / variance
            # (Guard against division by zero)
            if excedencias_var[i] != 0:
                excedencias_weight[i] = len(excedencias) / excedencias_var[i]
            else:
                excedencias_weight[i] = np.nan
        else:
            # If fewer than n0 exceedances, truncate arrays and stop
            excedencias_mean = excedencias_mean[:i]
            excedencias_var = excedencias_var[:i]
            excedencias_weight = excedencias_weight[:i]
            break

    # Trim the list of unique peaks to match the number of valid exceedances
    pks_unicos = pks_unicos[: len(excedencias_weight)]

    # Remove any NaN values from the peak and exceedance data to avoid issues in regression
    valid_indices = (
        ~np.isnan(pks_unicos)
        & ~np.isnan(excedencias_mean)
        & ~np.isnan(excedencias_weight)
    )
    pks_unicos_valid = pks_unicos[valid_indices]
    excedencias_mean_valid = excedencias_mean[valid_indices]
    excedencias_weight_valid = excedencias_weight[valid_indices]

    return (
        pks_unicos_valid,
        excedencias_mean_valid,
        excedencias_weight_valid,
        pks,
        locs,
        autocorrelations,
    )


class OptimalThreshold(BlueMathModel):
    """
    Class to compute the optimal threshold using different algorithms.

    Methods
    -------
    studentidez_residuals :
        Function to compute the threshold using the studentidez resiudals

    Notes
    -----
    The list of methods implemented to select the optimal threshold are:
    - Studentidez residuals method Mínguez (2025) [1].


    [1] Mínguez, R. (2025). Automatic Threshold Selection for Generalized
    Pareto and Pareto–Poisson Distributions in Rainfall Analysis: A Case
    Study Using the NOAA NCDC Daily Rainfall Database. Atmosphere, 16(1),
    61. https://doi.org/10.3390/atmos16010061
    """

    def __init__(
        self,
        data,
        threshold: float = 0.0,
        n0: int = 10,
        min_peak_distance: int = 2,
        sig_level: float = 0.05,
        method: str = "studentized",
        plot: bool = False,
        folder: str = None,
        display_flag: bool = False,
    ):
        self.data = data
        self.threshold = threshold
        self.n0 = n0
        self.min_peak_distance = min_peak_distance
        (
            self.pks_unicos_valid,
            self.excedencias_mean_valid,
            self.excedencias_weight_valid,
            self.pks,
            self.locs,
            self.autocorrelations,
        ) = pot(self.data, threshold, n0, min_peak_distance)
        self.method = method
        self.sig_level = sig_level
        self.plot = plot
        self.folder = folder
        self.display_flag = display_flag

    def fit(
        self,
    ):
        """
        Obtain the optimal threshold and POTs given the selected method

        Returns
        -------
        threshold : float
            Optimal threshold
        pks : np.ndarray
            POT
        pks_idx : np.ndarray
            Indices of POT
        """
        if self.method == "studentized":
            self.threshold, beta, fobj, r = self.studentized_residuals(
                self.pks_unicos_valid,
                self.excedencias_mean_valid,
                self.excedencias_weight_valid,
                self.sig_level,
                self.plot,
                self.folder,
                self.display_flag,
            )

        # TODO: Añadir más metodos

        _, _, _, self.pks, self.pks_idx, _ = pot(
            self.data, self.threshold, self.n0, self.min_peak_distance
        )
        return self.threshold.item(), self.pks, self.pks_idx

    def studentized_residuals(
        self,
        pks_unicos_valid: np.ndarray,
        exceedances_mean_valid: np.ndarray,
        exceedances_weight_valid: np.ndarray,
        sig_level: float = 0.05,
        plot: bool = False,
        folder: str = None,
        display_flag: bool = False,
    ):
        """
        Function to compute the optimal threshold based on Chi-Squared
        and studentized residuals. Optionally plot the results if plot_flag is true and
        displays messages if display_flag is true.

        Parameters
        ----------
        pks_unicos_valid : np.ndarray(n,)
            Vector of unique peaks (potential thresholds)
        exceedances_mean_valid : np.ndarray(n,)
            Vector of exceedance means
        exceedances_weight_valid : np.ndarray(n,)
            Vector of exceedance weights
        sig_level : bool, default=0.05
            Significance level for Chi-squared test
        plot_flag : bool, default=False
            Boolean flag, true to plot the graphs, false otherwise
        folder : str, default=None
            Path and name for making graphs
        display_flag : bool, default=False
            Boolean flag, true to display messages, false otherwise

        Returns
        -------
        threshold :
            Optimal threshold found
        beta : np.ndarray
            Optimal regression coefficients
        fobj :
            Optimal objective function (weighted leats squares)
        r : np.ndarray
            Optimal residuals
        """

        stop_search = 0
        it = 1
        threshold = pks_unicos_valid[0]  # Initial threshold

        while stop_search == 0 and it <= 10:
            # Find the current threshold in the pks_unicos_valid array
            pos = np.argwhere(pks_unicos_valid == threshold).item()
            u_values = pks_unicos_valid[pos:]  # Threshold starting from the current one
            e_values = exceedances_mean_valid[pos:]  # Exceedances
            w_values = exceedances_weight_valid[pos:]  # Weights

            # Perform the RWLS fitting and calculate studentidez residuals
            beta, fobj, r, rN = RWLSfit(u_values, e_values, w_values)

            # Plot resudals if required
            if plot:
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111)
                ax.plot(
                    u_values,
                    rN,
                    "k",
                    linewidth=1.5,
                    label=f"Internally Studentized Residuals\nMin threshold = {threshold.item()}",
                )
                ax.set_xlabel(r"Threshold $u$")
                ax.set_ylabel(r"$r$")
                ax.set_title(f"Studentized Residuals Iteration {it}")
                # ax.text(threshold + 0.5, min(rN) + 0.1 * (max(rN) - min(rN)), f'Min threshold = {threshold}')
                ax.grid()
                ax.legend(loc="upper right")
                if folder is not None:
                    plt.savefig(f"{folder}/StudenRes{it}.png", dpi=300)
                # plt.show()
                plt.close()

            if fobj > chi2.ppf(1 - sig_level, df=u_values.size - 2) or np.abs(
                rN[0]
            ) > norm.ppf(1 - sig_level / 2, 0, 1):
                if display_flag:
                    if fobj > chi2.ppf(1 - sig_level, df=u_values.size - 2):
                        print("Chi-squared test detects anomalies")
                    if np.abs(rN[0]) > norm.ppf(1 - sig_level / 2, 0, 1):
                        print(
                            "The maximum studentized residual of the first record detects anomalies"
                        )

                thresholdsearch = 1

            else:
                thresholdsearch = 0
                stop_search = 1  # If criteria met, stop the loop

            if thresholdsearch:
                if display_flag:
                    print(
                        f"Maximum sensitivity = {np.max(np.abs(rN))} and thus the optimal threshold seems to be on the right side of the minimum sample value, looking for the location"
                    )

                _, threshold = threshold_search(
                    u_values,
                    rN,
                    w_values,
                    plot,
                    folder,
                )
                if display_flag:
                    print(f"New threshold found: {threshold}")

            it += 1

        return threshold, beta, fobj, r

    def potplot(
        self,
        time: np.ndarray = None,
        ax: plt.Axes = None,
        figsize: tuple = (8, 5),
    ):
        """
        Auxiliar function which call generic potplot to plot the POT usign the optimal threshold obtained

        Parameters
        ----------
        time : np.ndarray, default=None
            Time of data
        ax : plt.Axes, default=None
            Axes
        figsize : tuple, default=(8,5)
            Figure size, by default (8, 5)

        Returns
        -------
        fig : plt.Figure
            Figure
        ax : plt.Axes
            Axes
        """
        fig, ax = potplot(
            self.data,
            self.threshold,
            time,
            self.n0,
            self.min_peak_distance,
            self.sig_level,
            ax,
            figsize,
        )

        return fig, ax


def potplot(
    data: np.ndarray,
    threshold: float = 0.0,
    time: np.ndarray = None,
    n0: int = 10,
    min_peak_distance: int = 2,
    sig_level: float = 0.05,
    ax: plt.Axes = None,
    figsize: tuple = (8, 5),
):
    """
    Plot the POT for data given a threshold.

    Parameters
    ----------
    data : np.ndarray
        Data
    threshold : float, default=0.0
        Threshold used to plot the peaks
    time : np.ndarray, default=None
        Time of data
    n0 : int, default=10
        Minimum number of data to compute the POT given a threshold
    min_peak_distance : int, default=2
        Minimum peak distance between POT (in index size)
    sig_level : float, default=0.05
        Significance level for Chi-squared test
    ax : plt.Axes, default=None
        Axes
    figsize : tuple, default=(8,5)
        Figure figsize

    Returns
    -------
    fig : plt.Figure
        Figure
    ax : plt.Axes
        Axes
    """
    _, _, _, _, pks_idx, _ = pot(data, threshold, n0, min_peak_distance, sig_level)

    if time is None:
        time = np.arange(data.size)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()

    # Plot complete series
    ax.plot(time, data, ls="-", color="#5199FF", lw=0.25, zorder=10)

    # Plot extremes
    ax.scatter(
        time[pks_idx],
        data[pks_idx],
        s=20,
        lw=0.5,
        edgecolor="w",
        facecolor="#F85C50",
        zorder=20,
    )

    ax.axhline(threshold, ls="--", lw=1, color="#FF756B", zorder=15)
    ax.set_xlabel("Time")

    return fig, ax


def mrlp(
    data: np.ndarray,
    threshold: float = None,
    conf_level: float = 0.95,
    ax: plt.Axes = None,
    figsize: tuple = (8, 5),
) -> plt.Axes:
    """
    Plot mean residual life for given threshold value.

    The mean residual life plot should be approximately linear above a threshold
    for which the Generalized Pareto Distribution model is valid.
    The strategy is to select the smallest threshold value immediately above
    which the plot is approximately linear.

    Parameters
    ----------
    data : np.ndarray
        Time series of the signal.
    threshold : float, optional
        An array of thresholds for which the mean residual life plot is plotted.
        If None (default), starting in the 90th quantile
    conf_level : float, default=0.95
        Confidence interval width in the range (0, 1), by default it is 0.95.
        If None, then confidence interval is not shown.
    ax : matplotlib.axes._axes.Axes, optional
        If provided, then the plot is drawn on this axes.
        If None (default), new figure and axes are created
    figsize : tuple, optional
        Figure size in inches in format (width, height).
        By default it is (8, 5).

    Returns
    -------
    matplotlib.axes._axes.Axes
        Axes object.

    """
    if threshold is None:
        threshold = np.nanquantile(data, q=0.9)

    (
        pks_unicos_valid,
        excedencias_mean_valid,
        excedencias_weight_valid,
        pks,
        locs,
        autocorrelations,
    ) = pot(data, threshold)

    if conf_level is not None:
        interlow, interup = norm.interval(
            0.95,
            loc=excedencias_mean_valid,
            scale=np.sqrt(1 / excedencias_weight_valid),
        )

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
        ax.grid(False)

    # Plotting central estimates of mean re sidual life
    ax.plot(
        pks_unicos_valid,
        excedencias_mean_valid,
        color="#F85C50",
        lw=2,
        ls="-",
        zorder=15,
    )

    if conf_level is not None:
        ax.plot(pks_unicos_valid, interlow, color="#5199FF", lw=1, ls="--", zorder=10)
        ax.plot(pks_unicos_valid, interup, color="#5199FF", lw=1, ls="--", zorder=10)

        ax.fill_between(
            pks_unicos_valid,
            interlow,
            interup,
            facecolor="#5199FF",
            edgecolor="None",
            alpha=0.25,
            zorder=5,
        )

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Mean excess")

    return ax
