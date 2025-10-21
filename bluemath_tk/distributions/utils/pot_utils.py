import numpy as np
from scipy.optimize import fminbound
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from csaps import csaps


def threshold_search(
    u_data: np.ndarray,
    e_data: np.ndarray,
    W_data: np.ndarray,
    plot: bool = False,
    folder: str = None,
):
    """
    Auxiliar function used in the studentidez_residuals method.

    Parameters
    ----------
    u_data : np.ndarray
        Threshold values
    e_data : np.ndarray
        Exceedances
    W_data : np.ndarray
        Weights vector
    plot : bool, default=False
        Flag for plotting
    folder : str, default=None
        File name to save plots

    Returns
    -------
    fitresult : ISmoothingSpline
        Fit object representing the smoothing spline fit
    threshold :
        Threshold value determined from the fit
    """

    if W_data is None:
        W_data = np.ones(u_data.size)

    orden = np.argsort(u_data)
    u_data = u_data[orden]
    e_data = e_data[orden]
    W_data = W_data[orden]

    # Fit: Smoothing spline
    u_mean = np.mean(u_data)
    u_std = np.std(u_data, ddof=1)

    def objective_function(x):
        return (smoothingspline(u_data, e_data, W_data, u_mean, u_std, x)[0] - 0.9) ** 2

    SmoothingParam = fminbound(objective_function, 0.5, 0.99)
    _, fitresult, _ = smoothingspline(
        u_data, e_data, W_data, u_mean, u_std, SmoothingParam
    )

    # Find the first zero from the left

    uc = np.linspace(u_data[0], u_data[-1], 2 * len(u_data))
    # uc = np.linspace(u_data[0], u_data[-1], 1000)
    ec = fitresult((uc - u_mean) / u_std)
    currentsign = np.sign(ec[0])

    ## If we want to show the fitted smoothing spline
    # if ploteat:
    # plt.figure(figsize=(10,6))
    # plt.plot(u_data,e_data, label="Data")
    # plt.plot(uc, ec, label="Fitted")
    # plt.title("Smoothing Spline Plot")
    # plt.xlabel("Threshold Values (u)")
    # plt.ylabel("Excedaances (e)")
    # plt.grid()
    # plt.show()

    zeroloc = [0, 0]
    cont = 0
    for i in range(1, len(ec)):
        if currentsign != np.sign(ec[i]):
            # Place midpoint into zeroloc[cont]
            zeroloc[cont] = (uc[i] + uc[i - 1]) / 2
            cont += 1
            currentsign = -currentsign
            if cont == 2:
                break

    pos1 = np.argwhere((u_data >= zeroloc[0]) & (u_data <= zeroloc[1]))
    posi = np.argmax(np.abs(e_data[pos1]))
    posi = pos1[0] + posi
    threshold = u_data[posi]
    mini = e_data[posi]

    if plot:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.plot(u_data, e_data, ".k", markersize=1.5, label="Data")
        ax.plot(
            [threshold] * 100,
            np.linspace(min(e_data), max(e_data), 100),
            "--",
            color=[0.5, 0.5, 0.5],
            linewidth=1.5,
        )
        ax.plot(
            threshold,
            mini,
            "ok",
            markersize=5,
            markerfacecolor="w",
            linewidth=2,
            label=f"Local optimum = {threshold.item()}",
        )
        ax.set_xlabel(r"Threshold $u$")
        ax.set_ylabel(r"$r^N$")
        ax.legend(loc="upper right")
        ax.grid()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        if folder is not None:
            plt.savefig(f"{folder}/thresholdlocation.png", dpi=300)
        plt.show()
        plt.close()

    return fitresult, threshold


def smoothingspline(
    x_data: np.ndarray,
    y_data: np.ndarray,
    w_data: np.ndarray,
    x_mean: float,
    x_std: float,
    lam: float,
):
    """
    Fits a smoothing spline to weighted data and calculates the goodness-of-fit (R^2).

    Parameters
    ----------
    x_data : np.ndarray
        Independent variable.
    y_data : np.ndarray
        Dependent variable.
    w_data : np.ndarray
        Weights for the fit.
    x_mean : float
        Mean of independent variable
    x_std : float
        Standard deviation of independent variable
    lam : float
        Smoothing parameter (controls the tradeoff between smoothness and fit).

    Returns
    -------
    fitresult : ISmoothingSpline
        The fitted spline model.
    r2 : float
        R-squared value of the fit.
    gof : dict
        Goodness-of-fit metrics containing R-squared.
    """
    # Normalize data
    x_norm = (x_data - x_mean) / x_std

    # Ensure strict increase in x for csaps by deduplicating normalized x
    x_unique, idx = np.unique(x_norm, return_index=True)
    y_use = y_data[idx]
    w_use = w_data[idx]

    # Final safety: if any nonpositive step remains (shouldn't after unique), nudge by eps
    dx = np.diff(x_unique)
    if np.any(dx <= 0):
        eps = np.finfo(float).eps
        bumps = np.maximum.accumulate((dx <= 0).astype(float))
        x_unique = x_unique + np.concatenate([[0.0], bumps]) * eps

    # Usando paquete CSAPS
    spline = csaps(x_unique, y_use, smooth=lam, weights=w_use)

    # Usando paquete de SCIPY
    # Fit smoothing spline (smoothing parameter scaled by data length)
    # s_value = SmoothingParam * len(x_data)
    # spline = UnivariateSpline(x_norm, y_data, w=w_data, s=s_value)

    # Compute fitted values
    y_fit = spline(x_norm)

    # Compute R-squared
    r2 = r2_score(y_data, y_fit)

    # Store goodness-of-fit metrics
    gof = {"rsquare": r2}

    return r2, spline, gof

    # @staticmethod


def RWLSfit(u, e, w):
    """
    Robust Weighted Least Squares (RWLS) regression.

    Parameters
    ----------
    u : np.ndarray
        Independent variable (predictor).
    e : np.ndarray
        Dependent variable (response).
    w : np.ndarray
        Weights for the weighted least squares fit.

    Returns
    -------
    beta : np.ndarray
        Estimated regression coefficients [intercept, slope].
    fobj : float
        Objective function value (weighted residual sum of squares).
    r : np.ndarray
        Residuals.
    rN : np.ndarray
        Internally studentized residuals.
    """
    if len(u) != len(e) or len(u) != len(w):
        raise ValueError(
            f"Error in the number of parameters (RWLSfit): input arrays must have the same length (u={len(u)}, e = {len(e)}, w={len(w)})."
        )

    # Data size
    n = len(u)

    # Design matrix X with intercept term
    X = np.column_stack((np.ones(n), u))
    Y = np.array(e)

    # Convert weights to diagonal matrix
    W = np.diag(w, 0)

    # Compute optimal estimates (beta)
    beta = np.linalg.inv(X.T @ W @ X) @ (
        X.T @ W @ Y
    )  # Equivalent to MATLAB: (X'*W*X)\(X'*W*Y)

    # Compute residuals
    r = Y - X @ beta

    # Objective function value (weighted residual sum of squares)
    fobj = r.T @ W @ r

    # Standard deviation of residuals
    sigres = np.sqrt(fobj / (n - 2))

    # Residual variance-covariance matrix
    # Hat or projection matrix
    P = X @ np.linalg.inv(X.T @ W @ X) @ X.T
    # Sensitivity matrix S = I - P * W
    # S = np.eye(n) - P @ W

    # Internally studentized residual
    rN = (np.sqrt(np.diag(W)) * r) / (sigres * np.sqrt(1 - np.diag(W) * np.diag(P)))

    return beta, fobj, r, rN