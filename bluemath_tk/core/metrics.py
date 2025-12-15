import numpy as np


def bias(X_true: np.ndarray, X_pred: np.ndarray) -> float:
    """
    Calculate the BIAS.

    Parameters
    ----------
    X_true : np.ndarray
        True values.
    X_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        The BIAS value.
    """

    if len(X_true) != len(X_pred):
        raise ValueError("X_true and X_pred must have the same length")

    return float(sum(X_true - X_pred) / len(X_true))


def si(X_true: np.ndarray, X_pred: np.ndarray) -> float:
    """
    Calculate the Scatter Index (SI).

    Parameters
    ----------
    X_true : np.ndarray
        True values.
    X_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        The Scatter Index value.
    """

    if len(X_true) != len(X_pred):
        raise ValueError("X_true and X_pred must have the same length")

    return float(
        np.sqrt(
            sum(((X_true - X_true.mean()) - (X_pred - X_pred.mean())) ** 2)
            / sum(X_true**2)
        )
    )


def mse(X_true: np.ndarray, X_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error (MSE).

    Parameters
    ----------
    X_true : np.ndarray
        True values.
    X_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Mean squared error.
    """

    if len(X_true) != len(X_pred):
        raise ValueError("X_true and X_pred must have the same length")

    return float(np.mean((X_true - X_pred) ** 2))


def mae(X_true: np.ndarray, X_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    Parameters
    ----------
    X_true : np.ndarray
        True values.
    X_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Mean absolute error.
    """

    if len(X_true) != len(X_pred):
        raise ValueError("X_true and X_pred must have the same length")

    return float(np.mean(np.abs(X_true - X_pred)))


def rmse(X_true: np.ndarray, X_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).

    Parameters
    ----------
    X_true : np.ndarray
        True values.
    X_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Root mean squared error.
    """

    return float(np.sqrt(mse(X_true, X_pred)))


def r2(X_true: np.ndarray, X_pred: np.ndarray) -> float:
    """
    Calculate the R² score.

    Parameters
    ----------
    X_true : np.ndarray
        True values.
    X_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        The R² score.
    """

    if len(X_true) != len(X_pred):
        raise ValueError("X_true and X_pred must have the same length")

    return float(
        1.0 - np.sum((X_true - X_pred) ** 2) / np.sum((X_true - X_true.mean()) ** 2)
    )
