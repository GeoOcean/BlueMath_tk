import numpy as np

from ._base_distributions import BaseDistribution


class GPDPoiss(BaseDistribution):
    def __init__(self) -> None:
        """
        Initialize the GPD-Poisson mode
        """
        super().__init__()

    @property
    def name() -> str:
        return "GPD-Poisson model"

    @property
    def nparams() -> int:
        """
        Number of parameters of GPD-Poisson model
        """
        return int(4)

    @property
    def param_names() -> list[str]:
        """
        Name of parameters of GPD-Poisson
        """
        return ["threshold", "scale", "shape", "poisson"]

    @staticmethod
    def pdf(
        x: np.ndarray,
        threshold: float = 0.0,
        scale: float = 1.0,
        shape: float = 0.0,
        poisson: float = 1.0,
    ) -> np.ndarray:
        """
        Probability density function

        Parameters
        ----------
        x : np.ndarray
            Values to compute the probability density value
        threshold : float, default=0.0
            Threshold parameter
        scale : float, default = 1.0
            Scale parameter.
            Must be greater than 0.
        shape : float, default = 0.0
            Shape parameter.
        poisson : float, default = 1.0

        Returns
        ----------
        pdf : np.ndarray
            Probability density function values

        Raises
        ------
        ValueError
            If scale is not greater than 0.
        ValueError
            If poisson is not greater than 0.
        """

        if scale <= 0:
            raise ValueError("Scale parameter must be > 0")

        if poisson <= 0:
            raise ValueError("Poisson parameter must be > 0")

        y = np.maximum(x - threshold, 0) / scale

        # # Gumbel case (shape = 0)
        # if shape == 0.0:
        #     pdf = (1 / scale) * (np.exp(-y) * np.exp(-np.exp(-y)))

        # # General case (Weibull and Frechet, shape != 0)
        # else:
        #     pdf = np.full_like(x, 0, dtype=float)  # 0
        #     yy = 1 + shape * y
        #     yymask = yy > 0
        #     pdf[yymask] = (1 / scale) * (
        #         yy[yymask] ** (-1 - (1 / shape)) * np.exp(-(yy[yymask] ** (-1 / shape)))
        #     )

        # return pdf

    @staticmethod
    def qf(
        p: np.ndarray,
        threshold: float = 0.0,
        scale: float = 1.0,
        shape: float = 0.0,
        poisson: float = 1.0,
    ) -> np.ndarray:
        """
        Quantile function (Inverse of Cumulative Distribution Function)

        Parameters
        ----------
        p : np.ndarray
            Probabilities to compute their quantile
        loc : float, default=0.0
            Location parameter
        scale : float, default = 1.0
            Scale parameter.
            Must be greater than 0.
        shape : float, default = 0.0
            Shape parameter.

        Returns
        ----------
        q : np.ndarray
            Quantile value

        Raises
        ------
        ValueError
            If probabilities are not in the range (0, 1).

        ValueError
            If scale is not greater than 0.
        """

        if np.min(p) <= 0 or np.max(p) >= 1:
            raise ValueError("Probabilities must be in the range (0, 1)")

        if scale <= 0:
            raise ValueError("Scale parameter must be > 0")

        if poisson <= 0:
            raise ValueError("Poisson parameter must be > 0")

        # Gumbel case (shape = 0)
        if shape == 0.0:
            q = threshold - scale * np.log(-np.log(p) / poisson)

        # General case (Weibull and Frechet, shape != 0)
        else:
            q = threshold - (1 - (-np.log(p) / poisson) ** (-shape)) * scale / shape

        return q

    def cdf(
        x: np.ndarray,
        threshold: float = 0.0,
        scale: float = 1.0,
        shape: float = 0.0,
        poisson: float = 1.0,
    ) -> np.ndarray:
        if shape == 0.0:
            expr = (np.maximum(x - threshold, 0)) / scale
            return np.exp(-poisson * np.exp(-expr))

        else:
            expr = 1 + shape * (np.maximum(x - threshold, 0) / scale)
            return np.exp(-poisson * expr ** (-1 / shape))
