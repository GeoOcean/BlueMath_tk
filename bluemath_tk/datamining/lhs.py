import pandas as pd
from typing import List
from scipy.stats import qmc
from ..core.models import BlueMathModel
from ..core.decorators import validate_data_lhs


class LHSError(Exception):
    """
    Custom exception for LHS class.
    """

    def __init__(self, message: str = "LHS error occurred."):
        self.message = message
        super().__init__(self.message)


class LHS(BlueMathModel):
    """
    Latin Hypercube Sampling (LHS) class.

    This class performs the LHS algorithm for some input data.

    Attributes
    ----------
    num_dimensions : int
        The number of dimensions to use in the LHS algorithm.
    num_samples : int
        The number of samples to generate.
    seed : int
        The random seed to use.
    lhs : qdc.LatinHypercube
        The Latin Hypercube object.

    Notes
    -----
    - This class is designed to perform the LHS algorithm.

    Examples
    --------
    >>> from bluemath_tk.datamining.lhs import LHS
    >>> dimensions_names = ['CM', 'SS', 'Qb']
    >>> lower_bounds = [0.5, -0.2, 1]
    >>> upper_bounds = [5.3, 1.5, 200]
    >>> lhs = LHS(num_dimensions=3, num_samples=100, seed=0)
    >>> lhs_sampled_df = lhs.fit(
    ...     dimensions_names=dimensions_names,
    ...     lower_bounds=lower_bounds,
    ...     upper_bounds=upper_bounds,
    ... )
    """

    def __init__(self, num_dimensions: int, num_samples: int, seed: int = 1) -> None:
        """
        Initializes the LHS class.

        Parameters
        ----------
        num_dimensions : int
            The number of dimensions to use in the LHS algorithm.
            Must be greater than 0.
        num_samples : int
            The number of samples to generate.
            Must be greater than 0.
        seed : int, optional
            The random seed to use.
            Must be greater or equal to 0.
            By default 1.

        Raises
        ------
        ValueError
            If num_dimensions or num_samples is not greater than 0.
            Or if seed is not greater or equal to 0.
        """
        super().__init__()
        self.set_logger_name(name=self.__class__.__name__)
        if num_dimensions > 0:
            self.num_dimensions = int(num_dimensions)
        else:
            raise ValueError("Variable num_dimensions must be > 0")
        if num_samples > 0:
            self.num_samples = int(num_samples)
        else:
            raise ValueError("Variable num_samples must be > 0")
        if seed >= 0:
            self.seed = int(seed)
        else:
            raise ValueError("Variable seed must be >= 0")
        self.lhs: qmc.LatinHypercube = None

    @validate_data_lhs
    def fit(
        self,
        dimensions_names: List[str],
        lower_bounds: List[float],
        upper_bounds: List[float],
    ):
        """
        Generate LHS samples.

        Parameters
        ----------
        dimensions_names : List[str]
            The names of the dimensions.
        lower_bounds : List[float]
            The lower bounds of the dimensions.
        upper_bounds : List[float]
            The upper bounds of the dimensions.

        Returns
        -------
        pd.DataFrame
            The LHS samples.
        """

        self.lhs = qmc.LatinHypercube(d=self.num_dimensions, seed=self.seed)
        lhs_samples = self.lhs.random(n=self.num_samples)
        lhs_scaled_data = qmc.scale(
            sample=lhs_samples, l_bounds=lower_bounds, u_bounds=upper_bounds
        )
        return pd.DataFrame(data=lhs_scaled_data, columns=dimensions_names)
