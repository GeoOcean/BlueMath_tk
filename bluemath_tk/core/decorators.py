import functools
from typing import List
import pandas as pd


def validate_data_mda(func):
    """
    Decorator to validate data in MDA class fit method.

    It checks that the DataFrame is not None and that it is indeed a pandas DataFrame.
    If these conditions are not met, it raises a ValueError.
    Moreover, ensures that all directional variables have an associated custom scale factor.

    Parameters
    ----------
    func : callable
        The function to be decorated

    Returns
    -------
    callable
        The decorated function
    """

    @functools.wraps(func)
    def wrapper(
        self,
        data: pd.DataFrame,
        directional_variables: List[str],
        custom_scale_factor: dict,
    ):
        # NOTE: Default custom scale factors are defined below
        _default_custom_scale_factor = {"Dir": [0, 360]}
        if data is None:
            raise ValueError("Data cannot be None")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        if not isinstance(directional_variables, list):
            raise TypeError("Directional variables must be a list")
        if not isinstance(custom_scale_factor, dict):
            raise TypeError("Custom scale factor must be a dict")
        for directional_variable in directional_variables:
            if directional_variable not in custom_scale_factor:
                if directional_variable in _default_custom_scale_factor:
                    custom_scale_factor[directional_variable] = (
                        _default_custom_scale_factor[directional_variable]
                    )
                    self.logger.warning(
                        f"Using default custom scale factor for {directional_variable}"
                    )
                else:
                    raise KeyError(
                        "All directional variables must have an associated custom scale factor"
                    )
        return func(self, data, directional_variables, custom_scale_factor)

    return wrapper
