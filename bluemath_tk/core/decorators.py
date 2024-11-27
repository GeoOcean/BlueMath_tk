import functools
from typing import List
import pandas as pd
import xarray as xr


def validate_data_mda(func):
    """
    Decorator to validate data in MDA class fit method.

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
        directional_variables: List[str] = [],
        custom_scale_factor: dict = {},
    ):
        # NOTE: Default custom scale factors are defined below
        _default_custom_scale_factor = {}
        if data is None:
            raise ValueError("Data cannot be None")
        elif not isinstance(data, pd.DataFrame):
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
                    self.logger.warning(
                        f"No custom scale factor provided for {directional_variable}, min and max values will be used"
                    )
        return func(self, data, directional_variables, custom_scale_factor)

    return wrapper


def validate_data_lhs(func):
    """
    Decorator to validate data in LHS class fit method.

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
        dimensions_names: List[str],
        lower_bounds: List[float],
        upper_bounds: List[float],
        num_samples: int,
    ):
        if not isinstance(dimensions_names, list):
            raise TypeError("Dimensions names must be a list")
        if not isinstance(lower_bounds, list):
            raise TypeError("Lower bounds must be a list")
        if not isinstance(upper_bounds, list):
            raise TypeError("Upper bounds must be a list")
        if len(dimensions_names) != len(lower_bounds) or len(lower_bounds) != len(
            upper_bounds
        ):
            raise ValueError(
                "Dimensions names, lower bounds and upper bounds must have the same length"
            )
        if not all(
            [lower <= upper for lower, upper in zip(lower_bounds, upper_bounds)]
        ):
            raise ValueError("Lower bounds must be less than or equal to upper bounds")
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("Variable num_samples must be integer and > 0")
        return func(self, dimensions_names, lower_bounds, upper_bounds, num_samples)

    return wrapper


def validate_data_kma(func):
    """
    Decorator to validate data in KMA class fit method.

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
        _default_custom_scale_factor = {}
        if data is None:
            raise ValueError("Data cannot be None")
        elif not isinstance(data, pd.DataFrame):
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
                    self.logger.warning(
                        f"No custom scale factor provided for {directional_variable}, min and max values will be used"
                    )
        return func(self, data, directional_variables, custom_scale_factor)

    return wrapper


def validate_data_pca(func):
    """
    Decorator to validate data in PCA class fit method.

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
        data: xr.Dataset,
        vars_to_stack: List[str],
        coords_to_stack: List[str],
        pca_dim_for_rows: str,
        window_in_pca_dim_for_rows: List[int] = [0],
        value_to_replace_nans: float = None,
    ):
        if data is None:
            raise ValueError("Data cannot be None")
        elif not isinstance(data, xr.Dataset):
            raise TypeError("Data must be an xarray Dataset")
        # Check that all vars_to_stack are in the data
        if not isinstance(vars_to_stack, list) or len(vars_to_stack) == 0:
            raise ValueError("Variables to stack must be a non-empty list")
        for var in vars_to_stack:
            if var not in data.data_vars:
                raise ValueError(f"Variable {var} not found in data")
        # Check that all variables in vars_to_stack have the same coordinates and dimensions
        first_var = vars_to_stack[0]
        first_var_dims = list(data[first_var].dims)
        first_var_coords = list(data[first_var].coords)
        for var in vars_to_stack:
            if list(data[var].dims) != first_var_dims:
                raise ValueError(
                    f"All variables must have the same dimensions. Variable {var} does not match."
                )
            if list(data[var].coords) != first_var_coords:
                raise ValueError(
                    f"All variables must have the same coordinates. Variable {var} does not match."
                )
        # Check that all coords_to_stack are in the data
        if not isinstance(coords_to_stack, list) or len(coords_to_stack) == 0:
            raise ValueError("Coordinates to stack must be a non-empty list")
        for coord in coords_to_stack:
            if coord not in data.coords:
                raise ValueError(f"Coordinate {coord} not found in data.")
        # Check that pca_dim_for_rows is in the data, and window > 0 if provided
        if not isinstance(pca_dim_for_rows, str) or pca_dim_for_rows not in data.dims:
            raise ValueError(
                "PCA dimension for rows must be a string and found in the data dimensions"
            )
        if window_in_pca_dim_for_rows is not None:
            if (
                not isinstance(window_in_pca_dim_for_rows, list)
                or len(window_in_pca_dim_for_rows) == 0
            ):
                raise ValueError(
                    "Window in PCA dimension for rows must be a non-empty list"
                )
        if value_to_replace_nans is not None:
            if not isinstance(value_to_replace_nans, float):
                raise ValueError("Value to replace NaNs must be float")
        return func(
            self,
            data,
            vars_to_stack,
            coords_to_stack,
            pca_dim_for_rows,
            window_in_pca_dim_for_rows,
            value_to_replace_nans,
        )

    return wrapper


def validate_data_rbf(func):
    """
    Decorator to validate data in RBF class fit method.

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
        subset_data: pd.DataFrame,
        target_data: pd.DataFrame,
        subset_directional_variables: List[str] = [],
        target_directional_variables: List[str] = [],
        subset_custom_scale_factor: dict = {},
        normalize_target_data: bool = True,
        target_custom_scale_factor: dict = {},
    ):
        if subset_data is None:
            raise ValueError("Subset data cannot be None")
        elif not isinstance(subset_data, pd.DataFrame):
            raise TypeError("Subset data must be a pandas DataFrame")
        if target_data is None:
            raise ValueError("Target data cannot be None")
        elif not isinstance(target_data, pd.DataFrame):
            raise TypeError("Target data must be a pandas DataFrame")
        if not isinstance(subset_directional_variables, list):
            raise TypeError("Subset directional variables must be a list")
        for directional_variable in subset_directional_variables:
            if directional_variable not in subset_data.columns:
                raise ValueError(
                    f"Directional variable {directional_variable} not found in subset data"
                )
        if not isinstance(target_directional_variables, list):
            raise TypeError("Target directional variables must be a list")
        for directional_variable in target_directional_variables:
            if directional_variable not in target_data.columns:
                raise ValueError(
                    f"Directional variable {directional_variable} not found in target data"
                )
        if not isinstance(subset_custom_scale_factor, dict):
            raise TypeError("Subset custom scale factor must be a dict")
        if not isinstance(normalize_target_data, bool):
            raise TypeError("Normalize target data must be a bool")
        if not isinstance(target_custom_scale_factor, dict):
            raise TypeError("Target custom scale factor must be a dict")
        return func(
            self,
            subset_data,
            target_data,
            subset_directional_variables,
            target_directional_variables,
            subset_custom_scale_factor,
            normalize_target_data,
            target_custom_scale_factor,
        )

    return wrapper
