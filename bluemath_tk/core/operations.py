import logging
from typing import Union, Tuple
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import StandardScaler


def normalize(
    data: Union[pd.DataFrame, xr.Dataset],
    custom_scale_factor: dict = {},
    logger: logging.Logger = None,
) -> Tuple[Union[pd.DataFrame, xr.Dataset], dict]:
    """
    Normalize data to 0-1 using min max scaler approach.

    Parameters
    ----------
    data : pd.DataFrame or xr.Dataset
        Input data to be normalized.
    custom_scale_factor : dict, optional
        Dictionary with variables as keys and a list with two values as
        values. The first value is the minimum and the second value is the
        maximum used to normalize the variable. If not provided, the
        minimum and maximum values of the variable are used.
    logger : logging.Logger, optional
        Logger object to log warnings if the custom min or max is bigger or
        lower than the datapoints.

    Returns
    -------
    normalized_data : pd.DataFrame or xr.Dataset
        Normalized data.
    scale_factor : dict
        Dictionary with variables as keys and a list with two values as
        values. The first value is the minimum and the second value is the
        maximum used to normalize the variable.

    Notes
    -----
    - This method does not modify the input data, it creates a copy of the
      dataframe / dataset and normalizes it.
    - The normalization is done variable by variable, i.e. the minimum and
      maximum values are calculated for each variable.
    - If custom min or max is bigger or lower than the datapoints, it will
      be changed to the minimum or maximum of the datapoints and a warning
      will be logged.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from bluemath_tk.core.operations import normalize
    >>> df = pd.DataFrame(
    ...     {
    ...         "Hs": np.random.rand(1000) * 7,
    ...         "Tp": np.random.rand(1000) * 20,
    ...         "Dir": np.random.rand(1000) * 360,
    ...     }
    ... )
    >>> normalized_data, scale_factor = normalize(data=df)
    >>> import numpy as np
    >>> import xarray as xr
    >>> from bluemath_tk.core.data import normalize
    >>> ds = xr.Dataset(
    ...     {
    ...         "Hs": (("time",), np.random.rand(1000) * 7),
    ...         "Tp": (("time",), np.random.rand(1000) * 20),
    ...         "Dir": (("time",), np.random.rand(1000) * 360),
    ...     },
    ...     coords={"time": pd.date_range("2000-01-01", periods=1000)},
    ... )
    >>> normalized_data, scale_factor = normalize(data=ds)
    """

    if isinstance(data, pd.DataFrame):
        vars_to_normalize = list(data.columns)
    elif isinstance(data, xr.Dataset):
        vars_to_normalize = list(data.data_vars)
    else:
        raise TypeError("Data must be a pandas DataFrame or an xarray Dataset")
    normalized_data = data.copy()  # Copy data to avoid bad memory replacements
    scale_factor = (
        custom_scale_factor.copy()
    )  # Copy dict to avoid bad memory replacements
    for data_var in vars_to_normalize:
        data_var_min = normalized_data[data_var].min()
        data_var_max = normalized_data[data_var].max()
        if custom_scale_factor.get(data_var):
            if custom_scale_factor.get(data_var)[0] > data_var_min:
                if logger is not None:
                    logger.warning(
                        f"Proposed min custom scaler for {data_var} is bigger than datapoint"  # , using smallest datapoint
                    )
                else:
                    print(
                        f"Proposed min custom scaler for {data_var} is bigger than datapoint"  # , using smallest datapoint
                    )
            #     scale_factor[data_var][0] = data_var_min
            # else:
            data_var_min = custom_scale_factor.get(data_var)[0]
            if custom_scale_factor.get(data_var)[1] < data_var_max:
                if logger is not None:
                    logger.warning(
                        f"Proposed max custom scaler for {data_var} is lower than datapoint"  # , using biggest datapoint
                    )
                else:
                    print(
                        f"Proposed max custom scaler for {data_var} is lower than datapoint"  # , using biggest datapoint
                    )
            #     scale_factor[data_var][1] = data_var_max
            # else:
            data_var_max = custom_scale_factor.get(data_var)[1]
        else:
            scale_factor[data_var] = [data_var_min, data_var_max]
        normalized_data[data_var] = (normalized_data[data_var] - data_var_min) / (
            data_var_max - data_var_min
        )
    return normalized_data, scale_factor


def denormalize(
    normalized_data: Union[pd.DataFrame, xr.Dataset],
    scale_factor: dict,
) -> Union[pd.DataFrame, xr.Dataset]:
    """
    Denormalize data using provided scale_factor.

    Parameters
    ----------
    normalized_data : pd.DataFrame or xr.Dataset
        Input data that has been normalized and needs to be denormalized.
    scale_factor : dict
        Dictionary with variables as keys and a list with two values as
        values. The first value is the minimum and the second value is the
        maximum used to denormalize the variable.

    Returns
    -------
    data : pd.DataFrame or xr.Dataset
        Denormalized data.

    Notes
    -----
    - This method does not modify the input data, it creates a copy of the
      dataframe / dataset and denormalizes it.
    - The denormalization is done variable by variable, i.e. the minimum and
      maximum values are used to scale the data back to its original range.
    - Assumes that the scale_factor dictionary contains appropriate min and
      max values for each variable in the normalized_data.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from bluemath_tk.core.operation import denormalize
    >>> df = pd.DataFrame(
    ...     {
    ...         "Hs": np.random.rand(1000),
    ...         "Tp": np.random.rand(1000),
    ...         "Dir": np.random.rand(1000),
    ...     }
    ... )
    >>> scale_factor = {
    ...     "Hs": [0, 7],
    ...     "Tp": [0, 20],
    ...     "Dir": [0, 360],
    ... }
    >>> denormalized_data = denormalize(normalized_data=df, scale_factor=scale_factor)
    >>> import numpy as np
    >>> import xarray as xr
    >>> from bluemath_tk.core.operations import denormalize
    >>> ds = xr.Dataset(
    ...     {
    ...         "Hs": (("time",), np.random.rand(1000)),
    ...         "Tp": (("time",), np.random.rand(1000)),
    ...         "Dir": (("time",), np.random.rand(1000)),
    ...     },
    ...     coords={"time": pd.date_range("2000-01-01", periods=1000)},
    ... )
    >>> scale_factor = {
    ...     "Hs": [0, 7],
    ...     "Tp": [0, 20],
    ...     "Dir": [0, 360],
    ... }
    >>> denormalized_data = denormalize(normalized_data=ds, scale_factor=scale_factor)
    """

    if isinstance(normalized_data, pd.DataFrame):
        vars_to_denormalize = list(normalized_data.columns)
    elif isinstance(normalized_data, xr.Dataset):
        vars_to_denormalize = list(normalized_data.data_vars)
    else:
        raise TypeError("Data must be a pandas DataFrame or an xarray Dataset")
    data = normalized_data.copy()  # Copy data to avoid bad memory replacements
    for data_var in vars_to_denormalize:
        data[data_var] = (
            data[data_var] * (scale_factor[data_var][1] - scale_factor[data_var][0])
            + scale_factor[data_var][0]
        )
    return data


def standarize(
    data: Union[np.ndarray, pd.DataFrame, xr.Dataset],
    scaler: StandardScaler = None,
) -> Tuple[Union[np.ndarray, pd.DataFrame, xr.Dataset], StandardScaler]:
    """
    Standarize data to have mean 0 and variance 1

    Parameters
    ----------
    data : np.ndarray, pd.DataFrame or xr.Dataset
        Input data to be standarized.
    scaler : StandardScaler, optional
        Scaler object to use for standarization. Default is None.

    Returns
    -------
    standarized_data : np.ndarray, pd.DataFrame or xr.Dataset
        Standarized data.
    scaler : StandardScaler
        Scaler object used for standarization.

    Examples
    --------
    >>> import numpy as np
    >>> from bluemath_tk.core.operations import standarize
    >>> data = np.random.rand(1000, 3) * 10.0
    >>> standarized_data, scaler = standarize(data=data)
    """

    scaler = scaler or StandardScaler()
    if isinstance(data, np.ndarray):
        standarized_data = scaler.fit_transform(X=data)
        return standarized_data, scaler
    elif isinstance(data, pd.DataFrame):
        standarized_data = scaler.fit_transform(X=data.values)
        standarized_data = pd.DataFrame(standarized_data, columns=data.columns)
    elif isinstance(data, xr.Dataset):
        standarized_data = scaler.fit_transform(X=data.to_array().values)
        standarized_data = xr.Dataset(
            {
                var_name: (tuple(data.coords), standarized_data[i_var])
                for i_var, var_name in enumerate(data.data_vars)
            },
            coords=data.coords,
        )
    return standarized_data, scaler


def destandarize(
    standarized_data: Union[np.ndarray, pd.DataFrame, xr.Dataset],
    scaler: StandardScaler,
) -> Union[np.ndarray, pd.DataFrame, xr.Dataset]:
    """
    Destandarize data using provided scaler.

    Parameters
    ----------
    standarized_data : np.ndarray, pd.DataFrame or xr.Dataset
        Standarized data to be destandarized.
    scaler : StandardScaler
        Scaler object used for standarization.

    Returns
    -------
    np.ndarray, pd.DataFrame or xr.Dataset
        Destandarized data.

    Examples
    --------
    >>> import numpy as np
    >>> from bluemath_tk.core.data import standarize, destandarize
    >>> data = np.random.rand(1000, 3) * 10.0
    >>> standarized_data, scaler = standarize(data=data)
    >>> data = destandarize(standarized_data=standarized_data, scaler=scaler)
    """

    if isinstance(standarized_data, np.ndarray):
        data = scaler.inverse_transform(X=standarized_data)
    elif isinstance(standarized_data, pd.DataFrame):
        data = scaler.inverse_transform(X=standarized_data.values)
        data = pd.DataFrame(data, columns=standarized_data.columns)
    elif isinstance(standarized_data, xr.Dataset):
        data = scaler.inverse_transform(X=standarized_data.to_array().values)
        data = xr.Dataset(
            {
                var_name: (tuple(standarized_data.coords), data[i_var])
                for i_var, var_name in enumerate(standarized_data.data_vars)
            },
            coords=standarized_data.coords,
        )
    return data


def get_uv_components(x_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    This method calculates the u and v components for the given directional data.

    Here, we assume that the directional data is in degrees,
        beign 0° the North direction,
        and increasing clockwise.

                0° N
                |
                |
    270° W <---------> 90° E
                |
                |
                90° S

    Parameters
    ----------
    x_deg : np.ndarray
        The directional data in degrees.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The u and v components.
    """

    # Convert degrees to radians and adjust by subtracting from π/2
    x_rad = x_deg * np.pi / 180

    # Calculate x and y components using cosine and sine
    xu = np.sin(x_rad)
    xv = np.cos(x_rad)

    # Return the u and v components
    return xu, xv


def get_degrees_from_uv(xu: np.ndarray, xv: np.ndarray) -> np.ndarray:
    """
    This method calculates the degrees from the u and v components.

    Here, we assume u and v represent angles between 0 and 360 degrees,
        where 0° is the North direction,
        and increasing clockwise.

                 (u=0, v=1)
                     |
                     |
    (u=-1, v=0) <---------> (u=1, v=0)
                     |
                     |
                 (u=0, v=-1)

    Parameters
    ----------
    xu : np.ndarray
        The u component.
    xv : np.ndarray
        The v component.

    Returns
    -------
    np.ndarray
        The degrees.
    """

    # Calculate the degrees using the arctangent function
    x_deg = np.arctan2(xu, xv) * 180 / np.pi % 360

    # Return the degrees
    return x_deg
