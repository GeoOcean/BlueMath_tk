import logging
from typing import Tuple
import pandas as pd


def normalize(
    data: pd.DataFrame, custom_scale_factor: dict = {}, logger: logging.Logger = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Normalize data to 0-1 using min max scaler approach
    """
    normalized_data = data.copy()  # Copy pd.DataFrame to avoid bad memory replacements
    scale_factor = (
        custom_scale_factor.copy()
    )  # Copy dict to avoid bad memory replacements
    for data_var in normalized_data.columns:
        data_var_min = normalized_data[data_var].min()
        data_var_max = normalized_data[data_var].max()
        if custom_scale_factor.get(data_var):
            if custom_scale_factor.get(data_var)[0] > data_var_min:
                if logger is not None:
                    logger.warning(
                        f"Proposed min custom scaler for {data_var} is bigger than datapoint, using smallest datapoint"
                    )
                scale_factor[data_var][0] = data_var_min
            else:
                data_var_min = custom_scale_factor.get(data_var)[0]
            if custom_scale_factor.get(data_var)[1] < data_var_max:
                if logger is not None:
                    logger.warning(
                        f"Proposed max custom scaler for {data_var} is lower than datapoint, using biggest datapoint"
                    )
                scale_factor[data_var][1] = data_var_max
            else:
                data_var_max = custom_scale_factor.get(data_var)[1]
        else:
            scale_factor[data_var] = [data_var_min, data_var_max]
        normalized_data[data_var] = (normalized_data[data_var] - data_var_min) / (
            data_var_max - data_var_min
        )
    return normalized_data, scale_factor


def denormalize(normalized_data: pd.DataFrame, scale_factor: dict) -> pd.DataFrame:
    """
    Denormalize data using provided scale_factor
    """
    data = normalized_data.copy()  # Copy pd.DataFrame to avoid bad memory replacements
    for data_var in data.columns:
        data[data_var] = (
            data[data_var] * (scale_factor[data_var][1] - scale_factor[data_var][0])
            + scale_factor[data_var][0]
        )
    return data
