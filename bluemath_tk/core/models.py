import logging
from typing import Union, Tuple
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from .logging import get_file_logger
from .operations import normalize, denormalize, standarize, destandarize


class BlueMathModel(ABC):
    @abstractmethod
    def __init__(self):
        self._logger = get_file_logger(name=self.__class__.__name__)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @logger.setter
    def logger(self, value: logging.Logger) -> None:
        self._logger = value

    def set_logger_name(self, name: str):
        """Sets the name of the logger."""
        self.logger = get_file_logger(name=name)

    def save_model(self, model_path: str):
        """Saves the model to a file."""
        self.logger.info(f"Saving model to {model_path}")
        with open(model_path, "wb") as f:
            pickle.dump(self, f)

    def check_nans(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset],
        replace_value=None,
    ):
        """
        Checks for NaNs in the data and optionally replaces them.

        Parameters
        ----------
        data : np.ndarray, pd.Series, pd.DataFrame, xr.DataArray or xr.Dataset
            The data to check for NaNs.
        replace_value : any, optional
            The value to replace NaNs with. If None, NaNs will not be replaced.

        Returns
        -------
        data : np.ndarray, pd.Series, pd.DataFrame, xr.DataArray or xr.Dataset
            The data with NaNs optionally replaced.

        Notes
        -----
        - This method is intended to be used in classes that inherit from the BlueMathModel class.
        - The method checks for NaNs in the data and optionally replaces them with the specified value.
        """

        if isinstance(data, np.ndarray):
            if np.isnan(data).any():
                self.logger.warning("Data contains NaNs.")
                if replace_value is not None:
                    data = np.nan_to_num(data, nan=replace_value)
                    self.logger.info(f"NaNs replaced with {replace_value}.")
        elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
            if data.isnull().values.any():
                self.logger.warning("Data contains NaNs.")
                if replace_value is not None:
                    data.fillna(replace_value, inplace=True)
                    self.logger.info(f"NaNs replaced with {replace_value}.")
        elif isinstance(data, xr.DataArray) or isinstance(data, xr.Dataset):
            if data.isnull().any():
                self.logger.warning("Data contains NaNs.")
                if replace_value is not None:
                    data = data.fillna(replace_value)
                    self.logger.info(f"NaNs replaced with {replace_value}.")
        else:
            self.logger.warning("Data type not supported for NaN check.")
        return data

    def normalize(
        self, data: Union[pd.DataFrame, xr.Dataset], custom_scale_factor: dict = {}
    ) -> Tuple[Union[pd.DataFrame, xr.Dataset], dict]:
        """
        Normalize data to 0-1 using min max scaler approach.

        Parameters
        ----------
        data : pd.DataFrame or xr.Dataset
            The data to normalize.
        custom_scale_factor : dict, optional
            Custom scale factors for normalization.

        Returns
        -------
        normalized_data : pd.DataFrame or xr.Dataset
            The normalized data.
        scale_factor : dict
            The scale factors used for normalization.
        """

        normalized_data, scale_factor = normalize(
            data=data, custom_scale_factor=custom_scale_factor, logger=self.logger
        )
        return normalized_data, scale_factor

    def denormalize(
        self, normalized_data: pd.DataFrame, scale_factor: dict
    ) -> pd.DataFrame:
        """
        Denormalize data using provided scale_factor.

        Parameters
        ----------
        normalized_data : pd.DataFrame
            The normalized data to denormalize.
        scale_factor : dict
            The scale factors used for denormalization.

        Returns
        -------
        data : pd.DataFrame
            The denormalized data.
        """

        data = denormalize(normalized_data=normalized_data, scale_factor=scale_factor)
        return data

    def standarize(
        self,
        data: Union[np.ndarray, pd.DataFrame, xr.Dataset],
        scaler: StandardScaler = None,
    ) -> Tuple[Union[np.ndarray, pd.DataFrame, xr.Dataset], StandardScaler]:
        """
        Standarize data using StandardScaler.

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
        """

        standarized_data, scaler = standarize(data=data, scaler=scaler)
        return standarized_data, scaler

    def destandarize(
        self,
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
        data : np.ndarray, pd.DataFrame or xr.Dataset
            Destandarized data.
        """

        data = destandarize(standarized_data=standarized_data, scaler=scaler)
        return data
