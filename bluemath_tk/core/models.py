import os
import sys
import logging
from typing import Union, Tuple
import pickle
import importlib
import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    explained_variance_score,
)
from .logging import get_file_logger
from .operations import (
    normalize,
    denormalize,
    standarize,
    destandarize,
    get_uv_components,
    get_degrees_from_uv,
)


class BlueMathModel(ABC):
    """
    Abstract base class for handling default functionalities across the project.
    """

    @abstractmethod
    def __init__(self) -> None:
        self._logger: logging.Logger = None

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = get_file_logger(name=self.__class__.__name__)
        return self._logger

    @logger.setter
    def logger(self, value: logging.Logger) -> None:
        self._logger = value

    def set_logger_name(self, name: str, level: str = "INFO") -> None:
        """Sets the name of the logger."""

        self.logger = get_file_logger(name=name)
        self.logger.setLevel(level)

    def save_model(self, model_path: str) -> None:
        """Saves the model to a file."""

        self.logger.info(f"Saving model to {model_path}")
        with open(model_path, "wb") as f:
            pickle.dump(self, f)

    def load_model(self, model_path: str) -> "BlueMathModel":
        """Loads the model from a file."""

        self.logger.info(f"Loading model from {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        return model

    def list_class_attributes(self) -> list:
        """
        Lists the attributes of the class.

        Returns
        -------
        list
            The attributes of the class.
        """

        return [
            attr
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        ]

    def list_class_methods(self) -> list:
        """
        Lists the methods of the class.

        Returns
        -------
        list
            The methods of the class.
        """

        return [
            attr
            for attr in dir(self)
            if callable(getattr(self, attr)) and not attr.startswith("__")
        ]

    def check_nans(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset],
        replace_value=None,
        raise_error: bool = False,
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset]:
        """
        Checks for NaNs in the data and optionally replaces them.

        Parameters
        ----------
        data : np.ndarray, pd.Series, pd.DataFrame, xr.DataArray or xr.Dataset
            The data to check for NaNs.
        replace_value : any, optional
            The value to replace NaNs with. If None, NaNs will not be replaced.
        raise_error : bool, optional
            Whether to raise an error if NaNs are found. Default is False.

        Returns
        -------
        data : np.ndarray, pd.Series, pd.DataFrame, xr.DataArray or xr.Dataset
            The data with NaNs optionally replaced.

        Raises
        ------
        ValueError
            If NaNs are found and raise_error is True.

        Notes
        -----
        - This method is intended to be used in classes that inherit from the BlueMathModel class.
        - The method checks for NaNs in the data and optionally replaces them with the specified value.
        """

        if isinstance(data, np.ndarray):
            if np.isnan(data).any():
                if raise_error:
                    raise ValueError("Data contains NaNs.")
                self.logger.warning("Data contains NaNs.")
                if replace_value is not None:
                    data = np.nan_to_num(data, nan=replace_value)
                    self.logger.info(f"NaNs replaced with {replace_value}.")
        elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
            if data.isnull().values.any():
                if raise_error:
                    raise ValueError("Data contains NaNs.")
                self.logger.warning("Data contains NaNs.")
                if replace_value is not None:
                    data.fillna(replace_value, inplace=True)
                    self.logger.info(f"NaNs replaced with {replace_value}.")
        elif isinstance(data, xr.DataArray) or isinstance(data, xr.Dataset):
            if data.isnull().any():
                if raise_error:
                    raise ValueError("Data contains NaNs.")
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
        More info in bluemath_tk.core.operations.normalize.

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
        More info in bluemath_tk.core.operations.denormalize.

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
        More info in bluemath_tk.core.operations.standarize.

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
        More info in bluemath_tk.core.operations.destandarize.

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

    @staticmethod
    def get_metrics(
        data1: Union[pd.DataFrame, xr.Dataset],
        data2: Union[pd.DataFrame, xr.Dataset],
    ) -> pd.DataFrame:
        """
        Gets the metrics of the model.

        Parameters
        ----------
        data1 : pd.DataFrame or xr.Dataset
            The first dataset.
        data2 : pd.DataFrame or xr.Dataset
            The second dataset.

        Returns
        -------
        metrics : pd.DataFrame
            The metrics of the model.

        Raises
        ------
        ValueError
            If the DataFrames or Datasets have different shapes.
        TypeError
            If the inputs are not both DataFrames or both xarray Datasets.
        """

        if isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame):
            if data1.shape != data2.shape:
                raise ValueError("DataFrames must have the same shape")
            variables = data1.columns
        elif isinstance(data1, xr.Dataset) and isinstance(data2, xr.Dataset):
            if sorted(list(data1.dims)) != sorted(list(data2.dims)) or sorted(
                list(data1.data_vars)
            ) != sorted(list(data2.data_vars)):
                raise ValueError(
                    "Datasets must have the same dimensions, coordinates and variables"
                )
            variables = data1.data_vars
        else:
            raise TypeError(
                "Inputs must be either both DataFrames or both xarray Datasets"
            )

        metrics = {}
        for var in variables:
            if isinstance(data1, pd.DataFrame):
                y_true = data1[var]
                y_pred = data2[var]
            else:
                y_true = data1[var].values.reshape(-1)
                y_pred = data2[var].values.reshape(-1)

            metrics[var] = {
                "mean_squared_error": mean_squared_error(y_true, y_pred),
                "r2_score": r2_score(y_true, y_pred),
                "mean_absolute_error": mean_absolute_error(y_true, y_pred),
                "explained_variance_score": explained_variance_score(y_true, y_pred),
            }

        return pd.DataFrame(metrics).T

    @staticmethod
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

        return get_uv_components(x_deg)

    @staticmethod
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

        return get_degrees_from_uv(xu, xv)

    def get_num_processors_available(self) -> int:
        """
        Gets the number of processors available.

        Returns
        -------
        int
            The number of processors available.

        TODO:
        - Check whether available processors are used or not.
        """

        return os.cpu_count()

    def set_num_processors_to_use(self, num_processors: int) -> None:
        """
        Sets the number of processors to use for parallel processing.

        Parameters
        ----------
        num_processors : int
            The number of processors to use.
            If -1, all available processors will be used.

        Raises
        ------
        ValueError
            If the number of processors requested exceeds the number of processors available
        """

        # Retrieve the number of processors available
        num_processors_available = self.get_num_processors_available()

        # Check if the number of processors requested is valid
        if num_processors == -1:
            num_processors = num_processors_available
        elif num_processors <= 0:
            raise ValueError("Number of processors must be greater than 0")
        elif num_processors > num_processors_available:
            raise ValueError(
                f"Number of processors requested ({num_processors}) "
                f"exceeds the number of processors available ({num_processors_available})"
            )

        # Calculate the percentage of processors to use
        percentage = round(num_processors / num_processors_available, 2)
        if percentage < 0.5:
            self.logger.info(
                f"Number of processors requested ({num_processors}) "
                f"is less than 50% of the available processors ({num_processors_available})"
            )
        else:
            self.logger.warning(
                f"Number of processors requested ({num_processors}) "
                f"is more than 50% of the available processors ({num_processors_available})"
            )
        self.logger.info(f"Using {percentage * 100}% of the available processors")
        os.environ["OMP_NUM_THREADS"] = str(num_processors)

        # Re-import numpy if it is already imported
        if "numpy" in sys.modules:
            importlib.reload(np)

    def get_num_processors_used(self) -> int:
        """
        Gets the number of processors used.

        Returns
        -------
        int
            The number of processors used.

        Notes
        -----
        - This method returns the number of processors used by the application.
        - 1 is returned if the number of processors used is not set, as is the case of
          serial processing like Python's built-in functions.
        - Remember that if we run a parallel processing task, the number of processors used
          will be the number of processors set by the task, ehich can be > 1.
          Examples: np.linalg. or numerical models compiled with OpenMP or MPI.
        """

        return int(os.environ.get("OMP_NUM_THREADS", 1))
