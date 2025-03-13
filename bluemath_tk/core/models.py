import importlib
import logging
import os
import pickle
import sys
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy import constants
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

from .logging import get_file_logger
from .operations import (
    denormalize,
    destandarize,
    get_degrees_from_uv,
    get_uv_components,
    normalize,
    standarize,
)

T = TypeVar("T")


class BlueMathModel(ABC):
    """
    Abstract base class for handling default functionalities across the project.
    """

    gravity = constants.g

    @abstractmethod
    def __init__(self) -> None:
        self._logger: logging.Logger = None
        self._exclude_attributes: List[str] = []
        self.num_workers: int = 1

        # [UNDER DEVELOPMENT] Below, we try to generalise parallel processing
        bluemath_num_workers = os.environ.get("BLUEMATH_NUM_WORKERS", None)
        omp_num_threads = os.environ.get("OMP_NUM_THREADS", None)
        if bluemath_num_workers is not None:
            self.logger.warning(
                f"Setting self.num_workers to {bluemath_num_workers} due to BLUEMATH_NUM_WORKERS. \n"
                "Change it using self.set_num_processors_to_use method. \n"
                "Also setting OMP_NUM_THREADS to 1, to avoid conflicts with BlueMath parallel processing."
            )
            self.set_num_processors_to_use(num_processors=int(bluemath_num_workers))
            self.set_omp_num_threads(num_threads=1)
        elif omp_num_threads is not None:
            self.logger.warning(
                f"Changing variable OMP_NUM_THREADS from {omp_num_threads} to 1. \n"
                f"And setting self.num_workers to {omp_num_threads}. \n"
                "To avoid conflicts with BlueMath parallel processing."
            )
            self.set_omp_num_threads(num_threads=1)
            self.set_num_processors_to_use(num_processors=int(omp_num_threads))
        else:
            self.num_workers = 1  # self.get_num_processors_available()
            self.logger.warning(
                f"Setting self.num_workers to {self.num_workers}. "
                "Change it using self.set_num_processors_to_use method."
            )

    def __getstate__(self):
        """Exclude certain attributes from being pickled."""

        state = self.__dict__.copy()
        for attr in self._exclude_attributes:
            if attr in state:
                del state[attr]
        # Iterate through the state attributes, warning about xr.Datasets
        for key, value in state.items():
            if isinstance(value, xr.Dataset) or isinstance(value, xr.DataArray):
                self.logger.warning(
                    f"Attribute {key} is an xarray Dataset / Dataarray and will be pickled!"
                )

        return state

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = get_file_logger(name=self.__class__.__name__)
        return self._logger

    @logger.setter
    def logger(self, value: logging.Logger) -> None:
        self._logger = value

    def set_logger_name(
        self, name: str, level: str = "INFO", console: bool = True
    ) -> None:
        """Sets the name of the logger."""

        self.logger = get_file_logger(name=name, level=level, console=console)

    def save_model(self, model_path: str, exclude_attributes: List[str] = None) -> None:
        """Saves the model to a file."""

        self.logger.info(f"Saving model to {model_path}")
        if exclude_attributes is not None:
            self._exclude_attributes += exclude_attributes
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
        replace_value: Union[float, callable] = None,
        raise_error: bool = False,
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset]:
        """
        Checks for NaNs in the data and optionally replaces them.

        Parameters
        ----------
        data : np.ndarray, pd.Series, pd.DataFrame, xr.DataArray or xr.Dataset
            The data to check for NaNs.
        replace_value : float or callable, optional
            The value to replace NaNs with. If None, NaNs will not be replaced.
            If a callable is provided, it will be called and the result will be returned.
            Default is None.
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

        # If replace_value is a callable, just call and return it
        if callable(replace_value):
            self.logger.debug(f"Replace value is a callable. Calling {replace_value}.")
            return replace_value(data)
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
        transform: bool = False,
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
        transform : bool
            Whether to just transform the data. Default to False.

        Returns
        -------
        standarized_data : np.ndarray, pd.DataFrame or xr.Dataset
            Standarized data.
        scaler : StandardScaler
            Scaler object used for standarization.
        """

        standarized_data, scaler = standarize(
            data=data, scaler=scaler, transform=transform
        )
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

    def set_omp_num_threads(self, num_threads: int) -> None:
        """
        Sets the number of threads for OpenMP.

        Parameters
        ----------
        num_threads : int
            The number of threads.

        Warning
        -----
        - This methos is under development.
        """

        os.environ["OMP_NUM_THREADS"] = str(num_threads)

        # Re-import numpy if it is already imported
        if "numpy" in sys.modules:
            importlib.reload(np)

    def get_num_processors_available(self) -> int:
        """
        Gets the number of processors available.

        Returns
        -------
        int
            The number of processors available.

        TODO
        ----
        - Check whether available processors are used or not.
        """

        return int(os.cpu_count() * 0.9)

    def set_num_processors_to_use(self, num_processors: int) -> None:
        """
        Sets the number of processors to use for parallel processing.

        Parameters
        ----------
        num_processors : int
            The number of processors to use.
            If -1, all available processors will be used.
        """

        # Retrieve the number of processors available
        num_processors_available = self.get_num_processors_available()

        # Check if the number of processors requested is valid
        if num_processors == -1:
            num_processors = num_processors_available
        elif num_processors <= 0:
            raise ValueError("Number of processors must be greater than 0")
        elif (num_processors_available - num_processors) < 2:
            self.logger.warning(
                "Number of processors requested is less than 2 processors available"
            )

        # Set the number of processors to use
        self.num_workers = num_processors

    def parallel_execute(
        self,
        func: Callable,
        items: List[Any],
        num_workers: int,
        cpu_intensive: bool = False,
        **kwargs,
    ) -> List[T]:
        """
        Execute a function in parallel using concurrent.futures.

        Parameters
        ----------
        func : Callable
            Function to execute for each item.
        items : List[Any]
            List of items to process.
        num_workers : int
            Number of parallel workers.
        cpu_intensive : bool, optional
            Whether the function is CPU intensive. Default is False.
        **kwargs : dict
            Additional keyword arguments for func.

        Returns
        -------
        List[T]
            List of results from each function call

        Warnings
        --------
        - cpu_intensive = True does not work with non-pickable objects (Under development).
        """

        results = {}

        executor_class = ProcessPoolExecutor if cpu_intensive else ThreadPoolExecutor
        self.logger.info(f"Using {executor_class.__name__} for parallel execution")

        with executor_class(max_workers=num_workers) as executor:
            future_to_item = {
                executor.submit(func, *item, **kwargs)
                if isinstance(item, tuple)
                else executor.submit(func, item, **kwargs): i
                for i, item in enumerate(items)
            }
            for future in as_completed(future_to_item):
                i = future_to_item[future]
                try:
                    result = future.result()
                    results[i] = result
                except Exception as exc:
                    self.logger.error(f"Job for {i} generated an exception: {exc}")

        return results
