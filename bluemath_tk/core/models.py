import importlib
import logging
import os
import pickle
import sys
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Tuple, Union

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

from .constants import EARTH_RADIUS
from .logging import get_file_logger
from .operations import (
    denormalize,
    destandarize,
    get_degrees_from_uv,
    get_uv_components,
    normalize,
    standarize,
)


class BlueMathModel(ABC):
    """
    Abstract base class for handling default functionalities across the project.

    This class provides core functionality used by all BlueMath models including:
    - Model saving and loading
    - Data normalization and denormalization
    - Parallel processing capabilities
    - Logging functionality
    - NaN handling
    - Directional data processing

    Attributes
    ----------
    gravity : float
        Gravitational constant from scipy.constants.
    earth_radius : float
        Earth radius in km.
    num_workers : int
        Number of parallel workers to use for processing.
    logger : logging.Logger
        Logger instance for the model.

    Notes
    -----
    All BlueMath models should inherit from this class to ensure consistent
    behavior and functionality across the project.
    """

    gravity = constants.g
    earth_radius = EARTH_RADIUS

    @abstractmethod
    def __init__(self) -> None:
        self._logger: logging.Logger = None
        self._exclude_attributes: List[str] = []
        self.num_workers: int = 1

        # [UNDER DEVELOPMENT] Below, we try to generalise parallel processing
        bluemath_num_workers = os.environ.get("BLUEMATH_NUM_WORKERS", None)
        omp_num_threads = os.environ.get("OMP_NUM_THREADS", None)
        if bluemath_num_workers is not None:
            self.logger.info(
                f"Setting self.num_workers to {bluemath_num_workers} due to BLUEMATH_NUM_WORKERS. \n"
                "Change it using self.set_num_processors_to_use method. \n"
                "Also setting OMP_NUM_THREADS to 1, to avoid conflicts with BlueMath parallel processing."
            )
            self.set_num_processors_to_use(num_processors=int(bluemath_num_workers))
            self.set_omp_num_threads(num_threads=1)
        elif omp_num_threads is not None:
            self.logger.info(
                f"Changing variable OMP_NUM_THREADS from {omp_num_threads} to 1. \n"
                f"And setting self.num_workers to {omp_num_threads}. \n"
                "To avoid conflicts with BlueMath parallel processing."
            )
            self.set_omp_num_threads(num_threads=1)
            self.set_num_processors_to_use(num_processors=int(omp_num_threads))
        else:
            self.num_workers = 1  # self.get_num_processors_available()
            self.logger.info(
                f"Setting self.num_workers to {self.num_workers}. "
                "Change it using self.set_num_processors_to_use method."
            )

    def __getstate__(self):
        """
        Control which attributes are pickled when saving the model.

        This method is automatically called by pickle.dump() to determine what
        to serialize. It excludes specified attributes and warns about xarray objects.

        Returns
        -------
        dict
            A copy of the instance's __dict__ with excluded attributes removed.

        Notes
        -----
        - Controlled by self._exclude_attributes list
        - Warns when encountering xarray objects (Dataset/DataArray)
        - Creates a deep copy of state to avoid modifying original

        See Also
        --------
        save_model : High-level method for saving model to file
        """

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
        """
        Get the logger instance for this model.

        Returns
        -------
        logging.Logger
            The logger instance. Creates a new file logger if none exists.

        Notes
        -----
        - Lazily instantiates logger on first access
        - Uses class name as default logger name
        - Thread-safe logger creation
        """

        if self._logger is None:
            self._logger = get_file_logger(name=self.__class__.__name__)
        return self._logger

    @logger.setter
    def logger(self, value: logging.Logger) -> None:
        """
        Set the logger instance for this model.

        Parameters
        ----------
        value : logging.Logger
            The logger instance to use.

        Raises
        ------
        ValueError
            If the logger is not an instance of logging.Logger.
        """

        if not isinstance(value, logging.Logger):
            raise ValueError("Logger must be an instance of logging.Logger")
        self._logger = value

    def set_logger_name(
        self, name: str, level: str = "INFO", console: bool = True
    ) -> None:
        """
        Configure the model's logger with a new name and settings.

        Parameters
        ----------
        name : str
            The name to give to the logger.
        level : str, optional
            The logging level to use. Default is "INFO".
            Valid values are: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        console : bool, optional
            Whether to output logs to console. Default is True.

        Notes
        -----
        - Creates a new file logger with specified settings
        - Previous logger settings are overwritten
        - Log files are created in the default logging directory

        Examples
        --------
        >>> model = BlueMathModel()
        >>> model.set_logger_name("my_model", level="DEBUG", console=False)
        """

        self.logger = get_file_logger(name=name, level=level, console=console)

    def save_model(self, model_path: str, exclude_attributes: List[str] = None) -> None:
        """
        Save the model to a file using pickle.

        Parameters
        ----------
        model_path : str
            Path where the model will be saved.
        exclude_attributes : List[str], optional
            List of attribute names to exclude from saving. Default is None.
            If provided, it will override the default _exclude_attributes.

        Notes
        -----
        - Uses pickle for serialization
        - Warns if any xarray Datasets/DataArrays are being pickled
        - Creates parent directories if they don't exist
        - Excludes specified attributes from serialization

        Warnings
        --------
        - Pickle files can be security risks if loaded from untrusted sources
        - xarray objects in the model will be pickled and may be large

        Examples
        --------
        >>> model = MyBlueMathModel()
        >>> model.save_model('model.pkl', exclude_attributes=['_logger'])
        """

        self.logger.info(f"Saving model to {model_path}")
        if exclude_attributes is not None:
            self._exclude_attributes = exclude_attributes
        with open(model_path, "wb") as f:
            pickle.dump(self, f)

    def load_model(self, model_path: str) -> "BlueMathModel":
        """Loads the model from a file."""

        raise NotImplementedError(
            "This method is deprecated. Use load_model() from bluemath_tk.core.io instead."
        )

    def list_class_attributes(self) -> list:
        """
        List all non-callable attributes of the class.

        Returns
        -------
        list
            Names of all non-callable, non-private attributes.

        Notes
        -----
        - Excludes methods and private attributes (starting with __)
        - Includes properties and class variables
        - Useful for introspection and debugging

        Examples
        --------
        >>> model = BlueMathModel()
        >>> attrs = model.list_class_attributes()
        >>> print(attrs)
        ['gravity', 'num_workers', '_logger']
        """

        return [
            attr
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        ]

    def list_class_methods(self) -> list:
        """
        List all callable methods of the class.

        Returns
        -------
        list
            Names of all callable, non-private methods.

        Notes
        -----
        - Excludes attributes and private methods (starting with __)
        - Includes instance methods and properties
        - Useful for introspection and debugging

        Examples
        --------
        >>> model = BlueMathModel()
        >>> methods = model.list_class_methods()
        >>> print(methods)
        ['normalize', 'denormalize', 'check_nans']
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
        Check for NaNs in the data and optionally replace them.

        Parameters
        ----------
        data : Union[np.ndarray, pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset]
            The data to check for NaNs.
        replace_value : Union[float, callable], optional
            Value to replace NaNs with. If callable, the function will be called
            on the data. Default is None (no replacement).
        raise_error : bool, optional
            Whether to raise an error if NaNs are found. Default is False.

        Returns
        -------
        data : Union[np.ndarray, pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset]
            The data with NaNs optionally replaced.

        Raises
        ------
        ValueError
            If NaNs are found and raise_error is True.

        Notes
        -----
        - For numpy arrays, uses np.isnan() to check for NaNs
        - For pandas objects, uses isnull() to check for NaNs
        - For xarray objects, uses isnull() to check for NaNs
        - If replace_value is callable, it takes precedence over other options

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> model = BlueMathModel()
        >>> df = pd.DataFrame({'a': [1, np.nan, 3]})
        >>> cleaned_df = model.check_nans(df, replace_value=0)
        >>> print(cleaned_df)
           a
        0  1
        1  0
        2  3
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
        Set the number of OpenMP threads for parallel operations.

        Parameters
        ----------
        num_threads : int
            Number of OpenMP threads to use.

        Notes
        -----
        - Sets the OMP_NUM_THREADS environment variable
        - Reloads numpy to ensure new thread settings take effect
        - May affect other libraries using OpenMP

        Warnings
        --------
        - This method is under development and behavior may change
        - Reloading numpy may have side effects in running calculations

        See Also
        --------
        set_num_processors_to_use : Set number of processors for BlueMath parallel processing
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
        Set the number of processors to use for parallel processing.

        Parameters
        ----------
        num_processors : int
            Number of processors to use. If -1, uses all available processors
            minus one for system processes.

        Raises
        ------
        ValueError
            If num_processors is <= 0 (except -1).

        Notes
        -----
        - Automatically adjusts if requesting too many processors
        - Sets the num_workers attribute used by parallel processing methods
        - Takes into account system resources to avoid overload

        See Also
        --------
        get_num_processors_available : Get number of available processors
        parallel_execute : Execute functions in parallel
        """

        # Retrieve the number of processors available
        num_processors_available = self.get_num_processors_available()

        # Check if the number of processors requested is valid
        if num_processors == -1:
            num_processors = num_processors_available
        elif num_processors <= 0:
            raise ValueError("Number of processors must be greater than 0")
        elif (num_processors_available - num_processors) < 2:
            self.logger.info(
                "Number of processors requested leaves less than 2 processors available"
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
    ) -> Dict[int, Any]:
        """
        Execute a function in parallel across multiple items.

        Parameters
        ----------
        func : Callable
            The function to execute. Should accept single item and **kwargs.
        items : List[Any]
            List of items to process in parallel.
        num_workers : int
            Number of parallel workers to use.
        cpu_intensive : bool, optional
            If True, uses ProcessPoolExecutor, otherwise ThreadPoolExecutor.
            Default is False.
        **kwargs : dict
            Additional keyword arguments passed to func.

        Returns
        -------
        Dict[int, Any]
            Dictionary mapping item indices to function results.

        Raises
        ------
        Exception
            Any exception raised by func is logged and the job continues.

        Notes
        -----
        - Uses ThreadPoolExecutor for I/O-bound tasks
        - Uses ProcessPoolExecutor for CPU-bound tasks
        - Results maintain original item order via index mapping
        - Failed jobs are logged but don't stop execution

        Warnings
        --------
        - ThreadPoolExecutor may have GIL limitations
        - ProcessPoolExecutor doesn't work with non-picklable objects
        - File operations may fail with ThreadPoolExecutor

        Examples
        --------
        >>> def square(x):
        ...     return x * x
        >>> model = BlueMathModel()
        >>> results = model.parallel_execute(square, [1, 2, 3], num_workers=2)
        >>> print(results)
        {0: 1, 1: 4, 2: 9}
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
