from typing import List, Union

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA as PCA_
from sklearn.decomposition import IncrementalPCA as IncrementalPCA_
from sklearn.preprocessing import StandardScaler

from ..core.decorators import validate_data_pca
from ._base_datamining import BaseReduction


class PCAError(Exception):
    """
    Custom exception for PCA class.
    """

    def __init__(self, message: str = "PCA error occurred."):
        self.message = message
        super().__init__(self.message)


class PCA(BaseReduction):
    """
    Principal Component Analysis (PCA) class.

    Attributes
    ----------
    n_components : Union[int, float]
        The number of components or the explained variance ratio.
    is_incremental : bool
        Indicates whether Incremental PCA is used.
    _pca : Union[PCA_, IncrementalPCA_]
        The PCA or Incremental PCA model.
    is_fitted : bool
        Indicates whether the PCA model has been fitted.
    data : xr.Dataset
        The original dataset.
    window_processed_data : xr.Dataset
        The windows processed dataset.
    stacked_data_matrix : np.ndarray
        The stacked data matrix.
    standarized_stacked_data_matrix : np.ndarray
        The standardized stacked data matrix.
    scaler : StandardScaler
        The scaler used for standardizing the data.
    vars_to_stack : List[str]
        The list of variables to stack.
    window_stacked_vars : List[str]
        The list of variables with windows.
    coords_to_stack : List[str]
        The list of coordinates to stack.
    pca_dim_for_rows : str
        The dimension for rows in PCA.
    windows_in_pca_dim_for_rows : dict
        The windows in PCA dimension for rows.
    value_to_replace_nans : dict
        The values to replace NaNs in the dataset.
    nan_threshold_to_drop : dict
        The threshold percentage to drop NaNs for each variable.
    num_cols_for_vars : int
        The number of columns for variables.
    pcs : xr.Dataset
        The Principal Components (PCs).
    eofs : xr.Dataset
        The Empirical Orthogonal Functions (EOFs).
    explained_variance : np.ndarray
        The explained variance.
    explained_variance_ratio : np.ndarray
        The explained variance ratio.
    cumulative_explained_variance_ratio : np.ndarray
        The cumulative explained variance ratio.

    Methods
    -------
    _generate_stacked_data -> np.ndarray
        Generate stacked data matrix.
    _preprocess_data -> np.ndarray
        Preprocess data for PCA.
    _reshape_EOFs -> xr.Dataset
        Reshape EOFs to the original data shape.
    _reshape_data -> xr.Dataset
        Reshape data to the original data shape.
    fit -> None
        Fit PCA model to data.
    transform -> xr.Dataset
        Transform data using the fitted PCA model.
    fit_transform -> xr.Dataset
        Fit and transform data using PCA model.
    inverse_transform -> xr.Dataset
        Inverse transform data using the fitted PCA model.

    Examples
    --------
    >>> from bluemath_tk.core.data.sample_data import get_2d_dataset
    >>> from bluemath_tk.datamining.pca import PCA
    >>> ds = get_2d_dataset()
    >>> pca = PCA(
    ...     n_components=5,
    ...     is_incremental=False,
    ...     debug=True,
    ... )
    >>> pca.fit(
    ...     data=ds,
    ...     vars_to_stack=["X", "Y"],
    ...     coords_to_stack=["coord1", "coord2"],
    ...     pca_dim_for_rows="coord3",
    ...     windows_in_pca_dim_for_rows={"X": [1, 2, 3]},
    ...     value_to_replace_nans={"X": 0.0},
    ...     nan_threshold_to_drop={"X": 0.95},
    ... )
    >>> pcs = pca.transform(
    ...     data=ds,
    ... )
    >>> reconstructed_ds = pca.inverse_transform(PCs=pcs)
    >>> eofs = pca.eofs
    >>> explained_variance = pca.explained_variance
    >>> explained_variance_ratio = pca.explained_variance_ratio
    >>> cumulative_explained_variance_ratio = pca.cumulative_explained_variance_ratio
    >>> # Save the full class in a pickle file
    >>> pca.save_model("pca_model.pkl")

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

    [2] https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html

    [3] https://www.sciencedirect.com/science/article/abs/pii/S0378383911000676
    """

    def __init__(
        self,
        n_components: Union[int, float] = 0.98,
        is_incremental: bool = False,
        debug: bool = False,
    ):
        """
        Initialize the PCA class.

        Parameters
        ----------
        n_components : int or float, optional
            Number of components to keep. If 0 < n_components < 1, it represents the
            proportion of variance to be explained by the selected components. If
            n_components >= 1, it represents the number of components to keep. Default is 0.98.
        is_incremental : bool, optional
            If True, use Incremental PCA which is useful for large datasets. Default is False.
        debug : bool, optional
            If True, enable debug mode. Default is False.

        Raises
        ------
        ValueError
            If n_components is less than or equal to 0.
        TypeError
            If n_components is not an integer when it is greater than or equal to 1.
        """

        initial_msg = f"""
        -------------------------------------------------------------------
        | Initializing PCA reduction model with the following parameters:
        |    - n_components: {n_components}
        |    - is_incremental: {is_incremental}
        | For more information, please refer to the documentation.
        -------------------------------------------------------------------
        """
        print(initial_msg)

        super().__init__()
        self.set_logger_name(
            name=self.__class__.__name__, level="DEBUG" if debug else "INFO"
        )
        if n_components <= 0:
            raise ValueError("Number of components must be greater than 0.")
        elif n_components >= 1:
            if not isinstance(n_components, int):
                raise TypeError("Number of components must be an integer when >= 1.")
            self.logger.info(f"Number of components: {n_components}")
        else:
            self.logger.info(f"Explained variance ratio: {n_components}")
        self.n_components = n_components
        if is_incremental:
            self.logger.info("Using Incremental PCA")
            self._pca = IncrementalPCA_(n_components=self.n_components)
        else:
            self.logger.info("Using PCA")
            self._pca = PCA_(n_components=self.n_components)
        self.is_fitted: bool = False
        self.is_incremental = is_incremental
        self._data: xr.Dataset = xr.Dataset()
        self._window_processed_data: xr.Dataset = xr.Dataset()
        self._stacked_data_matrix: np.ndarray = np.array([])
        self._standarized_stacked_data_matrix: np.ndarray = np.array([])
        self.scaler: StandardScaler = StandardScaler()
        self.vars_to_stack: List[str] = []
        self.window_stacked_vars: List[str] = []
        self.coords_to_stack: List[str] = []
        self.pca_dim_for_rows: str = None
        self.windows_in_pca_dim_for_rows: dict = {}
        self.value_to_replace_nans: dict = {}
        self.nan_threshold_to_drop: dict = {}
        self.not_nan_positions: dict = {}
        self.num_cols_for_vars: int = None
        self.pcs: xr.Dataset = xr.Dataset()

    @property
    def pca(self) -> Union[PCA_, IncrementalPCA_]:
        return self._pca

    @property
    def data(self) -> xr.Dataset:
        return self._data

    @property
    def window_processed_data(self) -> xr.Dataset:
        return self._window_processed_data

    @property
    def stacked_data_matrix(self) -> np.ndarray:
        return self._stacked_data_matrix

    @property
    def standarized_stacked_data_matrix(self) -> np.ndarray:
        return self._standarized_stacked_data_matrix

    @property
    def eofs(self) -> xr.Dataset:
        return self._reshape_EOFs(destandarize=False)

    @property
    def explained_variance(self) -> np.ndarray:
        return self.pca.explained_variance_

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_

    @property
    def cumulative_explained_variance_ratio(self) -> np.ndarray:
        return (
            np.cumsum(self.explained_variance_ratio)
            / np.sum(self.explained_variance_ratio)
            * 100.0
        )

    def _generate_stacked_data(self, data: xr.Dataset) -> np.ndarray:
        """
        Generate stacked data matrix.

        Parameters
        ----------
        data : xr.Dataset
            The data to stack.

        Returns
        -------
        np.ndarray
            The stacked data matrix
        """

        self.logger.info(
            f"Generating data matrix with variables to stack: {self.vars_to_stack} and coordinates to stack: {self.coords_to_stack}"
        )

        self.num_cols_for_vars = 1
        for coord_to_stack in self.coords_to_stack:
            self.num_cols_for_vars *= len(data[coord_to_stack])
        tmp_stacked_data = data.stack(positions=self.coords_to_stack)

        cleaned_vars_to_stack = []
        for var_to_clean in self.window_stacked_vars:
            var_to_clean_values = tmp_stacked_data[var_to_clean].values
            # Drop variables with more than 90% of NaNs if not specified
            var_to_clean_threshold = self.nan_threshold_to_drop.get(
                var_to_clean,
                self.nan_threshold_to_drop.get(
                    var_to_clean[:-2],
                    self.nan_threshold_to_drop.get(var_to_clean[:-3], 0.90),
                ),
            )
            not_nan_positions = np.where(
                np.mean(~np.isnan(var_to_clean_values), axis=0) > var_to_clean_threshold
            )[0]
            # Replace NaNs with the value specified in value_to_replace_nans
            # If not specified, try to get the value from the variable name, deleting window suffixes
            var_value_to_replace_nans = self.value_to_replace_nans.get(
                var_to_clean,
                self.value_to_replace_nans.get(
                    var_to_clean[:-2], self.value_to_replace_nans.get(var_to_clean[:-3])
                ),
            )
            self.logger.debug(
                f"Replacing NaNs for variable: {var_to_clean} with value: {var_value_to_replace_nans}"
            )
            cleaned_var = self.check_nans(
                data=var_to_clean_values[:, not_nan_positions],
                replace_value=var_value_to_replace_nans,
            )
            cleaned_vars_to_stack.append(cleaned_var)
            self.not_nan_positions[var_to_clean] = not_nan_positions

        stacked_data_matrix = np.hstack(
            [cleaned_var for cleaned_var in cleaned_vars_to_stack]
        )
        self.logger.info(
            f"Data matrix generated successfully with shape: {stacked_data_matrix.shape}"
        )

        return stacked_data_matrix

    def _preprocess_data(self, data: xr.Dataset, is_fit: bool = True) -> np.ndarray:
        """
        Preprocess data for PCA. Steps:
        - Add windows in PCA dimension for rows.
        - Generate stacked data matrix.
        - Standarize data matrix.

        Parameters
        ----------
        data : xr.Dataset
            The data to preprocess.
        is_fit : bool, optional
            If True, set the data. Default is True.

        Returns
        -------
        np.ndarray
            The standarized stacked data matrix.
        """

        window_processed_data = data.copy()
        if self.windows_in_pca_dim_for_rows is not None:
            self.logger.info("Adding windows in PCA dimension for rows")
            for variable, windows in self.windows_in_pca_dim_for_rows.items():
                self.logger.info(f"Adding windows: {windows} for variable: {variable}")
                for window in windows:
                    window_processed_data[f"{variable}_{window}"] = (
                        window_processed_data[variable].shift(
                            {self.pca_dim_for_rows: window}
                        )
                    )
            self.window_stacked_vars = list(window_processed_data.data_vars)

        self.logger.info("Generating stacked data matrix")
        stacked_data_matrix = self._generate_stacked_data(
            data=window_processed_data,
        )

        self.logger.info("Standarizing data matrix")
        standarized_stacked_data_matrix, scaler = self.standarize(
            data=stacked_data_matrix,
            scaler=self.scaler if not is_fit else None,
        )

        self.logger.info("Data preprocessed successfully")

        if is_fit:
            self._data = data.copy()
            self._window_processed_data = window_processed_data.copy()
            self._stacked_data_matrix = stacked_data_matrix.copy()
            self._standarized_stacked_data_matrix = (
                standarized_stacked_data_matrix.copy()
            )
            self.scaler = scaler

        return standarized_stacked_data_matrix

    def _reshape_EOFs(self, destandarize: bool = False) -> xr.Dataset:
        """
        Reshape EOFs to the original data shape.

        Parameters
        ----------
        destandarize : bool, optional
            If True, destandarize the EOFs. Default is True.

        Returns
        -------
        xr.Dataset
            The reshaped EOFs.
        """

        EOFs = self.pca.components_  # Get Empirical Orthogonal Functions (EOFs)

        if destandarize:
            EOFs = self.scaler.inverse_transform(EOFs)

        # Create a full of nans array with shape time, vars * cols
        nan_EOFs = np.full(
            (
                self.pca.n_components_,
                self.num_cols_for_vars * len(self.window_stacked_vars),
            ),
            np.nan,
        )
        # Fill the nan_EOFs array with the EOFs values
        filled_EOFs = 0
        for iev, eof_var in enumerate(self.window_stacked_vars):
            eofs_to_fill = len(self.not_nan_positions[eof_var])
            nan_EOFs[
                :, self.not_nan_positions[eof_var] + (iev * self.num_cols_for_vars)
            ] = EOFs[:, filled_EOFs : filled_EOFs + eofs_to_fill]
            filled_EOFs += eofs_to_fill

        # Reshape the nan_EOFs array to the original data shape
        EOFs_reshaped_vars_arrays = np.array_split(
            nan_EOFs, len(self.window_stacked_vars), axis=1
        )
        coords_to_stack_shape = [self.pca.n_components_] + [
            self.data[coord].shape[0] for coord in self.coords_to_stack
        ]
        EOFs_reshaped_vars_dict = {
            var: (
                ["n_component", *self.coords_to_stack],
                np.array(EOF_reshaped_var).reshape(*coords_to_stack_shape),
            )
            for var, EOF_reshaped_var in zip(
                self.window_stacked_vars, EOFs_reshaped_vars_arrays
            )
        }

        return xr.Dataset(
            EOFs_reshaped_vars_dict,
            coords={
                "n_component": np.arange(self.pca.n_components_),
                **{coord: self.data[coord] for coord in self.coords_to_stack},
            },
        )

    def _reshape_data(self, X: np.ndarray, destandarize: bool = True) -> xr.Dataset:
        """
        Reshape data to the original data shape.

        Parameters
        ----------
        X : np.ndarray
            The data to reshape.
        destandarize : bool, optional
            If True, destandarize the data. Default is True.

        Returns
        -------
        xr.Dataset
            The reshaped data.
        """

        if destandarize:
            X = self.scaler.inverse_transform(X)

        # Create a full of nans array with shape time, vars * cols
        nan_X = np.full(
            (
                X.shape[0],
                self.num_cols_for_vars * len(self.window_stacked_vars),
            ),
            np.nan,
        )
        # Fill the nan_X array with the X values
        filled_X = 0
        for iev, x_var in enumerate(self.window_stacked_vars):
            x_to_fill = len(self.not_nan_positions[x_var])
            nan_X[:, self.not_nan_positions[x_var] + (iev * self.num_cols_for_vars)] = (
                X[:, filled_X : filled_X + x_to_fill]
            )
            filled_X += x_to_fill

        # Reshape the nan_X array to the original data shape
        X_reshaped_vars_arrays = np.array_split(
            nan_X, len(self.window_stacked_vars), axis=1
        )
        coords_to_stack_shape = [X.shape[0]] + [
            self.data[coord].shape[0] for coord in self.coords_to_stack
        ]
        X_reshaped_vars_dict = {
            var: (
                [self.pca_dim_for_rows, *self.coords_to_stack],
                np.array(X_reshaped_var).reshape(*coords_to_stack_shape),
            )
            for var, X_reshaped_var in zip(
                self.window_stacked_vars, X_reshaped_vars_arrays
            )
        }

        return xr.Dataset(
            X_reshaped_vars_dict,
            coords={
                self.pca_dim_for_rows: self.data[self.pca_dim_for_rows],
                **{coord: self.data[coord] for coord in self.coords_to_stack},
            },
        )

    @validate_data_pca
    def fit(
        self,
        data: xr.Dataset,
        vars_to_stack: List[str],
        coords_to_stack: List[str],
        pca_dim_for_rows: str,
        windows_in_pca_dim_for_rows: dict = {},
        value_to_replace_nans: dict = {},
        nan_threshold_to_drop: dict = {},
    ) -> None:
        """
        Fit PCA model to data.

        Parameters
        ----------
        data : xr.Dataset
            The data to fit the PCA model.
        vars_to_stack : list of str
            The variables to stack.
        coords_to_stack : list of str
            The coordinates to stack.
        pca_dim_for_rows : str
            The PCA dimension to maintain in rows (usually the time).
        windows_in_pca_dim_for_rows : dict, optional
            The window steps to roll the pca_dim_for_rows for each variable. Default is {}.
        value_to_replace_nans : dict, optional
            The value to replace NaNs for each variable. Default is {}.
        nan_threshold_to_drop : dict, optional
            The threshold percentage to drop NaNs for each variable.
            By default, variables with more than 90% of NaNs are dropped.
            Default is {}.

        Notes
        -----
        For both value_to_replace_nans and nan_threshold_to_drop, the keys are the variables,
        and the suffixes for the windows are considered.
        Example: if you have variable "X", and apply windows [1, 2, 3], you can use "X_1", "X_2", "X_3".
        Nevertheless, you can also use the original variable name "X" to apply the same value for all windows.
        """

        self.vars_to_stack = vars_to_stack.copy()
        self.coords_to_stack = coords_to_stack.copy()
        self.pca_dim_for_rows = pca_dim_for_rows
        self.windows_in_pca_dim_for_rows = windows_in_pca_dim_for_rows.copy()
        self.value_to_replace_nans = value_to_replace_nans.copy()
        self.nan_threshold_to_drop = nan_threshold_to_drop.copy()

        self._preprocess_data(data=data[self.vars_to_stack], is_fit=True)
        self.logger.info("Fitting PCA model")
        self.pca.fit(X=self.standarized_stacked_data_matrix)
        self.is_fitted = True
        self.logger.info("PCA model fitted successfully")

    def transform(self, data: xr.Dataset, after_fitting: bool = False) -> xr.Dataset:
        """
        Transform data using the fitted PCA model.

        Parameters
        ----------
        data : xr.Dataset
            The data to transform.
        after_fitting : bool, optional
            If True, use the already processed data. Default is False.

        Returns
        -------
        xr.Dataset
            The transformed data.
        """

        if self.is_fitted is False:
            raise PCAError("PCA model must be fitted before transforming data")

        if not after_fitting:
            self.logger.info("Transforming data using PCA model")
            processed_data = self._preprocess_data(
                data=data[self.vars_to_stack], is_fit=False
            )
        else:
            processed_data = self.standarized_stacked_data_matrix.copy()

        transformed_data = self.pca.transform(X=processed_data)

        # Save the Principal Components (PCs) in an xr.Dataset
        self.pcs = xr.Dataset(
            {
                "PCs": ((self.pca_dim_for_rows, "n_component"), transformed_data),
            },
            coords={
                self.pca_dim_for_rows: data[self.pca_dim_for_rows],
                "n_component": np.arange(self.pca.n_components_),
            },
        )

        return self.pcs.copy()

    def fit_transform(
        self,
        data: xr.Dataset,
        vars_to_stack: List[str],
        coords_to_stack: List[str],
        pca_dim_for_rows: str,
        windows_in_pca_dim_for_rows: dict = {},
        value_to_replace_nans: dict = {},
        nan_threshold_to_drop: dict = {},
    ) -> xr.Dataset:
        """
        Fit and transform data using PCA model.

        Parameters
        ----------
        data : xr.Dataset
            The data to fit the PCA model.
        vars_to_stack : list of str
            The variables to stack.
        coords_to_stack : list of str
            The coordinates to stack.
        pca_dim_for_rows : str
            The PCA dimension to maintain in rows (usually the time).
        windows_in_pca_dim_for_rows : dict, optional
            The window steps to roll the pca_dim_for_rows for each variable. Default is {}.
        value_to_replace_nans : dict, optional
            The value to replace NaNs for each variable. Default is {}.
        nan_threshold_to_drop : dict, optional
            The threshold percentage to drop NaNs for each variable.
            By default, variables with more than 90% of NaNs are dropped.
            Default is {}.

        Returns
        -------
        xr.Dataset
            The transformed data representing the Principal Components (PCs).

        Notes
        -----
        For both value_to_replace_nans and nan_threshold_to_drop, the keys are the variables,
        and the suffixes for the windows are considered.
        Example: if you have variable "X", and apply windows [1, 2, 3], you can use "X_1", "X_2", "X_3".
        Nevertheless, you can also use the original variable name "X" to apply the same value for all windows.
        """

        self.fit(
            data=data,
            vars_to_stack=vars_to_stack,
            coords_to_stack=coords_to_stack,
            pca_dim_for_rows=pca_dim_for_rows,
            windows_in_pca_dim_for_rows=windows_in_pca_dim_for_rows,
            value_to_replace_nans=value_to_replace_nans,
            nan_threshold_to_drop=nan_threshold_to_drop,
        )

        return self.transform(data=data, after_fitting=True)

    def inverse_transform(self, PCs: Union[np.ndarray, xr.Dataset]) -> xr.Dataset:
        """
        Inverse transform data using the fitted PCA model.

        Parameters
        ----------
        X : np.ndarray or xr.Dataset
            The data to inverse transform.

        Returns
        -------
        xr.Dataset
            The inverse transformed data.
        """

        if self.is_fitted is False:
            raise PCAError("PCA model must be fitted before inverse transforming data")

        if isinstance(PCs, xr.Dataset):
            X = PCs["PCs"].values
        elif isinstance(PCs, xr.DataArray):
            X = PCs.values
        elif isinstance(PCs, np.ndarray):
            X = PCs

        self.logger.info("Inverse transforming data using PCA model")
        X_transformed = self.pca.inverse_transform(X=X)
        data_transformed = self._reshape_data(X=X_transformed, destandarize=True)
        # Transpose dimensions based on the original data
        data_transformed = data_transformed.transpose(*self.data.dims)

        return data_transformed
