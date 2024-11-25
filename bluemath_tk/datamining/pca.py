import numpy as np
import xarray as xr
from typing import Union, List
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as PCA_, IncrementalPCA as IncrementalPCA_
from ._base_datamining import BaseReduction
from ..core.decorators import validate_data_pca
from ..core.plotting.base_plotting import DefaultStaticPlotting


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
    n_components : int or float
        The number of components or the explained variance ratio.
    is_incremental : bool
        Indicates whether Incremental PCA is used.
    _pca : PCA_ or IncrementalPCA_
        The PCA or Incremental PCA model.
    is_fitted : bool
        Indicates whether the PCA model has been fitted.
    _data : xr.Dataset
        The original dataset.
    _stacked_data_matrix : np.ndarray
        The stacked data matrix.
    _standarized_stacked_data_matrix : np.ndarray
        The standardized stacked data matrix.
    scaler : StandardScaler
        The scaler used for standardizing the data.
    vars_to_stack : list of str
        The list of variables to stack.
    coords_to_stack : list of str
        The list of coordinates to stack.
    pca_dim_for_rows : str
        The dimension for rows in PCA.
    window_in_pca_dim_for_rows : list of int
        The window in PCA dimension for rows.
    value_to_replace_nans : float
        The value to replace NaNs in the dataset.
    num_cols_for_vars : int
        The number of columns for variables.
    """

    def __init__(
        self, n_components: Union[int, float] = 0.98, is_incremental: bool = False
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

        Raises
        ------
        ValueError
            If n_components is less than or equal to 0.
        TypeError
            If n_components is not an integer when it is greater than or equal to 1.
        """

        super().__init__()
        self.set_logger_name(name=self.__class__.__name__)
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
            self.is_fitted: bool = False
        else:
            self.logger.info("Using PCA")
            self._pca = PCA_(n_components=self.n_components)
            self.is_fitted: bool = False
        self.is_incremental = is_incremental
        self._data: xr.Dataset = xr.Dataset()
        self._stacked_data_matrix: np.ndarray = np.array([])
        self._standarized_stacked_data_matrix: np.ndarray = np.array([])
        self.scaler: StandardScaler = StandardScaler()
        self.vars_to_stack: List[str] = []
        self.coords_to_stack: List[str] = []
        self.pca_dim_for_rows: str = None
        self.window_in_pca_dim_for_rows: List[int] = [0]
        self.value_to_replace_nans: float = None
        self.num_cols_for_vars: int = None

    @property
    def pca(self) -> Union[PCA_, IncrementalPCA_]:
        return self._pca

    @property
    def data(self) -> xr.Dataset:
        return self._data

    @property
    def stacked_data_matrix(self) -> np.ndarray:
        return self._stacked_data_matrix

    @property
    def standarized_stacked_data_matrix(self) -> np.ndarray:
        return self._standarized_stacked_data_matrix

    def _generate_stacked_data(self, data: xr.Dataset):
        """
        Generate stacked data matrix.

        Parameters
        ----------
        data : xr.Dataset
            The data to stack.

        Returns
        -------
        stacked_data_matrix : np.ndarray
            The stacked data matrix
        """

        self.logger.info(
            f"Generating data matrix with variables to stack: {self.vars_to_stack} and coordinates to stack: {self.coords_to_stack}"
        )
        num_cols_for_vars = 1
        for coord_to_stack in self.coords_to_stack:
            num_cols_for_vars *= len(data[coord_to_stack])
        tmp_stacked_data = data.stack(positions=self.coords_to_stack)
        if (
            len(self.window_in_pca_dim_for_rows) != 0
            or self.window_in_pca_dim_for_rows[0] != 0
        ):
            self.logger.info(f"Rolling over coordinate: {self.pca_dim_for_rows}")
            tmp_stacked_data = xr.concat(
                [
                    tmp_stacked_data.shift({self.pca_dim_for_rows: i})
                    for i in self.window_in_pca_dim_for_rows
                ],
                dim="positions",
            )
        self.num_cols_for_vars = num_cols_for_vars * len(
            self.window_in_pca_dim_for_rows
        )
        stacked_data_matrix = np.hstack(
            [tmp_stacked_data[var].values for var in self.vars_to_stack]
        )
        self.logger.info(
            f"Data matrix generated successfully with shape: {self._stacked_data_matrix.shape}"
        )

        return stacked_data_matrix

    def _preprocess_data(self, data: xr.Dataset, is_fit: bool = True):
        """
        Preprocess data for PCA.

        Parameters
        ----------
        data : xr.Dataset
            The data to preprocess.
        is_fit : bool, optional
            If True, set the data. Default is True.

        Returns
        -------
        standarized_stacked_data_matrix : np.ndarray
            The standarized stacked data matrix.
        """

        self.logger.info("Preprocessing data")
        data = self.check_nans(
            data=data,
            replace_value=self.value_to_replace_nans,
        )
        self.logger.info("Generating stacked data matrix")
        stacked_data_matrix = self._generate_stacked_data(
            data=data,
        )
        self.logger.info("Standarizing data matrix")
        standarized_stacked_data_matrix, scaler = self.standarize(
            data=stacked_data_matrix,
            scaler=self.scaler if not is_fit else None,
        )
        self.logger.info("Removing NaNs from standarized data matrix")
        standarized_stacked_data_matrix = self.check_nans(
            data=standarized_stacked_data_matrix,
            replace_value=self.value_to_replace_nans,
        )
        self.logger.info("Data preprocessed successfully")

        if is_fit:
            self._data = data.copy()
            self._stacked_data_matrix = stacked_data_matrix.copy()
            self._standarized_stacked_data_matrix = (
                standarized_stacked_data_matrix.copy()
            )
            self.scaler = scaler

        return standarized_stacked_data_matrix

    def _reshape_EOFs(self, destandarize: bool = False):
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
        EOFs_reshaped_vars_arrays = np.array_split(
            EOFs, len(self.vars_to_stack), axis=1
        )
        coords_to_stack_shape = [
            len(self.window_in_pca_dim_for_rows),
            self.pca.n_components_,
        ] + [self.data[coord].shape[0] for coord in self.coords_to_stack]
        EOFs_reshaped_vars_dict = {
            var: (
                ["window", "n_component", *self.coords_to_stack],
                np.array(
                    np.array_split(
                        EOF_reshaped_var, len(self.window_in_pca_dim_for_rows), axis=1
                    )
                ).reshape(*coords_to_stack_shape),
            )
            for var, EOF_reshaped_var in zip(
                self.vars_to_stack, EOFs_reshaped_vars_arrays
            )
        }
        return xr.Dataset(
            EOFs_reshaped_vars_dict,
            coords={
                "window": self.window_in_pca_dim_for_rows,
                "n_component": np.arange(self.pca.n_components_),
                **{coord: self.data[coord] for coord in self.coords_to_stack},
            },
        )

    def _reshape_data(self, X: np.ndarray, destandarize: bool = True):
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
        X_reshaped_vars_arrays = np.array_split(X, len(self.vars_to_stack), axis=1)
        coords_to_stack_shape = [
            len(self.window_in_pca_dim_for_rows),
            self.data[self.pca_dim_for_rows].shape[0],
        ] + [self.data[coord].shape[0] for coord in self.coords_to_stack]
        X_reshaped_vars_dict = {
            var: (
                ["window", self.pca_dim_for_rows, *self.coords_to_stack],
                np.array(
                    np.array_split(
                        X_reshaped_var, len(self.window_in_pca_dim_for_rows), axis=1
                    )
                ).reshape(*coords_to_stack_shape),
            )
            for var, X_reshaped_var in zip(self.vars_to_stack, X_reshaped_vars_arrays)
        }
        return xr.Dataset(
            X_reshaped_vars_dict,
            coords={
                "window": self.window_in_pca_dim_for_rows,
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
        window_in_pca_dim_for_rows: List[int] = [0],
        value_to_replace_nans: float = None,
    ):
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
        window_in_pca_dim_for_rows : list of int, optional
            The window steps to roll the pca_dim_for_rows. Default is [0].
        value_to_replace_nans : float, optional
            The value to replace NaNs. Default is None.
        """

        self.vars_to_stack = vars_to_stack
        self.coords_to_stack = coords_to_stack
        self.pca_dim_for_rows = pca_dim_for_rows
        self.window_in_pca_dim_for_rows = window_in_pca_dim_for_rows
        self.value_to_replace_nans = value_to_replace_nans

        self._preprocess_data(data=data[self.vars_to_stack], is_fit=True)
        self.logger.info("Fitting PCA model")
        self.pca.fit(X=self.standarized_stacked_data_matrix)
        self.is_fitted = True
        self.logger.info("PCA model fitted successfully")

    def transform(self, data: xr.Dataset):
        """
        Transform data using the fitted PCA model.

        Parameters
        ----------
        data : xr.Dataset
            The data to transform.

        Returns
        -------
        transformed_data : xr.Dataset
            The transformed data.
        """

        if self.is_fitted is False:
            raise PCAError("PCA model must be fitted before transforming data")
        self.logger.info("Transforming data using PCA model")
        processed_data = self._preprocess_data(
            data=data[self.vars_to_stack], is_fit=False
        )
        transformed_data = self.pca.transform(X=processed_data)
        return xr.Dataset(
            {
                "PCs": ((self.pca_dim_for_rows, "n_component"), transformed_data),
            },
            coords={
                self.pca_dim_for_rows: data[self.pca_dim_for_rows],
                "n_component": np.arange(self.pca.n_components_),
            },
        )

    def fit_transform(
        self,
        data: xr.Dataset,
        vars_to_stack: List[str],
        coords_to_stack: List[str],
        pca_dim_for_rows: str,
        window_in_pca_dim_for_rows: List[int] = [0],
        value_to_replace_nans: float = None,
    ):
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
        window_in_pca_dim_for_rows : list of int, optional
            The window steps to roll the pca_dim_for_rows. Default is [0].
        value_to_replace_nans : float, optional
            The value to replace NaNs. Default is None.

        Returns
        -------
        transformed_data : xr.Dataset
            The transformed data.
        """

        self.fit(
            data=data,
            vars_to_stack=vars_to_stack,
            coords_to_stack=coords_to_stack,
            pca_dim_for_rows=pca_dim_for_rows,
            window_in_pca_dim_for_rows=window_in_pca_dim_for_rows,
            value_to_replace_nans=value_to_replace_nans,
        )
        # TODO: JAVI - Add a flag to use the already processed data??
        return self.transform(data=data)

    def inverse_transform(self, PCs: Union[np.ndarray, xr.Dataset]):
        """
        Inverse transform data using the fitted PCA model.

        Parameters
        ----------
        X : np.ndarray or xr.Dataset
            The data to inverse transform.

        Returns
        -------
        data_transformed : xr.Dataset
            The inverse transformed data.
        """

        if self.is_fitted is False:
            raise PCAError("PCA model must be fitted before inverse transforming data")
        if isinstance(PCs, xr.Dataset):
            X = PCs["PCs"].values
        elif isinstance(PCs, np.ndarray):
            X = PCs
        self.logger.info("Inverse transforming data using PCA model")
        X_transformed = self.pca.inverse_transform(X=X)
        data_transformed = self._reshape_data(X=X_transformed, destandarize=True)
        return data_transformed

    def postprocess_data(self, data: xr.Dataset = None):
        """
        Postprocess data after PCA.

        Parameters
        ----------
        data : xr.Dataset, optional
            The data to postprocess. Default is None.
        """

        self.logger.info("Postprocessing data")
        # TODO: JAVI - Add postprocessing steps
        self.logger.info("Data postprocessed successfully")
