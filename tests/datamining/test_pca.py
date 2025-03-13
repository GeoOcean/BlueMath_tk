import unittest

import numpy as np

from bluemath_tk.core.data.sample_data import get_2d_dataset
from bluemath_tk.datamining.pca import PCA


def replace_nans_with_previous(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Replaces NaNs in a 2D NumPy array with the previous value in the same column.
    If the first value in a column is NaN, it is replaced with `fill_value`.

    Parameters
    ----------
    arr : np.ndarray
        Input 2D array with NaN values.
    fill_value : float, optional
        Value to replace leading NaNs. Default is 0.0.

    Returns
    -------
    np.ndarray
        Array with NaNs replaced.
    """

    arr = arr.copy()  # Avoid modifying the original array
    mask = np.isnan(arr)

    # Replace NaNs in the first row with a default fill_value
    arr[0, mask[0, :]] = fill_value

    # Replace NaNs with the previous row's value
    for i in range(1, arr.shape[0]):  # Iterate over rows
        arr[i, :] = np.where(mask[i, :], arr[i - 1, :], arr[i, :])

    return arr


class TestPCA(unittest.TestCase):
    def setUp(self):
        self.ds = get_2d_dataset()
        self.pca = PCA(n_components=5, debug=True)
        self.ipca = PCA(n_components=5, is_incremental=True)

    def test_fit_transform(self):
        pcs = self.pca.fit_transform(
            data=self.ds,
            vars_to_stack=["X", "Y"],
            coords_to_stack=["coord1", "coord2"],
            pca_dim_for_rows="coord3",
            windows_in_pca_dim_for_rows={"X": [3], "Y": [1]},
            value_to_replace_nans={
                "X": 0.0,
                "X_3": 1.0,
                "Y": replace_nans_with_previous,
            },
            nan_threshold_to_drop={"X": 0.5, "Y": 0.5},
        )
        self.assertEqual(self.pca.is_fitted, True)
        self.assertEqual(pcs.PCs.shape[1], 5)
        self.assertEqual(pcs.PCs.shape[0], self.ds.sizes["coord3"])
        self.assertCountEqual(self.pca.eofs.data_vars, ["X", "X_3", "Y", "Y_1"])

    def test_inverse_transform(self):
        pcs = self.pca.fit_transform(
            data=self.ds,
            vars_to_stack=["X", "Y"],
            coords_to_stack=["coord1", "coord2"],
            pca_dim_for_rows="coord3",
            scale_data=False,
        )
        reconstructed_ds = self.pca.inverse_transform(PCs=pcs.isel(coord3=slice(0, 5)))
        self.assertAlmostEqual(
            self.ds.isel(coord1=5, coord2=5, coord3=1),
            reconstructed_ds.isel(coord1=5, coord2=5, coord3=1),
        )

    def test_incremental_fit(self):
        self.ipca.fit(
            data=self.ds,
            vars_to_stack=["X", "Y"],
            coords_to_stack=["coord1", "coord2"],
            pca_dim_for_rows="coord3",
        )
        self.assertEqual(self.ipca.is_fitted, True)


if __name__ == "__main__":
    unittest.main()
