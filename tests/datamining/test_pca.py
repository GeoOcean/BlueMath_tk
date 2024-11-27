import unittest
import numpy as np
import xarray as xr
from bluemath_tk.datamining.pca import PCA


def get_2d_dataset():
    # Define the coordinates
    coord1 = np.linspace(-100, 100, 20)
    coord2 = np.linspace(-100, 100, 20)
    coord3 = np.arange(1, 50)

    # Create a meshgrid
    coord1, coord2, coord3 = np.meshgrid(coord1, coord2, coord3, indexing="ij")

    # Create a 3D dataset
    X = (
        np.sin(np.radians(coord1)) * np.cos(np.radians(coord2)) * np.sin(coord3)
        + np.sin(2 * np.radians(coord1))
        * np.cos(2 * np.radians(coord2))
        * np.sin(2 * coord3)
        + np.sin(3 * np.radians(coord1))
        * np.cos(3 * np.radians(coord2))
        * np.sin(3 * coord3)
    )
    # Create a 3D dataset
    Y = -np.sin(X)

    # Create an xarray dataset
    ds = xr.Dataset(
        {
            "X": (["coord1", "coord2", "coord3"], X),
            "Y": (["coord1", "coord2", "coord3"], Y),
        },
        coords={
            "coord1": coord1[:, 0, 0],
            "coord2": coord2[0, :, 0],
            "coord3": coord3[0, 0, :],
        },
    )

    return ds


class TestPCA(unittest.TestCase):
    def setUp(self):
        self.ds = get_2d_dataset()
        self.pca = PCA(n_components=5)
        self.ipca = PCA(n_components=5, is_incremental=True)

    def test_fit(self):
        self.pca.fit(
            data=self.ds,
            vars_to_stack=["X", "Y"],
            coords_to_stack=["coord1", "coord2"],
            pca_dim_for_rows="coord3",
        )
        self.assertEqual(self.pca.is_fitted, True)

    def test_transform(self):
        self.pca.fit(
            data=self.ds,
            vars_to_stack=["X", "Y"],
            coords_to_stack=["coord1", "coord2"],
            pca_dim_for_rows="coord3",
        )
        pcs = self.pca.transform(
            data=self.ds,
        )
        self.assertEqual(pcs.PCs.shape[1], 5)

    def test_fit_transform(self):
        pcs = self.pca.fit_transform(
            data=self.ds,
            vars_to_stack=["X", "Y"],
            coords_to_stack=["coord1", "coord2"],
            pca_dim_for_rows="coord3",
        )
        self.assertEqual(pcs.PCs.shape[1], 5)

    def test_inverse_transform(self):
        pcs = self.pca.fit_transform(
            data=self.ds,
            vars_to_stack=["X", "Y"],
            coords_to_stack=["coord1", "coord2"],
            pca_dim_for_rows="coord3",
        )
        reconstructed_ds = self.pca.inverse_transform(PCs=pcs)
        self.assertAlmostEqual(
            self.ds.isel(coord1=5, coord2=5, coord3=5),
            reconstructed_ds.isel(coord1=5, coord2=5, coord3=5),
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
