import unittest
from bluemath_tk.core.data.sample_data import get_2d_dataset
from bluemath_tk.datamining.pca import PCA


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
