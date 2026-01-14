import unittest

import numpy as np
import pandas as pd

from bluemath_tk.datamining.kma import KMA, KMAError
from bluemath_tk.datamining.mda import MDA


class TestKMA(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)  # For reproducible tests
        self.df = pd.DataFrame(
            {
                "Hs": np.random.rand(1000) * 7,
                "Tp": np.random.rand(1000) * 20,
                "Dir": np.random.rand(1000) * 360,
            }
        )
        self.small_df = pd.DataFrame(
            {
                "Hs": np.random.rand(50) * 7,
                "Tp": np.random.rand(50) * 20,
                "Dir": np.random.rand(50) * 360,
            }
        )

    # ==================== Basic Functionality Tests ====================

    def test_init_basic(self):
        """Test basic initialization."""
        kma = KMA(num_clusters=5)
        self.assertEqual(kma.num_clusters, 5)
        self.assertIsNone(kma.seed)
        self.assertEqual(kma.algorithm_name, "kmeans")
        self.assertFalse(kma.is_fitted)

    def test_init_with_seed(self):
        """Test initialization with seed."""
        kma = KMA(num_clusters=5, seed=42)
        self.assertEqual(kma.seed, 42)

    def test_init_invalid_num_clusters(self):
        """Test initialization with invalid num_clusters."""
        with self.assertRaises(ValueError):
            KMA(num_clusters=0)
        with self.assertRaises(ValueError):
            KMA(num_clusters=-1)

    def test_init_invalid_seed(self):
        """Test initialization with invalid seed."""
        with self.assertRaises(ValueError):
            KMA(num_clusters=5, seed=-1)

    def test_init_invalid_algorithm(self):
        """Test initialization with invalid algorithm."""
        with self.assertRaises(ValueError):
            KMA(num_clusters=5, algorithm="invalid")

    # ==================== Algorithm Tests ====================

    def test_kmeans_algorithm(self):
        """Test kmeans algorithm."""
        kma = KMA(num_clusters=5, algorithm="kmeans")
        kma.fit(data=self.df)
        self.assertEqual(kma.algorithm_name, "kmeans")
        self.assertIsNotNone(kma._model)
        self.assertEqual(kma.centroids.shape[0], 5)
        self.assertTrue(kma.is_fitted)

    def test_kmedians_algorithm(self):
        """Test kmedians algorithm."""
        kma = KMA(num_clusters=5, algorithm="kmedians")
        kma.fit(data=self.df)
        self.assertEqual(kma.algorithm_name, "kmedians")
        self.assertIsNotNone(kma._model)
        self.assertEqual(kma.centroids.shape[0], 5)
        self.assertTrue(kma.is_fitted)

    def test_kmedoids_algorithm(self):
        """Test kmedoids algorithm."""
        kma = KMA(num_clusters=5, algorithm="kmedoids")
        kma.fit(data=self.df)
        self.assertEqual(kma.algorithm_name, "kmedoids")
        self.assertIsNotNone(kma._model)
        self.assertEqual(kma.centroids.shape[0], 5)
        self.assertTrue(kma.is_fitted)

    def test_algorithm_case_insensitive(self):
        """Test that algorithm names are case-insensitive."""
        kma1 = KMA(num_clusters=5, algorithm="KMEANS")
        kma2 = KMA(num_clusters=5, algorithm="kMeAnS")
        self.assertEqual(kma1.algorithm_name, "kmeans")
        self.assertEqual(kma2.algorithm_name, "kmeans")

    # ==================== Distance Matrix Tests ====================

    def test_distance_matrix_kmedoids(self):
        """Test distance matrix with kmedoids."""
        # Create a simple distance matrix
        n_samples = 100
        data = self.df.iloc[:n_samples]
        distances = np.random.rand(n_samples, n_samples)
        distances = (distances + distances.T) / 2  # Make symmetric
        np.fill_diagonal(distances, 0)  # Zero diagonal

        kma = KMA(num_clusters=5, algorithm="kmedoids", distance_matrix=distances)
        kma.fit(data=data)
        self.assertTrue(kma.is_fitted)
        self.assertEqual(kma.centroids.shape[0], 5)

    def test_distance_matrix_invalid_algorithm(self):
        """Test that distance matrix only works with kmedoids."""
        distances = np.random.rand(100, 100)
        distances = (distances + distances.T) / 2
        np.fill_diagonal(distances, 0)

        with self.assertRaises(ValueError):
            KMA(num_clusters=5, algorithm="kmeans", distance_matrix=distances)
        with self.assertRaises(ValueError):
            KMA(num_clusters=5, algorithm="kmedians", distance_matrix=distances)

    def test_distance_matrix_invalid_shape(self):
        """Test distance matrix validation."""
        # Non-square matrix
        distances = np.random.rand(100, 50)
        with self.assertRaises(ValueError):
            KMA(num_clusters=5, algorithm="kmedoids", distance_matrix=distances)

        # Non-2D matrix
        distances = np.random.rand(100)
        with self.assertRaises(ValueError):
            KMA(num_clusters=5, algorithm="kmedoids", distance_matrix=distances)

    # ==================== Fit Tests ====================

    def test_fit_basic(self):
        """Test basic fit."""
        kma = KMA(num_clusters=10)
        kma.fit(data=self.df)
        self.assertIsInstance(kma.centroids, pd.DataFrame)
        self.assertEqual(kma.centroids.shape[0], 10)
        self.assertTrue(kma.is_fitted)
        self.assertEqual(len(kma.centroid_real_indices), len(self.df))

    def test_fit_with_directional_variables(self):
        """Test fit with directional variables."""
        kma = KMA(num_clusters=5)
        kma.fit(data=self.df, directional_variables=["Dir"])
        self.assertTrue(kma.is_fitted)
        # Check that Dir_u and Dir_v are in centroids
        self.assertIn("Dir_u", kma.centroids.columns)
        self.assertIn("Dir_v", kma.centroids.columns)
        self.assertIn("Dir", kma.centroids.columns)

    def test_fit_with_custom_scale_factor(self):
        """Test fit with custom scale factor."""
        kma = KMA(num_clusters=5)
        custom_scale = {"Hs": [0, 10], "Tp": [0, 20]}
        kma.fit(
            data=self.df,
            custom_scale_factor=custom_scale,
            normalize_data=True,
        )
        self.assertTrue(kma.is_fitted)

    def test_fit_with_normalize_data(self):
        """Test fit with normalization."""
        kma = KMA(num_clusters=5)
        kma.fit(data=self.df, normalize_data=True)
        self.assertTrue(kma.is_fitted)
        # Check that normalized data exists
        self.assertFalse(kma.normalized_data.empty)

    def test_fit_without_normalize_data(self):
        """Test fit without normalization."""
        kma = KMA(num_clusters=5)
        kma.fit(data=self.df, normalize_data=False)
        self.assertTrue(kma.is_fitted)

    def test_fit_with_min_number_of_points(self):
        """Test fit with min_number_of_points constraint."""
        kma = KMA(num_clusters=5)
        kma.fit(data=self.df, min_number_of_points=50)
        self.assertTrue(kma.is_fitted)
        # Verify all clusters have at least min_number_of_points
        unique_labels, counts = np.unique(kma.centroid_real_indices, return_counts=True)
        self.assertTrue(np.all(counts >= 50))

    def test_fit_with_min_number_of_points_failure(self):
        """Test fit with impossible min_number_of_points."""
        kma = KMA(num_clusters=5)
        with self.assertRaises(ValueError):
            kma.fit(
                data=self.small_df,
                min_number_of_points=20,
                max_number_of_iterations=2,
            )

    def test_fit_with_max_iterations(self):
        """Test fit with max_iterations."""
        kma = KMA(num_clusters=5)
        kma.fit(
            data=self.df,
            min_number_of_points=50,
            max_number_of_iterations=20,
        )
        self.assertTrue(kma.is_fitted)

    def test_fit_with_regression_guided(self):
        """Test fit with regression-guided clustering."""
        data = self.df.copy()
        data["Fe"] = data["Hs"] ** 2 * data["Tp"]
        kma = KMA(num_clusters=5)
        kma.fit(
            data=data,
            regression_guided={"vars": ["Fe"], "alpha": [0.6]},
        )
        self.assertTrue(kma.is_fitted)
        # Check that Fe is in the fitting variables
        self.assertIn("Fe", kma.fitting_variables)

    def test_fit_with_regression_guided_multiple_vars(self):
        """Test fit with multiple regression-guided variables."""
        data = self.df.copy()
        data["Fe"] = data["Hs"] ** 2 * data["Tp"]
        data["Fe2"] = data["Hs"] * data["Tp"]
        kma = KMA(num_clusters=5)
        kma.fit(
            data=data,
            regression_guided={"vars": ["Fe", "Fe2"], "alpha": [0.4, 0.3]},
        )
        self.assertTrue(kma.is_fitted)
        self.assertIn("Fe", kma.fitting_variables)
        self.assertIn("Fe2", kma.fitting_variables)

    def test_fit_with_init_mda_centroids(self):
        """Test fit with MDA initialization."""
        mda = MDA(num_centers=5)
        mda.fit(data=self.df)
        init_centroids = mda.normalized_centroids.copy()
        kma = KMA(num_clusters=5)
        kma.fit(data=self.df, init_mda_centroids=init_centroids)
        self.assertTrue(kma.is_fitted)

    def test_fit_init_mda_centroids_invalid_shape(self):
        """Test fit with invalid MDA centroids shape."""
        kma = KMA(num_clusters=5)
        invalid_centroids = pd.DataFrame(
            np.random.rand(3, len(self.df.columns))
        )  # Wrong number of clusters
        with self.assertRaises(ValueError):
            kma.fit(data=self.df, init_mda_centroids=invalid_centroids)

    def test_fit_init_mda_centroids_invalid_columns(self):
        """Test fit with invalid MDA centroids columns."""
        kma = KMA(num_clusters=5)
        invalid_centroids = pd.DataFrame(
            np.random.rand(5, 2), columns=["Wrong", "Columns"]
        )
        with self.assertRaises(ValueError):
            kma.fit(data=self.df, init_mda_centroids=invalid_centroids)

    def test_fit_init_mda_centroids_invalid_type(self):
        """Test fit with invalid MDA centroids type."""
        kma = KMA(num_clusters=5)
        with self.assertRaises(ValueError):
            kma.fit(data=self.df, init_mda_centroids=np.array([[1, 2], [3, 4]]))

    # ==================== Predict Tests ====================

    def test_predict_basic(self):
        """Test basic prediction."""
        data_sample = pd.DataFrame(
            {
                "Hs": np.random.rand(15) * 7,
                "Tp": np.random.rand(15) * 20,
                "Dir": np.random.rand(15) * 360,
            }
        )
        kma = KMA(num_clusters=10)
        kma.fit(data=self.df)
        nearest_centroids, nearest_centroid_df = kma.predict(data=data_sample)
        self.assertIsInstance(nearest_centroids, pd.DataFrame)
        self.assertEqual(len(nearest_centroids), 15)
        self.assertIn("kma_bmus", nearest_centroids.columns)
        self.assertIsInstance(nearest_centroid_df, pd.DataFrame)
        self.assertEqual(nearest_centroid_df.shape[0], 15)

    def test_predict_with_directional_variables(self):
        """Test prediction with directional variables."""
        data_sample = pd.DataFrame(
            {
                "Hs": np.random.rand(15) * 7,
                "Tp": np.random.rand(15) * 20,
                "Dir": np.random.rand(15) * 360,
            }
        )
        kma = KMA(num_clusters=5)
        kma.fit(data=self.df, directional_variables=["Dir"])
        nearest_centroids, nearest_centroid_df = kma.predict(data=data_sample)
        self.assertEqual(len(nearest_centroids), 15)
        # Check that Dir is in the output
        self.assertIn("Dir", nearest_centroid_df.columns)

    def test_predict_not_fitted(self):
        """Test prediction without fitting."""
        kma = KMA(num_clusters=5)
        with self.assertRaises(KMAError):
            kma.predict(data=self.df)

    def test_predict_all_algorithms(self):
        """Test prediction for all algorithms."""
        data_sample = pd.DataFrame(
            {
                "Hs": np.random.rand(10) * 7,
                "Tp": np.random.rand(10) * 20,
                "Dir": np.random.rand(10) * 360,
            }
        )
        for algorithm in ["kmeans", "kmedians", "kmedoids"]:
            kma = KMA(num_clusters=5, algorithm=algorithm)
            kma.fit(data=self.df)
            nearest_centroids, nearest_centroid_df = kma.predict(data=data_sample)
            self.assertEqual(len(nearest_centroids), 10)
            self.assertEqual(nearest_centroid_df.shape[0], 10)

    # ==================== Fit-Predict Tests ====================

    def test_fit_predict_basic(self):
        """Test basic fit_predict."""
        kma = KMA(num_clusters=10)
        predicted_labels, predicted_labels_df = kma.fit_predict(data=self.df)
        self.assertIsInstance(predicted_labels, pd.DataFrame)
        self.assertEqual(len(predicted_labels), 1000)
        self.assertIsInstance(predicted_labels_df, pd.DataFrame)
        self.assertEqual(predicted_labels_df.shape[0], 1000)

    def test_fit_predict_with_min_number_of_points(self):
        """Test fit_predict with min_number_of_points."""
        kma = KMA(num_clusters=10)
        predicted_labels, predicted_labels_df = kma.fit_predict(
            data=self.df, min_number_of_points=50
        )
        _unique_labels, counts = np.unique(predicted_labels, return_counts=True)
        self.assertTrue(np.all(counts >= 50))
        self.assertEqual(len(predicted_labels), 1000)

    def test_fit_predict_with_all_options(self):
        """Test fit_predict with all options."""
        data = self.df.copy()
        data["Fe"] = data["Hs"] ** 2 * data["Tp"]
        kma = KMA(num_clusters=5)
        predicted_labels, predicted_labels_df = kma.fit_predict(
            data=data,
            directional_variables=["Dir"],
            custom_scale_factor={"Hs": [0, 10], "Tp": [0, 20]},
            normalize_data=True,
            regression_guided={"vars": ["Fe"], "alpha": [0.6]},
        )
        self.assertEqual(len(predicted_labels), 1000)
        self.assertEqual(predicted_labels_df.shape[0], 1000)

    # ==================== Property Tests ====================

    def test_properties(self):
        """Test class properties."""
        kma = KMA(num_clusters=5)
        kma.fit(data=self.df)
        # Test data property
        self.assertIsInstance(kma.data, pd.DataFrame)
        self.assertEqual(len(kma.data), 1000)
        # Test normalized_data property
        self.assertIsInstance(kma.normalized_data, pd.DataFrame)
        # Test data_to_fit property
        self.assertIsInstance(kma.data_to_fit, pd.DataFrame)
        # Test kma property (model)
        self.assertIsNotNone(kma.kma)

    # ==================== Regression Guided Tests ====================

    def test_add_regression_guided(self):
        """Test add_regression_guided static method."""
        data = self.df.copy()
        data["Fe"] = data["Hs"] ** 2 * data["Tp"]
        result = KMA.add_regression_guided(data=data, vars=["Fe"], alpha=[0.6])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("Fe", result.columns)
        self.assertIn("Hs", result.columns)
        self.assertIn("Tp", result.columns)

    def test_add_regression_guided_multiple_vars(self):
        """Test add_regression_guided with multiple variables."""
        data = self.df.copy()
        data["Fe"] = data["Hs"] ** 2 * data["Tp"]
        data["Fe2"] = data["Hs"] * data["Tp"]
        result = KMA.add_regression_guided(
            data=data, vars=["Fe", "Fe2"], alpha=[0.4, 0.3]
        )
        self.assertIn("Fe", result.columns)
        self.assertIn("Fe2", result.columns)

    # ==================== Seed Tests ====================

    def test_seed_reproducibility(self):
        """Test that seed produces reproducible results."""
        kma1 = KMA(num_clusters=5, seed=42)
        kma1.fit(data=self.df)
        kma2 = KMA(num_clusters=5, seed=42)
        kma2.fit(data=self.df)
        # Centroids should be the same with same seed
        np.testing.assert_array_almost_equal(
            kma1.centroids.values, kma2.centroids.values, decimal=5
        )

    def test_seed_none(self):
        """Test that seed=None produces different results."""
        kma1 = KMA(num_clusters=5, seed=None)
        kma1.fit(data=self.df)
        kma2 = KMA(num_clusters=5, seed=None)
        kma2.fit(data=self.df)
        # With seed=None, results may differ (though they might be the same by chance)

    # ==================== Algorithm-Specific Tests ====================

    def test_all_algorithms_produce_centroids(self):
        """Test that all algorithms produce centroids."""
        for algorithm in ["kmeans", "kmedians", "kmedoids"]:
            kma = KMA(num_clusters=5, algorithm=algorithm)
            kma.fit(data=self.df)
            self.assertEqual(kma.centroids.shape[0], 5)
            self.assertTrue(kma.is_fitted)

    def test_all_algorithms_with_directional_variables(self):
        """Test all algorithms with directional variables."""
        for algorithm in ["kmeans", "kmedians", "kmedoids"]:
            kma = KMA(num_clusters=5, algorithm=algorithm)
            kma.fit(data=self.df, directional_variables=["Dir"])
            self.assertIn("Dir", kma.centroids.columns)
            self.assertTrue(kma.is_fitted)

    def test_all_algorithms_with_normalization(self):
        """Test all algorithms with normalization."""
        for algorithm in ["kmeans", "kmedians", "kmedoids"]:
            kma = KMA(num_clusters=5, algorithm=algorithm)
            kma.fit(data=self.df, normalize_data=True)
            self.assertTrue(kma.is_fitted)

    # ==================== Edge Cases and Error Tests ====================

    def test_fit_empty_dataframe(self):
        """Test fit with empty dataframe."""
        kma = KMA(num_clusters=5)
        empty_df = pd.DataFrame()
        # This should raise an error from the decorator or base class
        with self.assertRaises((ValueError, TypeError)):
            kma.fit(data=empty_df)

    def test_fit_single_cluster(self):
        """Test fit with single cluster."""
        kma = KMA(num_clusters=1)
        kma.fit(data=self.df)
        self.assertEqual(kma.centroids.shape[0], 1)
        self.assertTrue(kma.is_fitted)

    def test_fit_many_clusters(self):
        """Test fit with many clusters."""
        kma = KMA(num_clusters=50)
        kma.fit(data=self.df)
        self.assertEqual(kma.centroids.shape[0], 50)
        self.assertTrue(kma.is_fitted)

    def test_predict_different_columns(self):
        """Test prediction with different columns (should fail)."""
        kma = KMA(num_clusters=5)
        kma.fit(data=self.df)
        different_df = pd.DataFrame({"X": np.random.rand(10), "Y": np.random.rand(10)})
        # This should raise an error
        with self.assertRaises((ValueError, KeyError)):
            kma.predict(data=different_df)

    # ==================== Combination Tests ====================

    def test_combination_seed_and_algorithm(self):
        """Test combination of seed and algorithm."""
        for algorithm in ["kmeans", "kmedians", "kmedoids"]:
            kma = KMA(num_clusters=5, seed=42, algorithm=algorithm)
            kma.fit(data=self.df)
            self.assertTrue(kma.is_fitted)

    def test_combination_directional_and_normalize(self):
        """Test combination of directional variables and normalization."""
        kma = KMA(num_clusters=5)
        kma.fit(
            data=self.df,
            directional_variables=["Dir"],
            normalize_data=True,
            custom_scale_factor={"Dir_u": [0, 1], "Dir_v": [0, 1]},
        )
        self.assertTrue(kma.is_fitted)
        self.assertIn("Dir", kma.centroids.columns)

    def test_combination_regression_and_directional(self):
        """Test combination of regression-guided and directional variables."""
        data = self.df.copy()
        data["Fe"] = data["Hs"] ** 2 * data["Tp"]
        kma = KMA(num_clusters=5)
        kma.fit(
            data=data,
            directional_variables=["Dir"],
            regression_guided={"vars": ["Fe"], "alpha": [0.6]},
        )
        self.assertTrue(kma.is_fitted)

    def test_combination_mda_init_and_min_points(self):
        """Test combination of MDA initialization and min points."""
        kma = KMA(num_clusters=5)
        kma.fit(data=self.df)
        init_centroids = kma.normalized_centroids.copy()

        kma2 = KMA(num_clusters=5)
        kma2.fit(
            data=self.df,
            init_mda_centroids=init_centroids,
            min_number_of_points=50,
        )
        self.assertTrue(kma2.is_fitted)

    def test_combination_all_options(self):
        """Test combination of all options together."""
        data = self.df.copy()
        data["Fe"] = data["Hs"] ** 2 * data["Tp"]
        kma = KMA(num_clusters=5, seed=42, algorithm="kmeans")
        kma.fit(
            data=data,
            directional_variables=["Dir"],
            custom_scale_factor={"Hs": [0, 10], "Tp": [0, 20]},
            normalize_data=True,
            regression_guided={"vars": ["Fe"], "alpha": [0.6]},
            min_number_of_points=50,
            max_number_of_iterations=20,
        )
        self.assertTrue(kma.is_fitted)
        # Verify min points constraint
        unique_labels, counts = np.unique(kma.centroid_real_indices, return_counts=True)
        self.assertTrue(np.all(counts >= 50))


if __name__ == "__main__":
    unittest.main()
