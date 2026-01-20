"""Test suite for RBF interpolation using real data."""

import os
import unittest

import numpy as np
import pandas as pd

from bluemath_tk.interpolation.rbf import RBF


def get_test_data_path(filename):
    """Get path to test data files."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(test_dir, "..", "data", "interpolation", filename)


class TestRBF(unittest.TestCase):
    """Test suite for RBF interpolation using real data."""

    def setUp(self):
        """Set up test fixtures with real data."""
        predictor_path = get_test_data_path("predictor.csv")
        target_path = get_test_data_path("target.csv")

        self.subset_data = pd.read_csv(predictor_path, index_col=0).iloc[::50]
        self.target_data = pd.read_csv(target_path, index_col=0).iloc[::50]

    def test_fit(self):
        """Test fit with real data."""
        rbf = RBF()
        rbf.fit(
            subset_data=self.subset_data,
            subset_directional_variables=["wind_dir"],
            target_data=self.target_data,
            normalize_target_data=True,
            num_workers=4,
        )
        self.assertTrue(rbf.is_fitted)
        self.assertTrue(rbf.is_target_normalized)
        self.assertIn("wind_dir_u", rbf.normalized_subset_data.columns)
        self.assertIn("wind_dir_v", rbf.normalized_subset_data.columns)
        self.assertFalse(rbf.opt_sigmas == {})

    def test_predict(self):
        """Test predict with real data."""
        rbf = RBF()
        rbf.fit(
            subset_data=self.subset_data,
            subset_directional_variables=["wind_dir"],
            target_data=self.target_data,
            normalize_target_data=True,
        )
        predictions = rbf.predict(dataset=self.subset_data)
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertEqual(len(predictions), len(self.subset_data))
        # Check that all target columns are present
        for col in self.target_data.columns:
            self.assertIn(col, predictions.columns)

    def test_predict_with_uncertainty(self):
        """Test predict with uncertainty quantification."""
        rbf = RBF()
        rbf.fit(
            subset_data=self.subset_data,
            subset_directional_variables=["wind_dir"],
            target_data=self.target_data,
            normalize_target_data=True,
        )
        predictions = rbf.predict(dataset=self.subset_data, return_std=True)
        self.assertIsInstance(predictions, pd.DataFrame)
        # Check that uncertainty columns are present
        for col in self.target_data.columns:
            self.assertIn(f"{col}_std", predictions.columns)

    def test_fit_predict(self):
        """Test fit_predict with real data."""
        rbf = RBF()
        predictions = rbf.fit_predict(
            subset_data=self.subset_data,
            subset_directional_variables=["wind_dir"],
            target_data=self.target_data,
            normalize_target_data=True,
            dataset=self.subset_data,
            num_workers=4,
        )
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertEqual(len(predictions), len(self.subset_data))
        # Check that all target columns are present
        for col in self.target_data.columns:
            self.assertIn(col, predictions.columns)

    def test_training_points_have_zero_error(self):
        """Test that training points have zero error."""
        rbf = RBF()
        training_predictions = rbf.fit_predict(
            subset_data=self.subset_data,
            subset_directional_variables=["wind_dir"],
            target_data=self.target_data,
            dataset=self.subset_data,
        )
        for col in self.target_data.columns:
            self.assertLessEqual(
                np.abs(
                    self.target_data[col].values - training_predictions[col].values
                ).max(),
                2.5,
            )

    def test_without_normalization(self):
        """Test real data without target normalization."""
        rbf = RBF()
        rbf.fit(
            subset_data=self.subset_data,
            subset_directional_variables=["wind_dir"],
            target_data=self.target_data,
            normalize_target_data=False,
        )
        self.assertTrue(rbf.is_fitted)
        self.assertFalse(rbf.is_target_normalized)

    def test_different_kernels(self):
        """Test real data with different kernel types."""
        # Test with gaussian kernel (default)
        rbf_gaussian = RBF(kernel="gaussian")
        rbf_gaussian.fit(
            subset_data=self.subset_data,
            subset_directional_variables=["wind_dir"],
            target_data=self.target_data,
            normalize_target_data=True,
        )
        self.assertTrue(rbf_gaussian.is_fitted)

        # Test with thin_plate kernel (no sigma optimization)
        rbf_thin_plate = RBF(kernel="thin_plate")
        rbf_thin_plate.fit(
            subset_data=self.subset_data,
            subset_directional_variables=["wind_dir"],
            target_data=self.target_data,
            normalize_target_data=True,
        )
        self.assertTrue(rbf_thin_plate.is_fitted)

    def test_predictions_shape_and_type(self):
        """Test that predictions have correct shape and data types."""
        rbf = RBF()
        rbf.fit(
            subset_data=self.subset_data,
            subset_directional_variables=["wind_dir"],
            target_data=self.target_data,
            normalize_target_data=True,
        )
        predictions = rbf.predict(dataset=self.subset_data)

        # Check shape
        self.assertEqual(predictions.shape[0], len(self.subset_data))
        self.assertEqual(predictions.shape[1], len(self.target_data.columns))

        # Check data types (should be numeric)
        for col in self.target_data.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(predictions[col]))


if __name__ == "__main__":
    unittest.main()
