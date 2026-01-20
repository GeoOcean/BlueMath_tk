import os
import unittest

import numpy as np
import pandas as pd

from bluemath_tk.interpolation.gps import ExactGPInterpolation


def get_test_data_path(filename):
    """Help to get path to test data files."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(test_dir, "..", "data", "interpolation", filename)


class TestExactGPInterpolation(unittest.TestCase):
    """Test suite for ExactGPInterpolation using real data."""

    def setUp(self):
        """Set up test fixtures with real data."""
        predictor_path = get_test_data_path("predictor.csv")
        target_path = get_test_data_path("target.csv")

        self.subset_data = pd.read_csv(predictor_path, index_col=0).iloc[::10]
        self.target_data = pd.read_csv(target_path, index_col=0).iloc[::10]

    def test_fit(self):
        """Test fit with real data."""
        gp = ExactGPInterpolation(kernel="rbf+matern", epochs=50)
        gp.fit(
            subset_data=self.subset_data,
            subset_directional_variables=["wind_dir"],
            target_data=self.target_data,
            normalize_target_data=True,
            verbose=1,
        )

        self.assertTrue(gp.is_fitted)
        self.assertTrue(gp.is_target_normalized)
        self.assertIn("wind_dir_u", gp.normalized_subset_data.columns)
        self.assertIn("wind_dir_v", gp.normalized_subset_data.columns)
        self.assertIsNotNone(gp.hyperparameters)
        self.assertGreater(len(gp.hyperparameters), 0)

    def test_predict(self):
        """Test predict with real data."""
        gp = ExactGPInterpolation(kernel="rbf+matern", epochs=50)
        gp.fit(
            subset_data=self.subset_data,
            subset_directional_variables=["wind_dir"],
            target_data=self.target_data,
            normalize_target_data=True,
            verbose=1,
        )

        # Predict on full dataset
        predictions = gp.predict(dataset=self.subset_data, verbose=1)
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertEqual(len(predictions), len(self.subset_data))
        # Check that all target columns are present
        for col in self.target_data.columns:
            self.assertIn(col, predictions.columns)

    def test_predict_with_uncertainty(self):
        """Test predict with uncertainty quantification."""
        gp = ExactGPInterpolation(kernel="rbf+matern", epochs=50)
        gp.fit(
            subset_data=self.subset_data,
            subset_directional_variables=["wind_dir"],
            target_data=self.target_data,
            normalize_target_data=True,
            verbose=1,
        )

        # Predict on full dataset
        predictions = gp.predict(dataset=self.subset_data, return_std=True, verbose=1)
        self.assertIsInstance(predictions, pd.DataFrame)
        # Check that uncertainty columns are present
        for col in self.target_data.columns:
            self.assertIn(f"{col}_lower_ci", predictions.columns)
            self.assertIn(f"{col}_upper_ci", predictions.columns)
            # Check that lower CI < upper CI
            self.assertTrue(
                (predictions[f"{col}_lower_ci"] <= predictions[f"{col}_upper_ci"]).all()
            )

    def test_fit_predict(self):
        """Test fit_predict with real data."""
        gp = ExactGPInterpolation(kernel="rbf+matern", epochs=50)
        predictions = gp.fit_predict(
            subset_data=self.subset_data,
            subset_directional_variables=["wind_dir"],
            target_data=self.target_data,
            normalize_target_data=True,
            dataset=self.subset_data,
            verbose=1,
        )

        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertEqual(len(predictions), len(self.subset_data))
        # Check that all target columns are present
        for col in self.target_data.columns:
            self.assertIn(col, predictions.columns)

    def test_training_poins_have_zero_error(self):
        """Test that training points have zero error."""
        gp = ExactGPInterpolation(kernel="rbf+matern", epochs=50)
        training_predictions = gp.fit_predict(
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
        gp = ExactGPInterpolation(kernel="rbf+matern", epochs=50)
        gp.fit(
            subset_data=self.subset_data,
            subset_directional_variables=["wind_dir"],
            target_data=self.target_data,
            normalize_target_data=False,
            verbose=1,
        )

        self.assertTrue(gp.is_fitted)
        self.assertFalse(gp.is_target_normalized)

    def test_different_kernels(self):
        """Test real data with different kernel types."""
        # Test with rbf kernel
        gp_rbf = ExactGPInterpolation(kernel="rbf", epochs=50)
        gp_rbf.fit(
            subset_data=self.subset_data,
            subset_directional_variables=["wind_dir"],
            target_data=self.target_data,
            normalize_target_data=True,
            verbose=1,
        )
        self.assertTrue(gp_rbf.is_fitted)

        # Test with matern kernel
        gp_matern = ExactGPInterpolation(kernel="matern", epochs=50)
        gp_matern.fit(
            subset_data=self.subset_data,
            subset_directional_variables=["wind_dir"],
            target_data=self.target_data,
            normalize_target_data=True,
            verbose=1,
        )
        self.assertTrue(gp_matern.is_fitted)

        # Test with rbf+matern kernel (default)
        gp_combined = ExactGPInterpolation(kernel="rbf+matern", epochs=50)
        gp_combined.fit(
            subset_data=self.subset_data,
            subset_directional_variables=["wind_dir"],
            target_data=self.target_data,
            normalize_target_data=True,
            verbose=1,
        )
        self.assertTrue(gp_combined.is_fitted)

    def test_hyperparameters_extraction(self):
        """Test that hyperparameters are properly extracted."""
        gp = ExactGPInterpolation(kernel="rbf+matern", epochs=50)
        gp.fit(
            subset_data=self.subset_data,
            subset_directional_variables=["wind_dir"],
            target_data=self.target_data,
            normalize_target_data=True,
            verbose=1,
        )

        # Check hyperparameters structure
        hyperparams = gp.hyperparameters
        self.assertIsInstance(hyperparams, dict)
        self.assertGreater(len(hyperparams), 0)

        # Check that each target variable has hyperparameters
        for target_var in self.target_data.columns:
            self.assertIn(target_var, hyperparams)
            target_hyperparams = hyperparams[target_var]
            self.assertIsInstance(target_hyperparams, dict)
            # Check for expected keys
            self.assertIn("noise", target_hyperparams)
            self.assertIn("outputscale", target_hyperparams)

    def test_predictions_shape_and_type(self):
        """Test that predictions have correct shape and data types."""
        gp = ExactGPInterpolation(kernel="rbf+matern", epochs=50)
        gp.fit(
            subset_data=self.subset_data,
            subset_directional_variables=["wind_dir"],
            target_data=self.target_data,
            normalize_target_data=True,
            verbose=1,
        )

        predictions = gp.predict(dataset=self.subset_data, verbose=1)

        # Check shape
        self.assertEqual(predictions.shape[0], len(self.subset_data))
        self.assertEqual(predictions.shape[1], len(self.target_data.columns))

        # Check data types (should be numeric)
        for col in self.target_data.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(predictions[col]))

    def test_fit_predict_with_uncertainty(self):
        """Test fit_predict with uncertainty quantification."""
        gp = ExactGPInterpolation(kernel="rbf+matern", epochs=50)
        predictions = gp.fit_predict(
            subset_data=self.subset_data,
            subset_directional_variables=["wind_dir"],
            target_data=self.target_data,
            normalize_target_data=True,
            dataset=self.subset_data,
            return_std=True,
            verbose=1,
        )

        self.assertIsInstance(predictions, pd.DataFrame)
        # Check that uncertainty columns are present
        for col in self.target_data.columns:
            self.assertIn(f"{col}_lower_ci", predictions.columns)
            self.assertIn(f"{col}_upper_ci", predictions.columns)


if __name__ == "__main__":
    unittest.main()
