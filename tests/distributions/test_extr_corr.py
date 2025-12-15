import os
import shutil
import tempfile
import unittest

import numpy as np
import xarray as xr

from bluemath_tk.distributions.extreme_correction import ExtremeCorrection
from bluemath_tk.distributions.gev import GEV


class TestExtremeCorrection(unittest.TestCase):
    """Test suite for ExtremeCorrection class"""

    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)

        # Create temporary directory for outputs
        self.test_dir = tempfile.mkdtemp()

        # Generate synthetic historical data (10 years, daily)
        n_years_hist = 10
        n_days = 365
        time_hist = xr.date_range(
            start="2000-01-01", periods=n_years_hist * n_days, freq="D", use_cftime=True
        )

        # Generate GEV distributed data
        self.loc_true = 5.0
        self.scale_true = 2.0
        self.shape_true = 0.1

        hist_values = GEV.random(
            size=len(time_hist),
            loc=self.loc_true,
            scale=self.scale_true,
            shape=self.shape_true,
            random_state=42,
        )

        self.data_hist = xr.Dataset(
            {
                "hs": (["time"], hist_values),
            },
            coords={"time": time_hist},
        )

        # Generate synthetic simulated data (5 simulations, 20 years each)
        n_years_sim = 20
        n_sims = 5
        time_sim = xr.date_range(
            start="2010-01-01", periods=n_years_sim * n_days, freq="D", use_cftime=True
        )

        sim_values = np.array(
            [
                GEV.random(
                    size=len(time_sim),
                    loc=self.loc_true * 0.8,  # Biased synthetic data
                    scale=self.scale_true * 0.9,
                    shape=self.shape_true * 1.2,
                    random_state=i,
                )
                for i in range(n_sims)
            ]
        )

        self.data_sim = xr.Dataset(
            {
                "hs": (["n_sim", "time"], sim_values),
            },
            coords={"n_sim": np.arange(n_sims), "time": time_sim},
        )

        # Basic configuration
        self.corr_config = {
            "var": "hs",
            "time_var": "time",
            "yyyy_var": "time.year",
            "freq": 365.25,
            "folder": self.test_dir,
        }

        self.pot_config = {
            "n0": 10,
            "min_peak_distance": 2,
            "init_threshold": 5.0,
            "siglevel": 0.05,
            "plot": False,
        }

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_initialization_am(self):
        """Test initialization with Annual Maxima method"""
        ec = ExtremeCorrection(
            corr_config=self.corr_config,
            pot_config=self.pot_config,
            method="am",
            conf_level=0.95,
        )

        self.assertEqual(ec.method, "am")
        self.assertEqual(ec.conf, 0.95)
        self.assertEqual(ec.var, "hs")
        self.assertTrue(hasattr(ec, "parameters"))

    def test_initialization_pot(self):
        """Test initialization with POT method"""
        ec = ExtremeCorrection(
            corr_config=self.corr_config,
            pot_config=self.pot_config,
            method="pot",
            conf_level=0.90,
        )

        self.assertEqual(ec.method, "pot")
        self.assertEqual(ec.conf, 0.90)
        self.assertTrue(hasattr(ec, "pot_config"))
        self.assertEqual(ec.pot_config["n0"], 10)

    def test_config_validation_missing_key(self):
        """Test that missing required config keys raise KeyError"""
        invalid_config = {}  # Missing required keys

        with self.assertRaises(KeyError):
            ExtremeCorrection(
                corr_config=invalid_config, pot_config=self.pot_config, method="am"
            )

    def test_config_validation_wrong_type(self):
        """Test that wrong config types raise TypeError"""
        invalid_config = self.corr_config.copy()
        invalid_config["var"] = 123  # Should be string

        with self.assertRaises(TypeError):
            ExtremeCorrection(
                corr_config=invalid_config, pot_config=self.pot_config, method="am"
            )

    def test_fit_am_method(self):
        """Test fitting with Annual Maxima method"""
        ec = ExtremeCorrection(
            corr_config=self.corr_config, pot_config=self.pot_config, method="am"
        )

        ec.fit(data_hist=self.data_hist, plot_diagnostic=False)

        # Check that parameters were fitted
        self.assertEqual(len(ec.parameters), 3)
        self.assertIsNotNone(ec.am_data)
        self.assertIsNotNone(ec.pit_data)
        self.assertEqual(ec.poiss_parameter, 1)  # GEV has lambda=1

    def test_fit_pot_method(self):
        """Test fitting with POT method"""
        ec = ExtremeCorrection(
            corr_config=self.corr_config, pot_config=self.pot_config, method="pot"
        )

        ec.fit(data_hist=self.data_hist, plot_diagnostic=False)

        # Check that parameters were fitted
        self.assertEqual(len(ec.parameters), 3)
        self.assertIsNotNone(ec.threshold)
        self.assertIsNotNone(ec.pot_data)
        self.assertGreater(ec.poiss_parameter, 0)

    def test_transform(self):
        """Test transform method"""
        ec = ExtremeCorrection(
            corr_config=self.corr_config, pot_config=self.pot_config, method="am"
        )

        ec.fit(data_hist=self.data_hist)
        result = ec.transform(data_sim=self.data_sim, prob="unif", random_state=42)

        # Check output structure
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("hs_corr", result.variables)

        # Check that corrected data exists
        self.assertIsNotNone(ec.sim_pit_data_corrected)
        self.assertIsNotNone(ec.sim_am_data_corr)

    def test_fit_transform(self):
        """Test fit_transform method"""
        ec = ExtremeCorrection(
            corr_config=self.corr_config, pot_config=self.pot_config, method="am"
        )

        result = ec.fit_transform(
            data_hist=self.data_hist,
            data_sim=self.data_sim,
            prob="unif",
            random_state=42,
        )

        # Check that both fit and transform were performed
        self.assertIsNotNone(ec.parameters)
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("hs_corr", result.variables)

    def test_transform_with_ecdf_prob(self):
        """Test transform with ECDF probabilities"""
        ec = ExtremeCorrection(
            corr_config=self.corr_config, pot_config=self.pot_config, method="am"
        )

        ec.fit(data_hist=self.data_hist)
        result = ec.transform(data_sim=self.data_sim, prob="ecdf", random_state=42)

        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("hs_corr", result.variables)

    def test_test_method(self):
        """Test the Cramer-von-Mises test"""
        ec = ExtremeCorrection(
            corr_config=self.corr_config, pot_config=self.pot_config, method="am"
        )

        ec.fit(data_hist=self.data_hist)
        ec.transform(data_sim=self.data_sim, random_state=42)

        test_result = ec.test()

        # Check test result structure
        self.assertIsInstance(test_result, dict)
        self.assertIn("Statistic", test_result)
        self.assertIn("P-value", test_result)
        self.assertGreaterEqual(test_result["P-value"], 0)
        self.assertLessEqual(test_result["P-value"], 1)

    def test_correlations(self):
        """Test correlation calculations"""
        ec = ExtremeCorrection(
            corr_config=self.corr_config, pot_config=self.pot_config, method="am"
        )

        ec.fit(data_hist=self.data_hist)
        ec.transform(data_sim=self.data_sim, random_state=42)

        corr_result = ec.correlations()

        # Check correlation result structure
        self.assertIsInstance(corr_result, dict)
        self.assertIn("Spearman", corr_result)
        self.assertIn("Kendall", corr_result)
        self.assertIn("Pearson", corr_result)

        # Check that correlations are in valid range
        for corr_type, corr_value in corr_result.items():
            self.assertGreaterEqual(corr_value, -1)
            self.assertLessEqual(corr_value, 1)

    def test_plot_methods(self):
        """Test plotting methods"""
        ec = ExtremeCorrection(
            corr_config=self.corr_config, pot_config=self.pot_config, method="am"
        )

        ec.fit(data_hist=self.data_hist)
        ec.transform(data_sim=self.data_sim, random_state=42)

        # Test hist_retper_plot
        fig1, ax1 = ec.hist_retper_plot()
        self.assertIsNotNone(fig1)
        self.assertIsNotNone(ax1)

        # Test sim_retper_plot
        fig2, ax2 = ec.sim_retper_plot()
        self.assertIsNotNone(fig2)
        self.assertIsNotNone(ax2)

        # Test plot method
        figs, axes = ec.plot()
        self.assertEqual(len(figs), 2)
        self.assertEqual(len(axes), 2)

    def test_preprocess_data_single_dataset(self):
        """Test data preprocessing with single dataset"""
        ec = ExtremeCorrection(
            corr_config=self.corr_config, pot_config=self.pot_config, method="am"
        )

        pit_data, am_data = ec._preprocess_data(
            data=self.data_hist, var="hs", sim=False, join_sims=True
        )

        self.assertIsInstance(pit_data, np.ndarray)
        self.assertIsInstance(am_data, np.ndarray)
        self.assertGreater(len(pit_data), len(am_data))

    def test_preprocess_data_multiple_sims(self):
        """Test data preprocessing with multiple simulations"""
        ec = ExtremeCorrection(
            corr_config=self.corr_config, pot_config=self.pot_config, method="am"
        )

        pit_data, am_data = ec._preprocess_data(
            data=self.data_sim, var="hs", sim=True, join_sims=True
        )

        self.assertIsInstance(pit_data, np.ndarray)
        self.assertIsInstance(am_data, np.ndarray)
        # Check that all simulations are joined
        expected_length = len(self.data_sim.time) * len(self.data_sim.n_sim)
        self.assertEqual(len(pit_data), expected_length)

    def test_parameters_shape_am(self):
        """Test that fitted parameters have correct shape for AM"""
        ec = ExtremeCorrection(
            corr_config=self.corr_config, pot_config=self.pot_config, method="am"
        )

        ec.fit(data_hist=self.data_hist)

        # GEV has 3 parameters: loc, scale, shape
        self.assertEqual(len(ec.parameters), 3)
        self.assertGreater(ec.parameters[1], 0)  # scale > 0

    def test_parameters_shape_pot(self):
        """Test that fitted parameters have correct shape for POT"""
        ec = ExtremeCorrection(
            corr_config=self.corr_config, pot_config=self.pot_config, method="pot"
        )

        ec.fit(data_hist=self.data_hist)

        # GPD has 3 parameters: threshold, scale, shape
        self.assertEqual(len(ec.parameters), 3)
        self.assertGreater(ec.parameters[1], 0)  # scale > 0

    def test_no_correction_when_pvalue_high(self):
        """Test that no correction is applied when p-value is high"""
        # Create data that already fits well
        ec = ExtremeCorrection(
            corr_config=self.corr_config, pot_config=self.pot_config, method="am"
        )

        # Fit with same distribution
        ec.fit(data_hist=self.data_hist)

        # Create sim data from same distribution
        time_sim = xr.date_range(
            start="2010-01-01", periods=20 * 365, freq="D", use_cftime=True
        )
        sim_values_good = GEV.random(
            size=len(time_sim),
            loc=ec.parameters[0],
            scale=ec.parameters[1],
            shape=ec.parameters[2],
            random_state=100,
        )
        data_sim_good = xr.Dataset(
            {"hs": (["n_sim", "time"], sim_values_good.reshape(1, -1))},
            coords={"n_sim": [0], "time": time_sim},
        )

        _ = ec.transform(data_sim=data_sim_good, siglevel=0.05, random_state=42)

        # If p-value > siglevel, data should not be corrected
        if ec.p_value > 0.05:
            np.testing.assert_array_equal(ec.sim_am_data_corr, ec.sim_am_data)


if __name__ == "__main__":
    unittest.main()
