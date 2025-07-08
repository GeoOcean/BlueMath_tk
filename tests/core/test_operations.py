import unittest

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import StandardScaler

from bluemath_tk.core.operations import (
    convert_lonlat_to_utm,
    convert_utm_to_lonlat,
    denormalize,
    destandarize,
    get_degrees_from_uv,
    get_uv_components,
    mathematical_to_nautical,
    nautical_to_mathematical,
    normalize,
    spatial_gradient,
    standarize,
)


class TestNormalize(unittest.TestCase):
    """Test the normalize function."""

    def setUp(self):
        """Set up test data."""

        self.df_data = pd.DataFrame(
            {
                "Hs": [1.0, 2.0, 3.0, 4.0, 5.0],
                "Tp": [5.0, 10.0, 15.0, 20.0, 25.0],
                "Dir": [0.0, 90.0, 180.0, 270.0, 360.0],
            }
        )
        self.ds_data = xr.Dataset(
            {
                "Hs": (("time",), [1.0, 2.0, 3.0, 4.0, 5.0]),
                "Tp": (("time",), [5.0, 10.0, 15.0, 20.0, 25.0]),
                "Dir": (("time",), [0.0, 90.0, 180.0, 270.0, 360.0]),
            },
            coords={"time": pd.date_range("2000-01-01", periods=5)},
        )

    def test_normalize_dataframe(self):
        """Test normalization of pandas DataFrame."""

        normalized_data, scale_factor = normalize(self.df_data)

        # Check that data is normalized to 0-1 range
        for col in normalized_data.columns:
            self.assertAlmostEqual(normalized_data[col].min(), 0.0, places=10)
            self.assertAlmostEqual(normalized_data[col].max(), 1.0, places=10)

        # Check scale factor structure
        for var in scale_factor:
            self.assertIn(var, self.df_data.columns)
            self.assertEqual(len(scale_factor[var]), 2)
            self.assertLess(scale_factor[var][0], scale_factor[var][1])

    def test_normalize_dataset(self):
        """Test normalization of xarray Dataset."""

        normalized_data, scale_factor = normalize(self.ds_data)

        # Check that data is normalized to 0-1 range
        for var in normalized_data.data_vars:
            self.assertAlmostEqual(normalized_data[var].min(), 0.0, places=10)
            self.assertAlmostEqual(normalized_data[var].max(), 1.0, places=10)

        # Check scale factor structure
        for var in scale_factor:
            self.assertIn(var, self.ds_data.data_vars)
            self.assertEqual(len(scale_factor[var]), 2)

    def test_normalize_with_custom_scale_factor(self):
        """Test normalization with custom scale factors."""

        custom_scale = {"Hs": [0.0, 10.0], "Tp": [0.0, 30.0], "Dir": [0.0, 360.0]}
        _normalized_data, scale_factor = normalize(self.df_data, custom_scale)

        # Check that custom scale factors are used
        self.assertEqual(scale_factor["Hs"], [0.0, 10.0])
        self.assertEqual(scale_factor["Tp"], [0.0, 30.0])
        self.assertEqual(scale_factor["Dir"], [0.0, 360.0])

    def test_normalize_invalid_data_type(self):
        """Test that invalid data type raises TypeError."""

        with self.assertRaises(TypeError):
            normalize(np.array([1, 2, 3]))

    def test_normalize_preserves_original_data(self):
        """Test that original data is not modified."""

        original_data = self.df_data.copy()
        normalize(self.df_data)
        pd.testing.assert_frame_equal(original_data, self.df_data)


class TestDenormalize(unittest.TestCase):
    """Test the denormalize function."""

    def setUp(self):
        """Set up test data."""

        self.normalized_df = pd.DataFrame(
            {"Hs": [0.0, 0.25, 0.5, 0.75, 1.0], "Tp": [0.0, 0.25, 0.5, 0.75, 1.0]}
        )
        self.normalized_ds = xr.Dataset(
            {
                "Hs": (("time",), [0.0, 0.25, 0.5, 0.75, 1.0]),
                "Tp": (("time",), [0.0, 0.25, 0.5, 0.75, 1.0]),
            },
            coords={"time": pd.date_range("2000-01-01", periods=5)},
        )
        self.scale_factor = {"Hs": [1.0, 5.0], "Tp": [5.0, 25.0]}

    def test_denormalize_dataframe(self):
        """Test denormalization of pandas DataFrame."""

        denormalized_data = denormalize(self.normalized_df, self.scale_factor)

        # Check that data is denormalized correctly
        expected_hs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_tp = np.array([5.0, 10.0, 15.0, 20.0, 25.0])

        np.testing.assert_allclose(denormalized_data["Hs"].values, expected_hs)
        np.testing.assert_allclose(denormalized_data["Tp"].values, expected_tp)

    def test_denormalize_dataset(self):
        """Test denormalization of xarray Dataset."""

        denormalized_data = denormalize(self.normalized_ds, self.scale_factor)

        # Check that data is denormalized correctly
        expected_hs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_tp = np.array([5.0, 10.0, 15.0, 20.0, 25.0])

        np.testing.assert_allclose(denormalized_data["Hs"].values, expected_hs)
        np.testing.assert_allclose(denormalized_data["Tp"].values, expected_tp)

    def test_denormalize_invalid_data_type(self):
        """Test that invalid data type raises TypeError."""

        with self.assertRaises(TypeError):
            denormalize(np.array([1, 2, 3]), self.scale_factor)

    def test_denormalize_preserves_original_data(self):
        """Test that original data is not modified."""

        original_data = self.normalized_df.copy()
        denormalize(self.normalized_df, self.scale_factor)
        pd.testing.assert_frame_equal(original_data, self.normalized_df)


class TestStandarize(unittest.TestCase):
    """Test the standarize function."""

    def setUp(self):
        """Set up test data."""

        self.np_data = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [5.0, 10.0, 15.0, 20.0, 25.0],
                [0.0, 90.0, 180.0, 270.0, 360.0],
            ]
        )
        self.df_data = pd.DataFrame(
            {
                "Hs": [1.0, 2.0, 3.0, 4.0, 5.0],
                "Tp": [5.0, 10.0, 15.0, 20.0, 25.0],
                "Dir": [0.0, 90.0, 180.0, 270.0, 360.0],
            }
        )
        self.ds_data = xr.Dataset(
            {
                "Hs": (("time",), [1.0, 2.0, 3.0, 4.0, 5.0]),
                "Tp": (("time",), [5.0, 10.0, 15.0, 20.0, 25.0]),
                "Dir": (("time",), [0.0, 90.0, 180.0, 270.0, 360.0]),
            },
            coords={"time": pd.date_range("2000-01-01", periods=5)},
        )

    def test_standarize_numpy_array(self):
        """Test standardization of numpy array."""

        standarized_data, scaler = standarize(self.np_data)

        # Check that data has mean close to 0 and std close to 1
        self.assertAlmostEqual(standarized_data.mean(), 0.0, delta=0.15)
        self.assertAlmostEqual(standarized_data.std(), 1.0, delta=0.15)

        # Check that scaler is StandardScaler instance
        self.assertIsInstance(scaler, StandardScaler)

    def test_standarize_dataframe(self):
        """Test standardization of pandas DataFrame."""

        standarized_data, _scaler = standarize(self.df_data)

        # Check that data has mean close to 0 and std close to 1
        for col in standarized_data.columns:
            self.assertAlmostEqual(standarized_data[col].mean(), 0.0, delta=0.15)
            self.assertAlmostEqual(standarized_data[col].std(), 1.0, delta=0.15)

        # Check that column names are preserved
        self.assertEqual(list(standarized_data.columns), list(self.df_data.columns))

    def test_standarize_dataset(self):
        """Test standardization of xarray Dataset."""

        standarized_data, _scaler = standarize(self.ds_data)

        # Check that data has mean close to 0 and std close to 1
        for var in standarized_data.data_vars:
            self.assertAlmostEqual(standarized_data[var].mean(), 0.0, delta=0.15)
            self.assertAlmostEqual(standarized_data[var].std(), 1.0, delta=0.15)

        # Check that coordinates are preserved
        self.assertEqual(list(standarized_data.coords), list(self.ds_data.coords))

    def test_standarize_with_existing_scaler(self):
        """Test standardization with existing scaler."""

        scaler = StandardScaler()
        _standarized_data, returned_scaler = standarize(self.np_data, scaler=scaler)

        self.assertIs(returned_scaler, scaler)

    def test_standarize_transform_only(self):
        """Test standardization with transform=True."""

        # First fit a scaler
        _, scaler = standarize(self.np_data)

        # Then transform new data
        new_data = np.array(
            [[1, 2, 3, 4, 5], [5, 10, 15, 20, 25], [0, 90, 180, 270, 360]], dtype=float
        )
        standarized_data, _ = standarize(new_data, scaler=scaler, transform=True)

        # Check that data is transformed but not fitted
        self.assertNotEqual(standarized_data.mean(), 0.0)


class TestDestandarize(unittest.TestCase):
    """Test the destandarize function."""

    def setUp(self):
        """Set up test data."""

        self.original_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        self.standarized_data, self.scaler = standarize(self.original_data)

    def test_destandarize_numpy_array(self):
        """Test destandarization of numpy array."""

        destandarized_data = destandarize(self.standarized_data, self.scaler)

        # Check that data is restored to original values
        np.testing.assert_allclose(destandarized_data, self.original_data, rtol=1e-10)

    def test_destandarize_dataframe(self):
        """Test destandarization of pandas DataFrame."""

        df_standarized = pd.DataFrame(self.standarized_data, columns=["A", "B", "C"])
        df_original = pd.DataFrame(self.original_data, columns=["A", "B", "C"])

        destandarized_data = destandarize(df_standarized, self.scaler)

        # Check that data is restored to original values
        pd.testing.assert_frame_equal(destandarized_data, df_original)

    def test_destandarize_dataset(self):
        """Test destandarization of xarray Dataset."""

        ds_standarized = xr.Dataset(
            {
                "A": (("time",), self.standarized_data[:, 0]),
                "B": (("time",), self.standarized_data[:, 1]),
                "C": (("time",), self.standarized_data[:, 2]),
            },
            coords={"time": pd.date_range("2000-01-01", periods=3)},
        )
        ds_original = xr.Dataset(
            {
                "A": (("time",), self.original_data[:, 0]),
                "B": (("time",), self.original_data[:, 1]),
                "C": (("time",), self.original_data[:, 2]),
            },
            coords={"time": pd.date_range("2000-01-01", periods=3)},
        )
        destandarized_data = destandarize(ds_standarized, self.scaler)

        # Check that data is restored to original values
        xr.testing.assert_allclose(destandarized_data, ds_original)


class TestUVComponents(unittest.TestCase):
    """Test the get_uv_components and get_degrees_from_uv functions."""

    def test_get_uv_components_scalar(self):
        """Test UV components calculation with scalar input."""

        xu, xv = get_uv_components(np.array([0]))

        # North direction (0°) should give u=0, v=1
        self.assertAlmostEqual(xu[0], 0.0, places=10)
        self.assertAlmostEqual(xv[0], 1.0, places=10)

    def test_get_uv_components_array(self):
        """Test UV components calculation with array input."""

        angles = np.array([0, 90, 180, 270])
        xu, xv = get_uv_components(angles)

        # Check specific directions
        # North (0°)
        self.assertAlmostEqual(xu[0], 0.0, places=10)
        self.assertAlmostEqual(xv[0], 1.0, places=10)

        # East (90°)
        self.assertAlmostEqual(xu[1], 1.0, places=10)
        self.assertAlmostEqual(xv[1], 0.0, places=10)

        # South (180°)
        self.assertAlmostEqual(xu[2], 0.0, places=10)
        self.assertAlmostEqual(xv[2], -1.0, places=10)

        # West (270°)
        self.assertAlmostEqual(xu[3], -1.0, places=10)
        self.assertAlmostEqual(xv[3], 0.0, places=10)

    def test_get_degrees_from_uv_scalar(self):
        """Test degrees calculation from UV components with scalar input."""

        xu = np.array([0])
        xv = np.array([1])
        degrees = get_degrees_from_uv(xu, xv)

        # u=0, v=1 should give 0° (North)
        self.assertAlmostEqual(degrees[0], 0.0, places=10)

    def test_get_degrees_from_uv_array(self):
        """Test degrees calculation from UV components with array input."""

        xu = np.array([0, 1, 0, -1])
        xv = np.array([1, 0, -1, 0])
        degrees = get_degrees_from_uv(xu, xv)

        # Check specific directions
        expected = np.array([0, 90, 180, 270])
        np.testing.assert_allclose(degrees, expected, rtol=1e-10)

    def test_uv_components_roundtrip(self):
        """Test roundtrip conversion: degrees -> UV -> degrees."""

        original_angles = np.array([0, 45, 90, 135, 180, 225, 270, 315])
        xu, xv = get_uv_components(original_angles)
        recovered_angles = get_degrees_from_uv(xu, xv)

        np.testing.assert_allclose(recovered_angles, original_angles, rtol=1e-10)


class TestCoordinateConversion(unittest.TestCase):
    """Test the coordinate conversion functions."""

    def setUp(self):
        """Set up test data."""

        self.lon = np.array([-3.0, -2.0, -1.0, 0.0, 1.0])
        self.lat = np.array([40.0, 41.0, 42.0, 43.0, 44.0])
        self.projection = "SPAIN"

    def test_convert_lonlat_to_utm_scalar(self):
        """Test longitude/latitude to UTM conversion with scalar input."""

        lon_scalar = np.array([0.0])
        lat_scalar = np.array([40.0])

        utm_x, utm_y = convert_lonlat_to_utm(lon_scalar, lat_scalar, self.projection)

        self.assertEqual(len(utm_x), 1)
        self.assertEqual(len(utm_y), 1)
        self.assertIsInstance(utm_x[0], float)
        self.assertIsInstance(utm_y[0], float)

    def test_convert_lonlat_to_utm_array(self):
        """Test longitude/latitude to UTM conversion with array input."""

        utm_x, utm_y = convert_lonlat_to_utm(self.lon, self.lat, self.projection)

        self.assertEqual(len(utm_x), len(self.lon))
        self.assertEqual(len(utm_y), len(self.lat))
        self.assertTrue(np.all(np.isfinite(utm_x)))
        self.assertTrue(np.all(np.isfinite(utm_y)))

    def test_convert_utm_to_lonlat_scalar(self):
        """Test UTM to longitude/latitude conversion with scalar input."""

        utm_x_scalar = np.array([500000.0])
        utm_y_scalar = np.array([4000000.0])

        lon, lat = convert_utm_to_lonlat(utm_x_scalar, utm_y_scalar, self.projection)

        self.assertEqual(len(lon), 1)
        self.assertEqual(len(lat), 1)
        self.assertIsInstance(lon[0], float)
        self.assertIsInstance(lat[0], float)

    def test_convert_utm_to_lonlat_array(self):
        """Test UTM to longitude/latitude conversion with array input."""

        # First convert to UTM
        utm_x, utm_y = convert_lonlat_to_utm(self.lon, self.lat, self.projection)

        # Then convert back to lon/lat
        lon_back, lat_back = convert_utm_to_lonlat(utm_x, utm_y, self.projection)

        # Check roundtrip conversion
        np.testing.assert_allclose(lon_back, self.lon, rtol=1e-6)
        np.testing.assert_allclose(lat_back, self.lat, rtol=1e-6)

    def test_convert_lonlat_to_utm_meshgrid(self):
        """Test longitude/latitude to UTM conversion with meshgrid input."""

        lon_mesh = np.array([-2.0, -1.0, 0.0])
        lat_mesh = np.array([40.0, 41.0])

        utm_x, utm_y = convert_lonlat_to_utm(lon_mesh, lat_mesh, self.projection)

        # Should return 1D arrays for meshgrid input
        self.assertEqual(len(utm_x), len(lon_mesh))
        self.assertEqual(len(utm_y), len(lat_mesh))

    def test_convert_utm_to_lonlat_meshgrid(self):
        """Test UTM to longitude/latitude conversion with meshgrid input."""

        utm_x_mesh = np.array([500000.0, 600000.0, 700000.0])
        utm_y_mesh = np.array([4000000.0, 4100000.0])

        lon, lat = convert_utm_to_lonlat(utm_x_mesh, utm_y_mesh, self.projection)

        # Should return 1D arrays for meshgrid input
        self.assertEqual(len(lon), len(utm_x_mesh))
        self.assertEqual(len(lat), len(utm_y_mesh))


class TestSpatialGradient(unittest.TestCase):
    """Test the spatial_gradient function."""

    def setUp(self):
        """Set up test data."""

        # Create a simple 2D field with known gradient
        time = pd.date_range("2000-01-01", periods=3)
        lat = np.array([40.0, 41.0, 42.0])
        lon = np.array([-3.0, -2.0, -1.0])

        # Create a linear field: f(x,y) = x + y
        data = np.zeros((3, 3, 3))
        for t in range(3):
            for i, la in enumerate(lat):
                for j, lo in enumerate(lon):
                    data[t, i, j] = la + lo

        self.data = xr.DataArray(
            data,
            coords={"time": time, "latitude": lat, "longitude": lon},
            dims=["time", "latitude", "longitude"],
        )

    def test_spatial_gradient_basic(self):
        """Test basic spatial gradient calculation."""

        gradient = spatial_gradient(self.data)

        # Check output structure
        self.assertEqual(gradient.dims, self.data.dims)
        self.assertEqual(gradient.shape, self.data.shape)
        self.assertIn("units", gradient.attrs)
        self.assertIn("name", gradient.attrs)

        # Check that gradient is finite
        self.assertTrue(np.all(np.isfinite(gradient.values)))

    def test_spatial_gradient_attributes(self):
        """Test that gradient has correct attributes."""

        gradient = spatial_gradient(self.data)

        self.assertEqual(gradient.attrs["units"], "m^2/s^2")
        self.assertEqual(gradient.attrs["name"], "Gradient")

    def test_spatial_gradient_non_zero(self):
        """Test that gradient is non-zero for non-constant field."""

        gradient = spatial_gradient(self.data)

        # For a linear field, gradient should be non-zero
        self.assertGreater(np.abs(gradient.values).max(), 0.0)


class TestAngleConversion(unittest.TestCase):
    """Test the angle conversion functions."""

    def test_nautical_to_mathematical_scalar(self):
        """Test nautical to mathematical conversion with scalar input."""

        # Test key directions
        self.assertAlmostEqual(
            nautical_to_mathematical(0), 90, places=10
        )  # North -> East
        self.assertAlmostEqual(
            nautical_to_mathematical(90), 0, places=10
        )  # East -> East
        self.assertAlmostEqual(
            nautical_to_mathematical(180), 270, places=10
        )  # South -> West
        self.assertAlmostEqual(
            nautical_to_mathematical(270), 180, places=10
        )  # West -> South

    def test_nautical_to_mathematical_array(self):
        """Test nautical to mathematical conversion with array input."""

        nautical_angles = np.array([0, 90, 180, 270])
        mathematical_angles = nautical_to_mathematical(nautical_angles)

        expected = np.array([90, 0, 270, 180])
        np.testing.assert_allclose(mathematical_angles, expected, rtol=1e-10)

    def test_mathematical_to_nautical_scalar(self):
        """Test mathematical to nautical conversion with scalar input."""

        # Test key directions
        self.assertAlmostEqual(
            mathematical_to_nautical(90), 0, places=10
        )  # East -> North
        self.assertAlmostEqual(
            mathematical_to_nautical(0), 90, places=10
        )  # East -> East
        self.assertAlmostEqual(
            mathematical_to_nautical(270), 180, places=10
        )  # West -> South
        self.assertAlmostEqual(
            mathematical_to_nautical(180), 270, places=10
        )  # South -> West

    def test_mathematical_to_nautical_array(self):
        """Test mathematical to nautical conversion with array input."""

        mathematical_angles = np.array([90, 0, 270, 180])
        nautical_angles = mathematical_to_nautical(mathematical_angles)

        expected = np.array([0, 90, 180, 270])
        np.testing.assert_allclose(nautical_angles, expected, rtol=1e-10)

    def test_angle_conversion_roundtrip(self):
        """Test roundtrip conversion: nautical -> mathematical -> nautical."""

        original_nautical = np.array([0, 45, 90, 135, 180, 225, 270, 315])
        mathematical = nautical_to_mathematical(original_nautical)
        recovered_nautical = mathematical_to_nautical(mathematical)

        np.testing.assert_allclose(recovered_nautical, original_nautical, rtol=1e-10)

    def test_mathematical_to_nautical_zero_case(self):
        """Test mathematical to nautical conversion with zero input."""

        result = mathematical_to_nautical(0)
        self.assertAlmostEqual(result, 90, places=10)


if __name__ == "__main__":
    unittest.main()
