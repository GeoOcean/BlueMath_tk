import unittest

import numpy as np
from shapely.geometry import Polygon

from bluemath_tk.core.geo import (
    buffer_area_for_polygon,
    create_polygon,
    filter_points_in_polygon,
    geodesic_azimuth,
    geodesic_distance,
    geodesic_distance_azimuth,
    mask_points_outside_polygon,
    points_in_polygon,
    shoot,
)


class TestGeodesicDistance(unittest.TestCase):
    """Test the geodesic_distance function."""

    def test_scalar_inputs(self):
        """Test geodesic distance calculation with scalar inputs."""

        result = geodesic_distance(0, 0, 0, 90)
        expected = 90.0
        self.assertAlmostEqual(result, expected, delta=0.01)

    def test_array_inputs(self):
        """Test geodesic distance calculation with array inputs."""

        lat1 = np.array([0, 45])
        lon1 = np.array([0, -90])
        lat2 = np.array([0, -45])
        lon2 = np.array([90, 90])
        result = geodesic_distance(lat1, lon1, lat2, lon2)
        expected = np.array([90.0, 180.0])
        np.testing.assert_allclose(result, expected)

    def test_identical_points(self):
        """Test that distance between identical points is zero."""

        result = geodesic_distance(45.0, -120.0, 45.0, -120.0)
        self.assertAlmostEqual(result, 0.0, delta=0.01)

    def test_antipodal_points(self):
        """Test distance between antipodal points (should be 180 degrees)."""

        result = geodesic_distance(0, 0, 0, 180)
        self.assertAlmostEqual(result, 180.0, delta=0.01)


class TestGeodesicAzimuth(unittest.TestCase):
    """Test the geodesic_azimuth function."""

    def test_scalar_inputs(self):
        """Test azimuth calculation with scalar inputs."""

        result = geodesic_azimuth(0, 0, 0, 90)
        expected = 90.0
        self.assertAlmostEqual(result, expected, delta=0.01)

    def test_array_inputs(self):
        """Test azimuth calculation with array inputs."""

        lat1 = np.array([0, 45])
        lon1 = np.array([0, -90])
        lat2 = np.array([0, -45])
        lon2 = np.array([90, 90])
        result = geodesic_azimuth(lat1, lon1, lat2, lon2)
        expected = np.array([90.0, 90.0])
        np.testing.assert_allclose(result, expected)

    def test_north_direction(self):
        """Test azimuth for north direction."""

        result = geodesic_azimuth(0, 0, 90, 0)
        self.assertAlmostEqual(result, 0.0, delta=0.01)

    def test_south_direction(self):
        """Test azimuth for south direction."""

        result = geodesic_azimuth(0, 0, -90, 0)
        self.assertAlmostEqual(result, 180.0, delta=0.01)


class TestGeodesicDistanceAzimuth(unittest.TestCase):
    """Test the geodesic_distance_azimuth function."""

    def test_scalar_inputs(self):
        """Test combined distance and azimuth calculation with scalar inputs."""

        dist, az = geodesic_distance_azimuth(0, 0, 0, 90)
        expected_dist = 90.0
        expected_az = 90.0
        self.assertAlmostEqual(dist, expected_dist, delta=0.01)
        self.assertAlmostEqual(az, expected_az, delta=0.01)

    def test_array_inputs(self):
        """Test combined distance and azimuth calculation with array inputs."""

        lat1 = np.array([0, 45])
        lon1 = np.array([0, -90])
        lat2 = np.array([0, -45])
        lon2 = np.array([90, 90])
        dist, az = geodesic_distance_azimuth(lat1, lon1, lat2, lon2)
        expected_dist = np.array([90.0, 180.0])
        expected_az = np.array([90.0, 90.0])
        np.testing.assert_allclose(dist, expected_dist)
        np.testing.assert_allclose(az, expected_az)


class TestShoot(unittest.TestCase):
    """Test the shoot function."""

    def test_scalar_inputs(self):
        """Test shooting with scalar inputs."""

        lon_f, lat_f, baz = shoot(0, 0, 90, 111.195)  # ~1 degree at equator
        self.assertAlmostEqual(lon_f, 1.0, delta=0.01)
        self.assertAlmostEqual(lat_f, 0.0, delta=0.01)
        self.assertAlmostEqual(baz, 270.0, delta=0.01)

    def test_array_inputs(self):
        """Test shooting with array inputs."""

        lon1 = np.array([0, 0])
        lat1 = np.array([0, 45])
        azimuth = np.array([90, 45])
        maxdist = np.array([111.195, 111.195])
        lon_f, lat_f, baz = shoot(lon1, lat1, azimuth, maxdist)
        self.assertEqual(len(lon_f), 2)
        self.assertEqual(len(lat_f), 2)
        self.assertEqual(len(baz), 2)

    def test_pole_error(self):
        """Test that shooting from pole in non-meridian direction raises error."""

        with self.assertRaises(ValueError):
            shoot(0, 90, 90, 100)  # Shooting east from North Pole
        with self.assertRaises(ValueError):  # Shooting west from South Pole
            shoot(0, -90, 90, 100)  # Shooting west from South Pole


class TestPolygonFunctions(unittest.TestCase):
    """Test polygon-related functions."""

    def setUp(self):
        """Set up test polygon."""

        self.coords = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]  # Square
        self.polygon = create_polygon(self.coords)

    def test_create_polygon(self):
        """Test polygon creation."""

        poly = create_polygon(self.coords)
        self.assertIsInstance(poly, Polygon)
        self.assertTrue(poly.is_valid)

    def test_points_in_polygon(self):
        """Test point-in-polygon checking."""

        lon = [0.5, 2.0]
        lat = [0.5, 2.0]
        mask = points_in_polygon(lon, lat, self.polygon)
        expected = np.array([True, False])
        np.testing.assert_array_equal(mask, expected)

    def test_points_in_polygon_array_input(self):
        """Test point-in-polygon with numpy array inputs."""

        lon = np.array([0.5, 2.0])
        lat = np.array([0.5, 2.0])
        mask = points_in_polygon(lon, lat, self.polygon)
        expected = np.array([True, False])
        np.testing.assert_array_equal(mask, expected)

    def test_filter_points_in_polygon(self):
        """Test filtering points to keep only those inside polygon."""

        lon = [0.5, 2.0]
        lat = [0.5, 2.0]
        filtered_lon, filtered_lat = filter_points_in_polygon(lon, lat, self.polygon)
        np.testing.assert_array_equal(filtered_lon, np.array([0.5]))
        np.testing.assert_array_equal(filtered_lat, np.array([0.5]))

    def test_points_in_polygon_shape_mismatch(self):
        """Test that mismatched shapes raise ValueError."""

        lon = [0.5, 2.0]
        lat = [0.5]  # Different length
        with self.assertRaises(ValueError):
            points_in_polygon(lon, lat, self.polygon)

    def test_buffer_area_for_polygon(self):
        """Test polygon buffering."""

        buffered = buffer_area_for_polygon(self.polygon, 0.1)
        self.assertIsInstance(buffered, Polygon)
        self.assertGreater(
            buffered.area, self.polygon.area
        )  # Buffered should be larger


class TestMaskPointsOutsidePolygon(unittest.TestCase):
    """Test the mask_points_outside_polygon function."""

    def test_triangle_elements(self):
        """Test masking of triangle elements."""

        elements = np.array([[0, 1, 2], [1, 2, 3]])
        node_coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        mask = mask_points_outside_polygon(elements, node_coords, poly)
        expected = np.array([True, True])  # Both triangles should be inside
        np.testing.assert_array_equal(mask, expected)

    def test_triangles_partially_outside(self):
        """Test triangles that are partially outside the polygon."""

        elements = np.array([[0, 1, 2], [1, 2, 3]])
        node_coords = np.array([[0, 0], [1, 0], [0, 1], [2, 2]])  # Last point outside
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        mask = mask_points_outside_polygon(elements, node_coords, poly)
        expected = np.array([True, True])
        np.testing.assert_array_equal(mask, expected)


if __name__ == "__main__":
    unittest.main()
