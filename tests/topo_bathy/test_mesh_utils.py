import unittest

import numpy as np

from bluemath_tk.topo_bathy.mesh_utils import (
    calculate_edges,
    detect_circumcenter_too_close,
    get_raster_resolution_meters,
)


class TestDetectCircumcenterTooClose(unittest.TestCase):
    """Test the detect_circumcenter_too_close function."""

    def test_function(self):
        """Test the function with a simple case."""
        # Define the input arrays
        X = np.array([0, 1, 0, 1, 2, 3, 3])
        Y = np.array([0, 0, 1, 1, 1, 0, 1])
        elements = np.array([[0, 1, 2], [1, 3, 2], [3, 4, 5], [4, 5, 6]])
        aj_threshold = 0.1

        # Call the function
        bad_elements_mask = detect_circumcenter_too_close(
            X=X, Y=Y, elements=elements, aj_threshold=aj_threshold
        )

        # Check the result
        expected_mask = np.array([True, True, False, False])
        np.testing.assert_array_equal(bad_elements_mask, expected_mask)


class TestCalculateEdges(unittest.TestCase):
    """Test the calculate_edges function."""

    def test_function(self):
        """Test the function with a simple case."""
        # Define the input arrays
        elements = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 0]])

        # Call the function
        edges = calculate_edges(elements)

        # Check the result
        expected_edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        np.testing.assert_array_equal(edges, expected_edges)


class TestGetRasterResolutionMeters(unittest.TestCase):
    """Test the get_raster_resolution_meters function."""

    def test_function(self):
        """Test the function with a simple case."""

        lon_center = 0
        lat_center = 0
        raster_resolution = 0.1
        project = lambda x, y: (x * 100000, y * 100000)

        resolution = get_raster_resolution_meters(
            lon_center=lon_center,
            lat_center=lat_center,
            raster_resolution=raster_resolution,
            project=project,
        )

        # Check the result
        expected_resolution = np.float64(7071.067811865475)
        np.testing.assert_almost_equal(resolution, expected_resolution, decimal=5)


if __name__ == "__main__":
    unittest.main()
