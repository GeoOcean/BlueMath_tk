import re
import unittest

from siphon.catalog import TDSCatalog

from bluemath_tk.config.paths import (
    PATHS,
    get_catalog_folders,
    get_paths,
    get_thredds_catalog,
    update_paths,
)


def test_paths_dictionary_structure():
    """
    Test that PATHS dictionary has the expected structure and keys.
    """

    tc = unittest.TestCase()
    required_keys = ["SHYTCWAVES_COEFS", "SHYTCWAVES_BULK", "SHYTCWAVES_MDA"]

    # Check all required keys exist
    for key in required_keys:
        tc.assertIn(key, PATHS, f"Missing required key: {key}")

    # Check all paths are strings
    for key, value in PATHS.items():
        tc.assertIsInstance(value, str, f"Path for {key} is not a string")

    # Check paths contain expected components using regex
    patterns = {
        "SHYTCWAVES_COEFS": r".*GEOOCEAN.*coef.*",
        "SHYTCWAVES_BULK": r".*GEOOCEAN.*bulk.*",
        "SHYTCWAVES_MDA": r".*GEOOCEAN.*mda.*",
    }

    for key, pattern in patterns.items():
        tc.assertTrue(
            re.search(pattern, PATHS[key], re.IGNORECASE),
            f"Path for {key} does not match pattern {pattern}",
        )


def test_update_paths():
    """
    Test the update_paths function.
    """

    tc = unittest.TestCase()
    # Test updating with new path
    new_path = "/test/path/to/data.nc"
    update_paths({"TEST_PATH": new_path})

    # Check new path was added
    tc.assertIn("TEST_PATH", PATHS)
    tc.assertEqual(PATHS["TEST_PATH"], new_path)


def test_get_paths():
    """
    Test the get_paths function returns the correct dictionary.
    """

    tc = unittest.TestCase()
    paths = get_paths()
    tc.assertIsInstance(paths, dict)
    tc.assertEqual(paths, PATHS)


def test_get_thredds_catalog():
    """
    Test thredds catalog is available.
    """

    tc = unittest.TestCase()
    catalog = get_thredds_catalog()
    tc.assertIsInstance(catalog, TDSCatalog)


def test_get_catalog_folders():
    """
    Test getting catalog folders from Thredds server.
    """

    tc = unittest.TestCase()
    folders = get_catalog_folders()
    tc.assertIsInstance(folders, dict)
    tc.assertGreater(len(folders), 0, "No folders found in catalog")


def suite():
    """
    Create a test suite.
    """

    suite = unittest.TestSuite()
    suite.addTest(unittest.FunctionTestCase(test_paths_dictionary_structure))
    suite.addTest(unittest.FunctionTestCase(test_update_paths))
    suite.addTest(unittest.FunctionTestCase(test_get_paths))
    suite.addTest(unittest.FunctionTestCase(test_get_thredds_catalog))
    suite.addTest(unittest.FunctionTestCase(test_get_catalog_folders))

    return suite


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
