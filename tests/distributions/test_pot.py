import unittest
import numpy as np

from bluemath_tk.distributions.pot import block_maxima

class TestPOT(unittest.TestCase):
    def setUp(self):
        # Create synthetic data for testing
        np.random.seed(42)
        self.x = np.random.lognormal(1, 1.2, size=int(365*5))  # 5-year of daily values
        
    def test_block_maxima_basic(self):
        # Test with default parameters
        idx, bmaxs = block_maxima(self.x)
        self.assertIsInstance(idx, np.ndarray)
        self.assertIsInstance(bmaxs, np.ndarray)
        self.assertEqual(idx.size, bmaxs.size)
        
    def test_block_maxima_custom_block(self):
        # Test with custom block size
        block_size = 5
        idx, bmaxs = block_maxima(self.x, block_size=block_size)
        expected_blocks = int(np.ceil(len(self.x) / block_size))
        self.assertEqual(bmaxs.size, expected_blocks)
        
    def test_block_maxima_independence(self):
        # Test minimum separation between peaks
        block_size = 5
        min_sep = 2
        idx, bmaxs = block_maxima(self.x, block_size=block_size, min_sep=min_sep)
        
        # Check that peaks are separated by at least min_sep
        differences = np.diff(idx)
        self.assertTrue(np.all(differences >= min_sep))
        
    def test_invalid_min_sep(self):
        # Test error raising for invalid min_sep
        block_size = 5
        min_sep = 4  # > (block_size + 1) / 2
        
        with self.assertRaises(ValueError):
            block_maxima(self.x, block_size=block_size, min_sep=min_sep)
            
    def test_block_maxima_types(self):
        # Test with different input types
        block_size = 5.0  # float instead of int
        idx, bmaxs = block_maxima(self.x, block_size=block_size)
        self.assertIsInstance(idx, np.ndarray)
        self.assertIsInstance(bmaxs, np.ndarray)
        
        # Test with list input
        x_list = self.x.tolist()
        idx, bmaxs = block_maxima(x_list, block_size=5)
        self.assertIsInstance(idx, np.ndarray)
        self.assertIsInstance(bmaxs, np.ndarray)
        
    def test_empty_input(self):
        # Test with empty input
        x_empty = np.array([])
        idx, bmaxs = block_maxima(x_empty)
        self.assertEqual(idx.size, 0)
        self.assertEqual(bmaxs.size, 0)
        
if __name__ == "__main__":
    unittest.main()