import unittest

import numpy as np

from bluemath_tk.distributions.gev import gev

class TestGEV(unittest.TestCase):
    def setUp(self):

        self.x = np.random.rand(1000) * 2

        self.loc = 0.0
        self.scale = 1.0
        self.shape_frechet = 0.1    # GEV (Frechet)
        self.shape_weibull = -0.1    # GEV (Weibull)
        self.shape_gumbel = 0.0     # Gumbel case

    def test_pdf_frechet(self):
        custom_pdf = gev.pdf(self.x, self.loc, self.scale, self.shape_frechet)

        self.assertIsInstance(custom_pdf, np.ndarray)
        self.assertEqual(custom_pdf.shape[0], 1000)

    def test_pdf_weibull(self):
        custom_pdf = gev.pdf(self.x, self.loc, self.scale, self.shape_weibull)

        self.assertIsInstance(custom_pdf, np.ndarray)
        self.assertEqual(custom_pdf.shape[0], 1000)

    def test_pdf_gumbel(self):
        custom_pdf = gev.pdf(self.x, self.loc, self.scale, self.shape_gumbel)

        self.assertIsInstance(custom_pdf, np.ndarray)
        self.assertEqual(custom_pdf.shape[0], 1000)

    def test_pdf_invalid_scale(self):
        # Scale must be > 0
        with self.assertRaises(ValueError):
            gev.pdf(self.x, self.loc, 0.0, self.shape_frechet)


if __name__ == '__main__':
    unittest.main()