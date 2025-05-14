import unittest

import numpy as np
import xarray as xr

from bluemath_tk.core.operations import spatial_gradient
from bluemath_tk.datamining.kma import KMA
from bluemath_tk.datamining.pca import PCA
from bluemath_tk.predictor.xwt import XWT

era5 = xr.open_dataset("https://geoocean.sci.unican.es/thredds/dodsC/geoocean/era5-msl")
era5["time"] = era5["time"].astype("timedelta64[D]") + np.datetime64("1940-01-01")
era5 = era5.sel(time=slice("2015", None)).chunk({"time": 365}).load()
era5["msl_gradient"] = spatial_gradient(era5["msl"])


class TestXWT(unittest.TestCase):
    def setUp(self):
        self.pca = PCA(n_components=0.95)
        self.kma = KMA(num_clusters=25, seed=42)
        self.xwt = XWT(steps={"pca": self.pca, "kma": self.kma})

    def test_fit(self):
        self.xwt.fit(
            data=era5,
            fit_params={
                "pca": {
                    "vars_to_stack": ["msl", "msl_gradient"],
                    "coords_to_stack": ["latitude", "longitude"],
                    "pca_dim_for_rows": "time",
                    "value_to_replace_nans": {"msl": 101325.0, "msl_gradient": 0.0},
                },
                "kma": {
                    "normalize_data": False,
                },
            },
        )
        self.assertIsInstance(self.xwt.num_clusters, int)
        self.assertEqual(self.xwt.num_clusters, 25)


if __name__ == "__main__":
    unittest.main()
