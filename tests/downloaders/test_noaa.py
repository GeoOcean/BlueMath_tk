import os
import tempfile
import unittest

import pandas as pd

from bluemath_tk.downloaders.noaa.noaa_downloader import read_bulk_parameters


class TestReadBulkParameters(unittest.TestCase):
    def test_years_are_not_duplicated(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            buoy_dir = os.path.join(tmp_dir, "NDBC", "buoy_data", "41001")
            os.makedirs(buoy_dir)
            pd.DataFrame(
                {
                    "YYYY": [2020, 2021, 2022],
                    "MM": [1, 1, 1],
                    "DD": [1, 1, 1],
                    "hh": [0, 0, 0],
                    "mm": [0, 0, 0],
                    "WVHT": [1.0, 2.0, 3.0],
                }
            ).to_csv(os.path.join(buoy_dir, "buoy_41001_bulk_parameters.csv"))

            df = read_bulk_parameters(tmp_dir, "41001", [2020, 2021])
            self.assertEqual(df["WVHT"].tolist(), [1.0, 2.0])
            self.assertIsNone(read_bulk_parameters(tmp_dir, "41001", 1999))


if __name__ == "__main__":
    unittest.main()
