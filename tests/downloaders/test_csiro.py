import os
import unittest

import numpy as np
import pandas as pd
from test_transfer import LocalServerTestCase

from bluemath_tk.downloaders.csiro.csiro_downloader import (
    CSIRODownloader,
    nearest_point_index,
)


class TestNearestPointIndex(unittest.TestCase):
    def test_longitude_conventions(self):
        lons = np.array([0.0, 90.0, 355.0, 180.0])
        lats = np.array([0.0, 0.0, 44.0, 0.0])
        idx, distance = nearest_point_index(lons, lats, -5.0, 44.0)
        self.assertEqual(idx, 2)
        self.assertAlmostEqual(distance, 0.0, places=3)

    def test_distance(self):
        idx, distance = nearest_point_index(np.array([0.0]), np.array([0.0]), 1.0, 0.0)
        self.assertEqual(idx, 0)
        self.assertAlmostEqual(distance, 111.19, delta=0.1)


class TestCSIROPaths(unittest.TestCase):
    def setUp(self):
        self.whacs = CSIRODownloader(product="WHACS", base_path_to_download="/data")
        self.cawcr = CSIRODownloader(product="CAWCR", base_path_to_download="/data")

    def test_unknown_product(self):
        with self.assertRaises(ValueError):
            CSIRODownloader(product="OTHER", base_path_to_download="/data")

    def test_whacs_paths(self):
        period = pd.Period("2000-02", freq="M")
        url = self.whacs.get_url("global-smc", period, "phs1")
        self.assertTrue(
            url.startswith("https://data-cbr.csiro.au/thredds5/fileServer/catch_all/")
        )
        self.assertTrue(
            url.endswith(
                "/global-smc/1hr/phs1/"
                "phs1_WHACS_hindcast_WHACS_ERA5_1hr_200002010000-200002292300.nc"
            )
        )
        self.assertEqual(
            self.whacs.get_local_path("global-smc", period, "phs1"),
            os.path.normpath(
                "/data/WHACS/global-smc/phs1/"
                "phs1_WHACS_hindcast_WHACS_ERA5_1hr_200002010000-200002292300.nc"
            ),
        )
        self.assertIn(
            "/thredds5/dodsC/",
            self.whacs.get_url("global-smc", period, "phs1", service="opendap"),
        )

    def test_cawcr_spec_eras(self):
        self.assertEqual(
            self.cawcr.get_local_path("spec", pd.Period("2013-05", freq="M")),
            os.path.normpath("/data/CAWCR/CSIRO_spectra/1979/ww3.201305_spec.nc"),
        )
        self.assertEqual(
            self.cawcr.get_local_path("spec", pd.Period("2013-06", freq="M")),
            os.path.normpath("/data/CAWCR/CSIRO_spectra/2013/ww3.201306_spec.nc"),
        )
        self.assertEqual(
            self.cawcr.get_local_path("gridded", pd.Period("2025-01", freq="M")),
            os.path.normpath("/data/CAWCR/CSIRO_partition/ww3.glob_24m.202501.nc"),
        )

    def test_months_are_clipped_to_availability(self):
        months = self.whacs.get_months("global-smc", "1970-01", "2030-12")
        self.assertEqual(str(months[0]), "1979-01")
        self.assertEqual(str(months[-1]), "2023-12")
        self.assertEqual(
            len(self.whacs.get_months("global-smc", "2000-01", "2000-12")), 12
        )

    def test_variables_are_checked(self):
        with self.assertRaises(ValueError):
            self.whacs.download_data("global-smc")
        with self.assertRaises(ValueError):
            self.whacs.download_data("global-smc", variables=["nope"])
        with self.assertRaises(ValueError):
            self.whacs.download_data("unknown-dataset", variables=["hs"])


class TestCSIRODownloadFile(LocalServerTestCase):
    def setUp(self):
        super().setUp()
        self.downloader = CSIRODownloader(
            product="CAWCR",
            base_path_to_download=self.tmp_dir.name,
            retries=1,
            retry_backoff=0,
        )
        self.url = f"{self.base_url}/data.bin"
        self.output_file = os.path.join(self.tmp_dir.name, "data.bin")

    def test_download_and_skip(self):
        result = self.downloader._download_file(
            self.url, self.output_file, force=False, verify_existing=True, dry_run=False
        )
        self.assertEqual(result.downloaded_files, [self.output_file])
        with open(self.output_file, "rb") as f:
            self.assertEqual(f.read(), self.content)

        result = self.downloader._download_file(
            self.url, self.output_file, force=False, verify_existing=True, dry_run=False
        )
        self.assertEqual(result.skipped_files, [self.output_file])

    def test_incomplete_local_file_is_downloaded_again(self):
        with open(self.output_file, "wb") as f:
            f.write(self.content[:100])

        result = self.downloader._download_file(
            self.url, self.output_file, force=False, verify_existing=True, dry_run=True
        )
        self.assertEqual(result.skipped_files, [self.output_file])
        self.assertIn(
            "would download again", result.metadata["skip_reasons"][self.output_file]
        )

        result = self.downloader._download_file(
            self.url, self.output_file, force=False, verify_existing=True, dry_run=False
        )
        self.assertEqual(result.downloaded_files, [self.output_file])
        self.assertEqual(os.path.getsize(self.output_file), len(self.content))

    def test_not_available_on_server(self):
        result = self.downloader._download_file(
            f"{self.base_url}/missing.nc",
            os.path.join(self.tmp_dir.name, "missing.nc"),
            force=False,
            verify_existing=True,
            dry_run=False,
        )
        self.assertEqual(len(result.skipped_files), 1)
        self.assertFalse(result.has_errors)


if __name__ == "__main__":
    unittest.main()
