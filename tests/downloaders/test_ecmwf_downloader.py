import tempfile
import unittest

from bluemath_tk.downloaders.ecmwf.ecmwf_downloader import ECMWFDownloader


class TestECMWFDownloader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.downloader = ECMWFDownloader(
            product="OpenData",
            base_path_to_download=self.temp_dir,
        )

    def test_list_datasets(self):
        """Test listing available datasets."""
        datasets = self.downloader.list_datasets()
        self.assertIsInstance(datasets, list)
        self.assertTrue(len(datasets) > 0)
        print(f"Available datasets: {datasets}")

    def test_download_data(self):
        """Test downloading data."""
        dataset = self.downloader.download_data(
            load_data=False,
            param=["msl"],
            step=[0, 240],
            type="fc",
            force=False,
        )
        self.assertIsInstance(dataset, str)
        print(dataset)

    def test_download_data_dry_run(self):
        """Test dry_run functionality."""
        dataset = self.downloader.download_data(
            load_data=False,
            param=["msl"],
            step=[0, 240],
            type="fc",
            dry_run=True,
        )
        self.assertIsInstance(dataset, str)
        print(f"\nDry run result: {dataset}")

    def test_download_data_open_data(self):
        """Test product-specific download method."""
        dataset = self.downloader.download_data_open_data(
            param=["msl"],
            step=[0, 240],
            type="fc",
            dry_run=True,
        )
        self.assertIsInstance(dataset, str)
        print(f"\nOpenData download result: {dataset}")

    def test_product_parameter(self):
        """Test that product parameter is required and validated."""
        # Test with valid product
        downloader = ECMWFDownloader(
            product="OpenData",
            base_path_to_download=self.temp_dir,
        )
        self.assertEqual(downloader.product, "OpenData")

        # Test with invalid product
        with self.assertRaises(ValueError):
            ECMWFDownloader(
                product="INVALID",
                base_path_to_download=self.temp_dir,
            )


if __name__ == "__main__":
    unittest.main()
