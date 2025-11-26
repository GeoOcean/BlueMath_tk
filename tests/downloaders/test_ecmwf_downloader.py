import tempfile
import unittest

from bluemath_tk.downloaders.ecmwf.ecmwf_downloader import ECMWFDownloader


class TestECMWFDownloader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.downloader = ECMWFDownloader(
            product="OpenData",
            base_path_to_download=self.temp_dir,
            check=True,  # Just check paths to download, do not actually download
        )

    def test_list_datasets(self):
        datasets = self.downloader.list_datasets()
        self.assertIsInstance(datasets, list)
        self.assertTrue(len(datasets) > 0)
        print(f"Available datasets: {datasets}")

    def test_download_data(self):
        dataset = self.downloader.download_data(
            load_data=False,
            param=["msl"],
            step=[0, 240],
            type="fc",
            force=False,
        )
        self.assertIsInstance(dataset, str)
        print(dataset)


if __name__ == "__main__":
    unittest.main()
