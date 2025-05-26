import tempfile
import unittest

from bluemath_tk.downloaders.noaa.noaa_downloader import NOAADownloader


class TestNOAADownloader(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.downloader = NOAADownloader(
            base_path_to_download=self.temp_dir,
            debug=True,
            check=False,  # Just check paths to download
        )

    def test_download_bulk_parameters(self):
        """Test downloading bulk parameters."""
        result = self.downloader.download_data(
            data_type="bulk_parameters",
            buoy_id="41001",
            years=[2023],
        )
        self.assertIsNotNone(result)
        print(f"\nBulk parameters result: {result}")

    def test_download_wave_spectra(self):
        """Test downloading wave spectra."""
        result = self.downloader.download_data(
            data_type="wave_spectra",
            buoy_id="41001",
            years=[2023],
        )
        self.assertIsNotNone(result)
        print(f"\nWave spectra result: {result}")

    def test_download_directional_spectra(self):
        """Test downloading directional spectra."""
        result = self.downloader.download_data(
            data_type="directional_spectra",
            buoy_id="41001",
            years=[2023],
        )
        self.assertIsNotNone(result)
        print(f"\nDirectional spectra result: {result}")

    def test_list_data_types(self):
        """Test listing available data types."""
        data_types = self.downloader.list_data_types()
        self.assertIsInstance(data_types, list)
        self.assertTrue(len(data_types) > 0)
        print(f"\nAvailable data types: {data_types}")

    def test_list_datasets(self):
        """Test listing available datasets."""
        datasets = self.downloader.list_datasets()
        self.assertIsInstance(datasets, list)
        self.assertTrue(len(datasets) > 0)
        print(f"\nAvailable datasets: {datasets}")

    def test_show_markdown_table(self):
        """Test showing markdown table."""
        self.downloader.show_markdown_table()


if __name__ == "__main__":
    unittest.main()
