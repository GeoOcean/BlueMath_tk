import tempfile
import unittest

from bluemath_tk.downloaders.copernicus.copernicus_downloader import (
    CopernicusDownloader,
)


class TestCopernicusDownloader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.downloader = CopernicusDownloader(
            product="ERA5",
            base_path_to_download=self.temp_dir,
            token=None,  # "5cf7efae-13fc-4085-8a98-80d82bdb55f5",
            check=True,  # Just check paths to download
        )

    def test_download_data_era5(self):
        result = self.downloader.download_data_era5(
            variables=["swh"],
            years=["2024"],
            months=[
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
            ],
        )
        print(result)


if __name__ == "__main__":
    unittest.main()
