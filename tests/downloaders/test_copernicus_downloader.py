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
            token=None,
            check=True,  # Just check paths to download
        )

    def test_download_data_era5(self):
        result = self.downloader.download_data_era5(
            variables=["spectra"],
            years=[f"{year:04d}" for year in range(2020, 2025)],
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
            area=[43.4, 350.4, 43.6, 350.6],  # [lat_min, lon_min, lat_max, lon_max]
        )
        print(result)


if __name__ == "__main__":
    unittest.main()


# mean_wave_period_based_on_first_moment/
# wave_spectral_directional_width/
# wave_spectral_directional_width_for_swell/
# wave_spectral_directional_width_for_wind_waves/
