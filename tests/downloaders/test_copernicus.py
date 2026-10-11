import os
import tempfile
import unittest

try:
    from bluemath_tk.downloaders.copernicus import CopernicusDownloader
except ImportError:  # cdsapi is an optional dependency
    CopernicusDownloader = None


@unittest.skipIf(CopernicusDownloader is None, "cdsapi not installed")
class TestCopernicusTasks(unittest.TestCase):
    """
    Requests and file names, checked with dry runs (no request is sent to the CDS).
    """

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.cerra = CopernicusDownloader(
            product="CERRA",
            base_path_to_download=self.tmp_dir.name,
            api_key="dummy",
            debug=False,
        )
        self.era5 = CopernicusDownloader(
            product="ERA5",
            base_path_to_download=self.tmp_dir.name,
            api_key="dummy",
            debug=False,
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _tasks(self, downloader, **kwargs):
        tasks = []
        downloader._run_tasks = lambda new_tasks, force, dry_run: tasks.extend(
            new_tasks
        )
        downloader.download_data(**kwargs)
        return tasks

    @staticmethod
    def _names(tasks):
        return [os.path.basename(task["output_file"]) for task in tasks]

    def test_file_periods(self):
        full_year = dict(
            variables=["2m_temperature"], years=[2020], months=range(1, 13)
        )
        self.assertEqual(
            self._names(self._tasks(self.cerra, **full_year)),
            ["2m_temperature_2020.nc"],
        )
        monthly = self._names(self._tasks(self.cerra, file_period="month", **full_year))
        self.assertEqual(len(monthly), 12)
        self.assertEqual(monthly[0], "2m_temperature_2020_01.nc")
        self.assertEqual(
            self._names(
                self._tasks(
                    self.cerra, variables=["2m_temperature"], years=[2020], months=[3]
                )
            ),
            ["2m_temperature_2020_03.nc"],
        )
        with self.assertRaises(ValueError):
            self._tasks(self.cerra, file_period="week", **full_year)

    def test_cerra_request(self):
        (task,) = self._tasks(
            self.cerra, variables=["mean_sea_level_pressure"], years=[2024], months=[2]
        )
        self.assertEqual(task["dataset"], "reanalysis-cerra-single-levels")
        self.assertEqual(task["request"]["month"], ["02"])
        self.assertEqual(task["request"]["year"], ["2024"])
        self.assertEqual(len(task["request"]["time"]), 8)

    def test_era5_spectra_dates(self):
        (task,) = self._tasks(
            self.era5,
            variables=["spectra"],
            years=[2024],
            months=[2],
            area=[45, -5, 43, -3],
        )
        self.assertEqual(task["request"]["date"], "2024-02-01/to/2024-02-29")
        self.assertEqual(task["request"]["area"], "45/-5/43/-3")

    def test_legacy_yearly_file_is_recognised(self):
        (task,) = self._tasks(
            self.cerra, variables=["2m_temperature"], years=[2020], months=range(1, 13)
        )
        legacy_file = task["legacy_file"]
        os.makedirs(os.path.dirname(legacy_file))
        open(legacy_file, "w").close()

        downloader = CopernicusDownloader(
            product="CERRA",
            base_path_to_download=self.tmp_dir.name,
            api_key="dummy",
            debug=False,
        )
        result = downloader.download_data(
            variables=["2m_temperature"],
            years=[2020],
            months=list(range(1, 13)),
            file_period="month",
            dry_run=True,
        )
        self.assertEqual(result.skipped_files, [legacy_file])


if __name__ == "__main__":
    unittest.main()
