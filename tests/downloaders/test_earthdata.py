import hashlib
import os
import tempfile
import unittest
from collections import namedtuple
from datetime import datetime
from unittest import mock

from test_transfer import LocalServerTestCase

from bluemath_tk.downloaders.earthdata import earthdata_downloader
from bluemath_tk.downloaders.earthdata.earthdata_downloader import (
    GRANULE_COLUMNS,
    EarthdataDownloader,
    cmr_time_range,
    parse_granule,
)

DATASET = "L2_LR_SSH_WINDWAVE_D"
ARCHIVE = "https://archive.example/protected/SWOT_L2_LR_SSH_D"
Granule = namedtuple("Granule", GRANULE_COLUMNS)


def swot_name(product="WindWave", cycle_pass="016_085", times=None):
    times = times or "20240601T045132_20240601T054300"
    return f"SWOT_L2_LR_SSH_{product}_{cycle_pass}_{times}_PGD0_01.nc"


def cmr_item(name, start="2024-06-01T04:51:17.972Z", size=1000, md5="abc"):
    """Minimal CMR UMM-JSON item, with the links of a real SWOT granule."""
    return {
        "meta": {"concept-id": "G1-POCLOUD", "revision-date": "2026-03-11T04:39:38Z"},
        "umm": {
            "GranuleUR": name.replace(".nc", "_swot"),
            "TemporalExtent": {
                "RangeDateTime": {
                    "BeginningDateTime": start,
                    "EndingDateTime": "2024-06-01T05:43:15.472Z",
                }
            },
            "DataGranule": {
                "ArchiveAndDistributionInformation": [
                    {"Name": name.replace(".nc", ".log"), "SizeInBytes": 5},
                    {
                        "Name": name,
                        "SizeInBytes": size,
                        "Checksum": {"Value": md5, "Algorithm": "MD5"},
                    },
                ]
            },
            "RelatedUrls": [
                {"Type": "EXTENDED METADATA", "URL": f"{ARCHIVE}/{name}.iso.xml"},
                {"Type": "GET DATA", "URL": f"{ARCHIVE}/{name}"},
                {"Type": "GET DATA VIA DIRECT ACCESS", "URL": f"s3://bucket/{name}"},
            ],
        },
    }


class FakeResponse:
    def __init__(self, items, search_after=None):
        self._items = items
        self.headers = {"CMR-Search-After": search_after} if search_after else {}

    def json(self):
        return {"items": self._items}


class TestCmrTimeRange(unittest.TestCase):
    def test_resolution_of_strings(self):
        self.assertEqual(
            cmr_time_range("2024-06", "2024-06"),
            "2024-06-01T00:00:00Z,2024-06-30T23:59:59Z",
        )
        self.assertEqual(
            cmr_time_range("2024-02-29", "2024-02-29"),
            "2024-02-29T00:00:00Z,2024-02-29T23:59:59Z",
        )

    def test_open_sides_and_datetimes(self):
        self.assertEqual(cmr_time_range("2024", None), "2024-01-01T00:00:00Z,")
        self.assertEqual(
            cmr_time_range(None, datetime(2024, 1, 2, 3, 4, 5)),
            ",2024-01-02T03:04:05Z",
        )


class TestParseGranule(unittest.TestCase):
    def test_only_https_data_files(self):
        name = swot_name()
        rows = parse_granule(cmr_item(name, size=12428786, md5="c1658"))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(sorted(row), sorted(GRANULE_COLUMNS))
        self.assertEqual(row["filename"], name)
        self.assertEqual(row["url"], f"{ARCHIVE}/{name}")
        self.assertEqual(row["size"], 12428786)
        self.assertEqual((row["checksum_algorithm"], row["checksum"]), ("MD5", "c1658"))
        self.assertEqual(row["start_time"], "2024-06-01T04:51:17.972Z")

    def test_file_not_listed_in_data_granule(self):
        item = cmr_item("a.nc")
        item["umm"]["DataGranule"]["ArchiveAndDistributionInformation"] = []
        row = parse_granule(item)[0]
        self.assertIsNone(row["size"])
        self.assertIsNone(row["checksum"])


class TestSearchGranules(unittest.TestCase):
    def setUp(self):
        self.downloader = EarthdataDownloader(
            product="SWOT", base_path_to_download="/data", retries=1, retry_backoff=0
        )

    def test_unknown_dataset(self):
        with self.assertRaises(ValueError):
            self.downloader.search_granules("nope")

    def test_paging_filters_and_pattern(self):
        late = swot_name(cycle_pass="016_086", times="20240601T054300_20240601T063400")
        early = swot_name()
        stray = swot_name(product="Expert")
        pages = [
            FakeResponse([cmr_item(late, start="2024-06-01T05:43:00Z")], "[1,2]"),
            FakeResponse([cmr_item(early), cmr_item(stray)]),
        ]
        calls = []

        def fake_get(session, url, params=None, headers=None):
            calls.append((dict(params), dict(headers)))
            return pages[len(calls) - 1]

        with mock.patch.object(earthdata_downloader, "http_get", fake_get):
            granules = self.downloader.search_granules(
                DATASET,
                start="2024-06",
                end="2024-06",
                bbox=(2.5, 51, 7.5, 54),
                updated_since="2026-10-01",
            )

        self.assertEqual(list(granules["filename"]), [early, late])
        params, headers = calls[0]
        self.assertEqual(params["short_name"], "SWOT_L2_LR_SSH_WINDWAVE_D")
        self.assertEqual(
            params["temporal"], "2024-06-01T00:00:00Z,2024-06-30T23:59:59Z"
        )
        self.assertEqual(params["bounding_box"], "2.5,51.0,7.5,54.0")
        self.assertEqual(params["updated_since"], "2026-10-01T00:00:00Z")
        self.assertEqual(headers, {})
        self.assertEqual(calls[1][1], {"CMR-Search-After": "[1,2]"})

    def test_default_start_is_dataset_start(self):
        calls = []

        def fake_get(session, url, params=None, headers=None):
            calls.append(dict(params))
            return FakeResponse([])

        with mock.patch.object(earthdata_downloader, "http_get", fake_get):
            granules = self.downloader.search_granules(DATASET)
        self.assertEqual(calls[0]["temporal"], "2023-03-01T00:00:00Z,")
        self.assertTrue(granules.empty)

    def test_local_path_uses_month_of_start(self):
        self.assertEqual(
            self.downloader.get_local_path(DATASET, "f.nc", "2024-05-31T23:58:00Z"),
            os.path.normpath("/data/SWOT/L2_LR_SSH_WINDWAVE_D/2024-05/f.nc"),
        )


class TestCredentials(unittest.TestCase):
    def check_with_netrc(self, content):
        with tempfile.TemporaryDirectory() as home:
            path = os.path.join(home, ".netrc")
            with open(path, "w") as f:
                f.write(content)
            os.chmod(path, 0o600)
            # "~" is HOME on POSIX and USERPROFILE on Windows
            with mock.patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
                EarthdataDownloader.check_credentials()

    def test_entry_found(self):
        self.check_with_netrc(
            "machine urs.earthdata.nasa.gov login user password secret\n"
        )

    def test_entry_missing(self):
        with self.assertRaises(ValueError):
            self.check_with_netrc("machine other.host login user password secret\n")


class TestEarthdataDownloadFile(LocalServerTestCase):
    def setUp(self):
        super().setUp()
        self.downloader = EarthdataDownloader(
            product="SWOT",
            base_path_to_download=self.tmp_dir.name,
            retries=1,
            retry_backoff=0,
        )
        self.output_file = os.path.join(self.tmp_dir.name, "data.bin")
        self.md5 = hashlib.md5(self.content).hexdigest()

    def granule(self, size=None, checksum=None):
        return Granule(
            granule_ur="data",
            filename="data.bin",
            url=f"{self.base_url}/data.bin",
            size=len(self.content) if size is None else size,
            checksum_algorithm="MD5",
            checksum=self.md5 if checksum is None else checksum,
            start_time="2024-06-01T00:00:00Z",
            end_time="2024-06-01T01:00:00Z",
            revision_date="2026-01-01T00:00:00Z",
        )

    def download(self, granule, dry_run=False):
        return self.downloader._download_file(
            granule,
            self.output_file,
            force=False,
            verify_existing=True,
            dry_run=dry_run,
        )

    def test_download_and_skip(self):
        result = self.download(self.granule())
        self.assertEqual(result.downloaded_files, [self.output_file])
        with open(self.output_file, "rb") as f:
            self.assertEqual(f.read(), self.content)
        self.assertEqual(
            result.metadata["remote_sizes"], {self.output_file: len(self.content)}
        )

        result = self.download(self.granule())
        self.assertEqual(result.skipped_files, [self.output_file])

    def test_wrong_checksum_is_an_error(self):
        result = self.download(self.granule(checksum="0" * 32))
        self.assertEqual(result.error_files, [self.output_file])
        self.assertFalse(os.path.exists(self.output_file))
        self.assertFalse(os.path.exists(self.output_file + ".part"))

    def test_incomplete_local_file(self):
        with open(self.output_file, "wb") as f:
            f.write(self.content[:100])

        result = self.download(self.granule(), dry_run=True)
        self.assertIn(
            "would download again", result.metadata["skip_reasons"][self.output_file]
        )

        result = self.download(self.granule())
        self.assertEqual(result.downloaded_files, [self.output_file])
        self.assertEqual(os.path.getsize(self.output_file), len(self.content))


if __name__ == "__main__":
    unittest.main()
