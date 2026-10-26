import hashlib
import os
import unittest
from collections import namedtuple
from unittest import mock

import pandas as pd
from test_transfer import LocalServerTestCase

from bluemath_tk.downloaders.imos import imos_downloader
from bluemath_tk.downloaders.imos.imos_downloader import (
    FILE_COLUMNS,
    IMOSDownloader,
    parse_listing,
)

DATASET = "Wave-Wind-Altimetry-DM00"
PREFIX = "IMOS/SRS/Surface-Waves/Wave-Wind-Altimetry-DM00/"
File = namedtuple("File", FILE_COLUMNS)


def cell_name(mission, cell="044N-356E"):
    return f"IMOS_SRS-Surface-Waves_MW_{mission}_FV02_{cell}-DM00.nc"


def listing(keys, prefixes=(), token=None):
    """S3 ListObjectsV2 answer with the given keys (size 100, 2026-07-14)."""
    contents = "".join(
        f"<Contents><Key>{k}</Key><LastModified>2026-07-14T10:06:44.000Z</LastModified>"
        f"<ETag>&quot;{'a' * 32}&quot;</ETag><Size>100</Size></Contents>"
        for k in keys
    )
    common = "".join(
        f"<CommonPrefixes><Prefix>{p}</Prefix></CommonPrefixes>" for p in prefixes
    )
    tail = (
        f"<IsTruncated>true</IsTruncated><NextContinuationToken>{token}</NextContinuationToken>"
        if token
        else "<IsTruncated>false</IsTruncated>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"{contents}{common}{tail}</ListBucketResult>"
    ).encode()


class FakeResponse:
    def __init__(self, content):
        self.content = content


class TestParseListing(unittest.TestCase):
    def test_objects_prefixes_and_token(self):
        key = PREFIX + "SWOT/040N_340E/" + cell_name("SWOT")
        objects, prefixes, token = parse_listing(
            listing([key], prefixes=[PREFIX + "SWOT/"], token="abc")
        )
        self.assertEqual(token, "abc")
        self.assertEqual(prefixes, [PREFIX + "SWOT/"])
        self.assertEqual(objects[0]["key"], key)
        self.assertEqual(objects[0]["size"], 100)
        self.assertEqual(objects[0]["etag"], "a" * 32)
        self.assertEqual(
            objects[0]["last_modified"], pd.Timestamp("2026-07-14T10:06:44Z")
        )

    def test_last_page(self):
        self.assertIsNone(parse_listing(listing([]))[2])


class TestListFiles(unittest.TestCase):
    def setUp(self):
        self.downloader = IMOSDownloader(
            product="SRS", base_path_to_download="/data", retries=1, retry_backoff=0
        )

    def test_unknown_dataset(self):
        with self.assertRaises(ValueError):
            self.downloader.list_files("nope")

    def test_subfolders(self):
        answer = listing([], prefixes=[PREFIX + "JASON-3/", PREFIX + "SWOT/"])
        with mock.patch.object(
            imos_downloader, "http_get", lambda *a, **k: FakeResponse(answer)
        ):
            self.assertEqual(
                self.downloader.list_subfolders(DATASET), ["JASON-3", "SWOT"]
            )

    def test_paging_pattern_and_modified_since(self):
        swot = PREFIX + "SWOT/040N_340E/"
        pages = {
            None: listing([swot + cell_name("SWOT", "041N-341E")], token="t1"),
            "t1": listing([swot + cell_name("SWOT", "040N-340E"), swot + "README.txt"]),
        }
        calls = []

        def fake_get(session, url, params=None):
            calls.append(dict(params))
            return FakeResponse(pages[params.get("continuation-token")])

        with mock.patch.object(imos_downloader, "http_get", fake_get):
            files = self.downloader.list_files(DATASET, subfolders=["SWOT"])
            newer = self.downloader.list_files(
                DATASET, subfolders=["SWOT"], modified_since="2026-08-01"
            )

        self.assertEqual(list(files.columns), FILE_COLUMNS)
        self.assertEqual(
            list(files["path"]),
            [
                "SWOT/040N_340E/" + cell_name("SWOT", "040N-340E"),
                "SWOT/040N_340E/" + cell_name("SWOT", "041N-341E"),
            ],
        )
        self.assertTrue(files["url"].iloc[0].startswith("https://imos-data.s3"))
        self.assertEqual(calls[0]["prefix"], PREFIX + "SWOT/")
        self.assertNotIn("continuation-token", calls[0])
        self.assertEqual(calls[1]["continuation-token"], "t1")
        self.assertTrue(newer.empty)

    def test_local_path_mirrors_the_bucket(self):
        self.assertEqual(
            self.downloader.get_local_path(DATASET, "SWOT/040N_340E/f.nc"),
            os.path.normpath(
                "/data/SRS/Surface-Waves/Wave-Wind-Altimetry-DM00/SWOT/040N_340E/f.nc"
            ),
        )


class TestIMOSDownloadFile(LocalServerTestCase):
    def setUp(self):
        super().setUp()
        self.downloader = IMOSDownloader(
            product="SRS",
            base_path_to_download=self.tmp_dir.name,
            retries=1,
            retry_backoff=0,
        )
        self.output_file = os.path.join(self.tmp_dir.name, "sub", "data.bin")
        self.modified = pd.Timestamp("2026-07-14T10:06:44Z")

    def file(self, etag=None, modified=None):
        return File(
            key="data.bin",
            path="data.bin",
            url=f"{self.base_url}/data.bin",
            size=len(self.content),
            etag=hashlib.md5(self.content).hexdigest() if etag is None else etag,
            last_modified=self.modified if modified is None else modified,
        )

    def download(self, row, dry_run=False):
        return self.downloader._download_file(
            row, self.output_file, force=False, verify_existing=True, dry_run=dry_run
        )

    def test_download_sets_bucket_time_and_skips(self):
        result = self.download(self.file())
        self.assertEqual(result.downloaded_files, [self.output_file])
        self.assertEqual(os.path.getmtime(self.output_file), self.modified.timestamp())
        result = self.download(self.file())
        self.assertEqual(result.skipped_files, [self.output_file])

    def test_newer_in_bucket_is_downloaded_again(self):
        self.download(self.file())
        newer = self.file(modified=self.modified + pd.Timedelta(days=30))
        result = self.download(newer, dry_run=True)
        self.assertIn(
            "Updated in the bucket", result.metadata["skip_reasons"][self.output_file]
        )
        result = self.download(newer)
        self.assertEqual(result.downloaded_files, [self.output_file])
        self.assertEqual(
            os.path.getmtime(self.output_file),
            (self.modified + pd.Timedelta(days=30)).timestamp(),
        )

    def test_wrong_md5_is_an_error(self):
        result = self.download(self.file(etag="0" * 32))
        self.assertEqual(result.error_files, [self.output_file])
        self.assertFalse(os.path.exists(self.output_file))
        self.assertFalse(os.path.exists(self.output_file + ".part"))

    def test_multipart_etag_is_not_a_checksum(self):
        result = self.download(self.file(etag="0" * 32 + "-3"))
        self.assertEqual(result.downloaded_files, [self.output_file])


if __name__ == "__main__":
    unittest.main()
