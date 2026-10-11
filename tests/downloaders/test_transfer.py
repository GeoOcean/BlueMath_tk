import os
import tempfile
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import xarray as xr

from bluemath_tk.downloaders import BaseDownloader, DownloadResult
from bluemath_tk.downloaders._transfer import (
    RemoteFileNotFound,
    http_download,
    make_session,
    remote_size,
    retry_call,
    validate_file,
)


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """Static file handler with support for "Range: bytes=N-" requests."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        range_header = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not range_header or not os.path.isfile(path):
            return super().do_GET()
        start = int(range_header.split("=")[1].split("-")[0])
        with open(path, "rb") as f:
            data = f.read()[start:]
        self.send_response(206)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class LocalServerTestCase(unittest.TestCase):
    """Serve a temporary directory over HTTP during the tests."""

    @classmethod
    def setUpClass(cls):
        cls.served_dir = tempfile.TemporaryDirectory()
        handler = partial(RangeRequestHandler, directory=cls.served_dir.name)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.content = os.urandom(3 * 1024 * 1024 + 123)
        with open(os.path.join(cls.served_dir.name, "data.bin"), "wb") as f:
            f.write(cls.content)
        open(os.path.join(cls.served_dir.name, "empty.bin"), "wb").close()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.served_dir.cleanup()

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.session = make_session(retries=0)

    def tearDown(self):
        self.tmp_dir.cleanup()


class TestHttpDownload(LocalServerTestCase):
    def test_full_download(self):
        target = os.path.join(self.tmp_dir.name, "data.bin")
        url = f"{self.base_url}/data.bin"
        http_download(self.session, url, target, expected_size=len(self.content))
        with open(target, "rb") as f:
            self.assertEqual(f.read(), self.content)

    def test_resume_partial_file(self):
        target = os.path.join(self.tmp_dir.name, "data.bin")
        with open(target, "wb") as f:
            f.write(self.content[:1000])
        http_download(
            self.session,
            f"{self.base_url}/data.bin",
            target,
            expected_size=len(self.content),
        )
        with open(target, "rb") as f:
            self.assertEqual(f.read(), self.content)

    def test_size_mismatch_raises(self):
        target = os.path.join(self.tmp_dir.name, "data.bin")
        with self.assertRaises(IOError):
            http_download(
                self.session, f"{self.base_url}/data.bin", target, expected_size=10
            )

    def test_missing_file(self):
        url = f"{self.base_url}/missing.bin"
        with self.assertRaises(RemoteFileNotFound):
            remote_size(self.session, url)
        with self.assertRaises(RemoteFileNotFound):
            http_download(self.session, url, os.path.join(self.tmp_dir.name, "x"))

    def test_empty_remote_file_is_missing(self):
        with self.assertRaises(RemoteFileNotFound):
            remote_size(self.session, f"{self.base_url}/empty.bin")

    def test_remote_size(self):
        size = remote_size(self.session, f"{self.base_url}/data.bin")
        self.assertEqual(size, len(self.content))


class TestRetryCall(unittest.TestCase):
    def test_succeeds_after_failures(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("temporary")
            return "ok"

        self.assertEqual(retry_call(flaky, retries=3, backoff=0), "ok")
        self.assertEqual(len(calls), 3)

    def test_gives_up(self):
        def failing():
            raise ConnectionError("permanent")

        with self.assertRaises(ConnectionError):
            retry_call(failing, retries=2, backoff=0)

    def test_no_retry_for_missing_files(self):
        calls = []

        def missing():
            calls.append(1)
            raise RemoteFileNotFound("missing")

        with self.assertRaises(RemoteFileNotFound):
            retry_call(missing, retries=3, backoff=0)
        self.assertEqual(len(calls), 1)


class TestValidateFile(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _write_netcdf(self) -> str:
        path = os.path.join(self.tmp_dir.name, "data.nc")
        xr.Dataset({"hs": ("time", np.arange(1000.0))}).to_netcdf(
            path, engine="netcdf4"
        )
        return path

    def test_valid_netcdf(self):
        validate_file(self._write_netcdf())

    def test_truncated_netcdf(self):
        path = self._write_netcdf()
        with open(path, "rb") as f:
            data = f.read()
        with open(path, "wb") as f:
            f.write(data[: len(data) // 2])
        with self.assertRaises(IOError):
            validate_file(path)

    def test_empty_file(self):
        path = os.path.join(self.tmp_dir.name, "empty.nc")
        open(path, "wb").close()
        with self.assertRaises(IOError):
            validate_file(path)

    def test_grib_end_marker(self):
        path = os.path.join(self.tmp_dir.name, "data.grib")
        with open(path, "wb") as f:
            f.write(b"GRIB" + b"\x00" * 100 + b"7777")
        validate_file(path)
        with open(path, "wb") as f:
            f.write(b"GRIB" + b"\x00" * 100)
        with self.assertRaises(IOError):
            validate_file(path)


class DummyDownloader(BaseDownloader):
    products_configs = {
        "MyProduct": {"datasets": {"a": {}, "b": {}}},
        "Other": {"datasets": {}},
    }

    def __init__(self, base_path_to_download: str, product: str = "MyProduct"):
        super().__init__(
            product=product,
            base_path_to_download=base_path_to_download,
            retries=2,
            retry_backoff=0,
        )

    def download_data_my_product(self, value, dry_run=True):
        return value, dry_run


class TestFetchFile(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.downloader = DummyDownloader(self.tmp_dir.name)
        self.output_file = os.path.join(self.tmp_dir.name, "sub", "file.txt")
        self.result = DownloadResult()

    def tearDown(self):
        self.tmp_dir.cleanup()

    @staticmethod
    def _write(part_file: str) -> None:
        with open(part_file, "w") as f:
            f.write("data")

    def test_download(self):
        self.downloader.fetch_file(
            self.output_file, self._write, self.result, dry_run=False
        )
        self.assertEqual(self.result.downloaded_files, [self.output_file])
        self.assertTrue(os.path.exists(self.output_file))
        self.assertFalse(os.path.exists(self.output_file + ".part"))

    def test_existing_file_is_skipped(self):
        os.makedirs(os.path.dirname(self.output_file))
        self._write(self.output_file)
        self.downloader.fetch_file(
            self.output_file, self._write, self.result, dry_run=False
        )
        self.assertEqual(self.result.skipped_files, [self.output_file])
        self.assertEqual(self.result.downloaded_files, [])

    def test_dry_run_writes_nothing(self):
        self.downloader.fetch_file(self.output_file, self._write, self.result)
        self.assertEqual(self.result.skipped_files, [self.output_file])
        self.assertFalse(os.path.exists(os.path.dirname(self.output_file)))

    def test_failure_leaves_no_file(self):
        calls = []

        def failing(part_file):
            calls.append(1)
            with open(part_file, "w") as f:
                f.write("half")
            raise ConnectionError("interrupted")

        self.downloader.fetch_file(
            self.output_file, failing, self.result, dry_run=False
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.result.error_files, [self.output_file])
        self.assertFalse(os.path.exists(self.output_file))
        self.assertFalse(os.path.exists(self.output_file + ".part"))

    def test_resumable_failure_keeps_part_file(self):
        def failing(part_file):
            with open(part_file, "a") as f:
                f.write("half")
            raise ConnectionError("interrupted")

        self.downloader.fetch_file(
            self.output_file, failing, self.result, dry_run=False, resumable=True
        )
        self.assertFalse(os.path.exists(self.output_file))
        self.assertTrue(os.path.exists(self.output_file + ".part"))

    def test_invalid_file_is_not_kept(self):
        def empty(part_file):
            open(part_file, "w").close()

        self.downloader.fetch_file(self.output_file, empty, self.result, dry_run=False)
        self.assertEqual(self.result.error_files, [self.output_file])
        self.assertFalse(os.path.exists(self.output_file))
        self.assertFalse(os.path.exists(self.output_file + ".part"))

    def test_missing_remote_file_is_skipped(self):
        def missing(part_file):
            raise RemoteFileNotFound("http://example.com/file.txt")

        self.downloader.fetch_file(
            self.output_file, missing, self.result, dry_run=False
        )
        self.assertEqual(self.result.skipped_files, [self.output_file])
        self.assertEqual(self.result.error_files, [])


class TestBaseDownloader(unittest.TestCase):
    def test_routing_to_product_method(self):
        downloader = DummyDownloader(tempfile.gettempdir())
        self.assertEqual(downloader.download_data(1), (1, True))
        self.assertEqual(downloader.download_data(2, dry_run=False), (2, False))
        self.assertEqual(downloader.list_datasets(), ["a", "b"])

    def test_unknown_product(self):
        with self.assertRaises(ValueError):
            DummyDownloader(tempfile.gettempdir(), product="Missing")
        self.assertEqual(DummyDownloader.list_products(), ["MyProduct", "Other"])

    def test_product_without_download_method(self):
        downloader = DummyDownloader(tempfile.gettempdir(), product="Other")
        with self.assertRaises(ValueError):
            downloader.download_data()


class TestDownloadResult(unittest.TestCase):
    def test_merge_and_save(self):
        result = DownloadResult(downloaded_files=["a"])
        other = DownloadResult(downloaded_files=["a", "b"])
        other.add_skipped("c", "Already downloaded")
        other.add_error("d", ValueError("bad"))
        result.merge(other)
        self.assertEqual(result.downloaded_files, ["a", "b"])
        self.assertEqual(result.skipped_files, ["c"])
        self.assertEqual(result.error_files, ["d"])
        self.assertEqual(result.metadata["skip_reasons"], {"c": "Already downloaded"})

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "logs", "result.json")
            result.save(path)
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
