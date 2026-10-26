import importlib
import json
import os
import unittest

import bluemath_tk.downloaders as downloaders

DOWNLOADERS_DIR = os.path.dirname(downloaders.__file__)

# provider package -> downloader class
PROVIDERS = {
    "copernicus": "CopernicusDownloader",
    "csiro": "CSIRODownloader",
    "ecmwf": "ECMWFDownloader",
    "noaa": "NOAADownloader",
}


def _files(path):
    return sorted(
        name
        for name in os.listdir(path)
        if name != "__pycache__" and not name.startswith(".")
    )


class TestDownloadersLayout(unittest.TestCase):
    """
    Each provider folder holds only __init__.py, <provider>_downloader.py and one
    folder per product with <PRODUCT>_config.json.
    """

    def test_providers_are_known(self):
        providers = [
            name
            for name in _files(DOWNLOADERS_DIR)
            if os.path.isdir(os.path.join(DOWNLOADERS_DIR, name))
        ]
        self.assertEqual(sorted(providers), sorted(PROVIDERS))

    def test_provider_folders(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                path = os.path.join(DOWNLOADERS_DIR, provider)
                files = [
                    name
                    for name in _files(path)
                    if os.path.isfile(os.path.join(path, name))
                ]
                self.assertEqual(
                    files, sorted(["__init__.py", f"{provider}_downloader.py"])
                )
                for product in self._product_dirs(provider):
                    self.assertEqual(
                        _files(os.path.join(path, product)),
                        [f"{product}_config.json"],
                    )
                    with open(
                        os.path.join(path, product, f"{product}_config.json")
                    ) as f:
                        self.assertIsInstance(json.load(f), dict)

    def test_products_match_config_folders(self):
        for provider, class_name in PROVIDERS.items():
            with self.subTest(provider=provider):
                try:
                    module = importlib.import_module(
                        f"bluemath_tk.downloaders.{provider}"
                    )
                except ImportError as e:
                    self.skipTest(f"optional dependency missing: {e}")
                downloader_class = getattr(module, class_name)
                self.assertEqual(
                    sorted(downloader_class.list_products()),
                    self._product_dirs(provider),
                )

    @staticmethod
    def _product_dirs(provider):
        path = os.path.join(DOWNLOADERS_DIR, provider)
        return [
            name for name in _files(path) if os.path.isdir(os.path.join(path, name))
        ]


if __name__ == "__main__":
    unittest.main()
