"""
Project: BlueMath_tk
Sub-Module: downloaders
Author: GeoOcean Research Group, Universidad de Cantabria
Repository: https://github.com/GeoOcean/BlueMath_tk.git
Status: Under development (Working)

Downloaders for climate and ocean data providers.

Each provider has its own subpackage with one downloader class and one folder
per product holding the product configuration::

    downloaders/<provider>/<provider>_downloader.py
    downloaders/<provider>/<PRODUCT>/<PRODUCT>_config.json

Providers are imported from their subpackage, so that their optional
dependencies are only needed when used, e.g.::

    from bluemath_tk.downloaders.csiro import CSIRODownloader
    from bluemath_tk.downloaders.copernicus import CopernicusDownloader
"""

from ._base_downloaders import BaseDownloader
from ._download_result import DownloadResult

__all__ = ["DownloadResult", "BaseDownloader"]
