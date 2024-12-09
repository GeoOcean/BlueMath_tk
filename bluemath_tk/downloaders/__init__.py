"""
Project: BlueMath_tk
Sub-Module: downloaders
Author: GeoOcean Research Group, Universidad de Cantabria
Creation Date: 9 August 2024
License: MIT
Repository: https://github.com/GeoOcean/BlueMath_tk.git
Status: Under development (Working)
"""

# Import essential functions/classes to be available at the package level.
from ._base_downloaders import BaseDownloader
from .copernicus.copernicus_downloader import CopernicusDownloader

# Optionally, define the module's `__all__` variable to control what gets imported when using `from module import *`.
__all__ = [
    "BaseDownloader",
    "CopernicusDownloader",
]
