"""
Project: BlueMath_tk
Sub-Module: core
Author: GeoOcean Research Group, Universidad de Cantabria
Creation Date: 9 December 2024
Repository: https://github.com/GeoOcean/BlueMath_tk.git
Status: Under development (Working)
"""

# Import essential functions/classes to be available at the package level.
from .dask import setup_dask_client
from .operations import (
    convert_lonlat_to_utm,
    convert_utm_to_lonlat,
    denormalize,
    destandarize,
    get_degrees_from_uv,
    get_uv_components,
    mathematical_to_nautical,
    nautical_to_mathematical,
    normalize,
    spatial_gradient,
    standarize,
)

# Optionally, define the module's `__all__` variable to control what gets imported when using `from module import *`.
__all__ = [
    "setup_dask_client",
    "convert_lonlat_to_utm",
    "convert_utm_to_lonlat",
    "denormalize",
    "destandarize",
    "get_degrees_from_uv",
    "get_uv_components",
    "mathematical_to_nautical",
    "nautical_to_mathematical",
    "normalize",
    "spatial_gradient",
    "standarize",
]
