"""
Project: BlueMath_tk
Module: bluemath_tk
Author: GeoOcean Research Group, Universidad de Cantabria
Repository: https://github.com/GeoOcean/BlueMath_tk.git
Status: Under development (Working)
"""

# Import specific modules instead of using wildcard imports
from . import (
    additive,
    config,
    core,
    datamining,
    distributions,
    downloaders,
    interpolation,
    predictor,
    risk,
    tcs,
    teslakit,
    tide,
    topo_bathy,
    waves,
    wrappers,
)

# Add __all__ variable to control what gets imported when using `from module import *`.
__all__ = [
    "additive",
    "config",
    "core",
    "datamining",
    "distributions",
    "downloaders",
    "interpolation",
    "predictor",
    "risk",
    "tcs",
    "teslakit",
    "tide",
    "topo_bathy",
    "waves",
    "wrappers",
]
