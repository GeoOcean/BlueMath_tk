"""
Project: BlueMath_tk
Sub-Module: waves.hywaves
Author: GeoOcean Research Group, Universidad de Cantabria
Repository: https://github.com/GeoOcean/BlueMath_tk.git
Status: Under development (Working)

HyWaves: hybrid downscaling of offshore waves to the nearshore. Stationary
SnapWave runs from each offshore *goal* (see :mod:`.sectors`) train a
metamodel; nearshore conditions at the output points (see
:mod:`.output_points`) are reconstructed by linear summation of the goal
contributions.
"""

from . import metamodel, output_points, reconstruction, sectors, wind_correction

__all__ = [
    "metamodel",
    "output_points",
    "reconstruction",
    "sectors",
    "wind_correction",
]
