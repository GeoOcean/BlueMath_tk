import unittest

import numpy as np
import pandas as pd
import xarray as xr

from bluemath_tk.tcs.vortex import vortex_model_grid


class TestVortexModelGrid(unittest.TestCase):
    """Test the vortex_model_grid function."""

    def test_vortex_model_grid(self):
        storm_track = pd.DataFrame(
            {
                "vfx": [10, 12],
                "vfy": [5, 6],
                "p0": [1000, 990],
                "pn": [980, 970],
                "vmax": [50, 55],
                "rmw": [30, 35],
                "lon": [10.001, 12.001],
                "lat": [20.001, 22.001],
            },
            index=pd.date_range("2023-10-01", periods=2),
        )
        cg_lon = np.array([9.5, 10.0, 10.5])
        cg_lat = np.array([19.5, 20.0, 20.5])

        ds = vortex_model_grid(storm_track, cg_lon, cg_lat, coords_mode="SPHERICAL")

        W_vals = np.array(
            [
                [
                    [17.09417413, 0.82665737],
                    [22.66057334, 1.14495022],
                    [19.54808437, 1.54414607],
                ],
                [[15.94360075, 1.1403561], [0.0, 1.68962993], [20.625051, 2.44988633]],
                [
                    [10.76028132, 1.52098785],
                    [12.98617365, 2.42132863],
                    [14.33530855, 3.80841364],
                ],
            ]
        )
        Dir_vals = np.array(
            [
                [
                    [1.29496987e02, 1.29942357e02],
                    [9.01036600e01, 1.19620263e02],
                    [5.08422102e01, 1.11400370e02],
                ],
                [
                    [1.79763735e02, 1.41716653e02],
                    [1.34656325e02, 1.30581118e02],
                    [2.39753156e-01, 1.19617666e02],
                ],
                [
                    [2.30162978e02, 1.52609284e02],
                    [2.69894691e02, 1.43594058e02],
                    [3.09495882e02, 1.32127127e02],
                ],
            ]
        )
        p_vals = np.array(
            [
                [
                    [98722.50246466, 97023.7119469],
                    [99257.26417213, 97029.78898728],
                    [98725.18999552, 97036.80829455],
                ],
                [
                    [99371.28980686, 97030.96909655],
                    [100000.0, 97041.30637199],
                    [99378.55955678, 97054.71911504],
                ],
                [
                    [98727.68758907, 97040.02996857],
                    [99264.62166108, 97057.51709726],
                    [98730.39083852, 97083.99580787],
                ],
            ]
        )
        lat = np.array([19.5, 20.0, 20.5])
        lon = np.array([9.5, 10.0, 10.5])
        time = np.array(
            ["2023-10-01T00:00:00.000000000", "2023-10-02T00:00:00.000000000"],
            dtype="datetime64[ns]",
        )

        ds_expected = xr.Dataset(
            {
                "W": (["lat", "lon", "time"], W_vals, {"units": "m/s"}),
                "Dir": (["lat", "lon", "time"], Dir_vals, {"units": "º"}),
                "p": (["lat", "lon", "time"], p_vals, {"units": "Pa"}),
            },
            coords={"lat": lat, "lon": lon, "time": time},
        )

        xr.testing.assert_allclose(ds, ds_expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
