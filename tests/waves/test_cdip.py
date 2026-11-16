from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import xarray as xr

from bluemath_tk.waves.cdip import bulk_from_moments, cdip_hourly, cdip_hourly_spectra, cdip_url, mem_spreading


class TestBulkFromMoments(unittest.TestCase):
    def test_single_band(self):
        freq = np.array([0.05, 0.1, 0.2])
        bw = np.full(3, 0.01)
        energy = np.array([[0.0, 100.0, 0.0]])
        theta = np.radians(200.0)
        r1 = 0.8
        a1 = np.full((1, 3), r1 * np.cos(theta))
        b1 = np.full((1, 3), r1 * np.sin(theta))
        out = bulk_from_moments(energy, a1, b1, freq, bw)
        self.assertAlmostEqual(float(out["hs"][0]), 4.0, places=6)
        self.assertAlmostEqual(float(out["tp"][0]), 10.0, places=6)
        self.assertAlmostEqual(float(out["tm01"][0]), 10.0, places=6)
        self.assertAlmostEqual(float(out["dir"][0]), 200.0, places=6)
        self.assertAlmostEqual(float(out["spr"][0]), np.degrees(np.sqrt(0.4)), places=6)

    def test_mean_period_two_bands(self):
        freq = np.array([0.1, 0.2])
        energy = np.array([[100.0, 100.0]])
        zeros = np.zeros((1, 2))
        out = bulk_from_moments(energy, zeros, zeros, freq, np.full(2, 0.01))
        self.assertAlmostEqual(float(out["tm01"][0]), 2.0 / 0.3, places=6)

    def test_no_energy_is_nan(self):
        out = bulk_from_moments(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), np.array([0.1, 0.2]), np.ones(2))
        self.assertTrue(np.isnan(out["hs"][0]))


class TestMemSpreading(unittest.TestCase):
    def test_reproduces_moments(self):
        dirs = np.arange(0.0, 360.0, 0.5)
        rad = np.radians(dirs)
        moments = np.array([[0.6, 0.3, 0.2, 0.4], [0.7, -0.2, 0.5, -0.3]])
        d = mem_spreading(*moments.T, dirs)
        self.assertEqual(d.shape, (2, 720))
        self.assertTrue(np.all(d >= 0))
        np.testing.assert_allclose(d.sum(axis=1) * 0.5, 1.0)
        back = [(d * f).sum(axis=1) * 0.5 for f in (np.cos(rad), np.sin(rad), np.cos(2 * rad), np.sin(2 * rad))]
        np.testing.assert_allclose(np.array(back).T, moments, atol=1e-4)

    def test_unimodal_peak(self):
        dirs = np.arange(0.0, 360.0, 2.0)
        theta = np.radians(30.0)  # r2 >= r1**2: one peak, at the mean direction
        d = mem_spreading(
            np.array([0.8 * np.cos(theta)]), np.array([0.8 * np.sin(theta)]),
            np.array([0.7 * np.cos(2 * theta)]), np.array([0.7 * np.sin(2 * theta)]), dirs,
        )
        self.assertEqual(float(dirs[np.argmax(d[0])]), 30.0)

    def test_isotropic_is_flat(self):
        dirs = np.arange(0.0, 360.0, 10.0)
        z = np.zeros(1)
        d = mem_spreading(z, z, z, z, dirs)
        np.testing.assert_allclose(d[0], 1.0 / 360.0)


class TestCdipHourly(unittest.TestCase):
    def test_hourly_mean_flags_and_position(self):
        t = pd.date_range("2018-01-01", periods=4, freq="30min")
        freq = np.array([0.1, 0.2])
        energy = np.array([[4.0, 0.0], [12.0, 0.0], [8.0, 0.0], [999.0, 0.0]])
        ds = xr.Dataset(
            {
                "waveFlagPrimary": ("waveTime", [1, 1, 1, 4]),
                "waveEnergyDensity": (("waveTime", "waveFrequency"), energy),
                "waveA1Value": (("waveTime", "waveFrequency"), np.full((4, 2), -0.5)),
                "waveB1Value": (("waveTime", "waveFrequency"), np.zeros((4, 2))),
                "waveBandwidth": ("waveFrequency", [1.0, 1.0]),
                "gpsLatitude": ("gpsTime", [-14.0, -14.5]),
                "gpsLongitude": ("gpsTime", [-170.0, -170.5]),
            },
            coords={"waveTime": t, "waveFrequency": freq, "gpsTime": t[[0, 2]]},
        )
        out = cdip_hourly(ds)
        self.assertEqual(out.sizes["time"], 2)
        np.testing.assert_allclose(out["hs"].values, [4 * np.sqrt(8.0), 4 * np.sqrt(8.0)])
        np.testing.assert_allclose(out["n_records"].values, [2, 1])
        np.testing.assert_allclose(out["dir"].values, [180.0, 180.0])
        np.testing.assert_allclose(out["latitude"].values, [-14.0, -14.5])

    def test_spectrum_keeps_energy(self):
        t = pd.date_range("2018-01-01", periods=2, freq="30min")
        freq = np.array([0.1, 0.2])
        theta = np.radians(250.0)  # r1 0.8, r2 0.7: unimodal
        ds = xr.Dataset(
            {
                "waveFlagPrimary": ("waveTime", [1, 1]),
                "waveEnergyDensity": (("waveTime", "waveFrequency"), [[4.0, 1.0], [6.0, 1.0]]),
                "waveA1Value": (("waveTime", "waveFrequency"), np.full((2, 2), 0.8 * np.cos(theta))),
                "waveB1Value": (("waveTime", "waveFrequency"), np.full((2, 2), 0.8 * np.sin(theta))),
                "waveA2Value": (("waveTime", "waveFrequency"), np.full((2, 2), 0.7 * np.cos(2 * theta))),
                "waveB2Value": (("waveTime", "waveFrequency"), np.full((2, 2), 0.7 * np.sin(2 * theta))),
                "waveBandwidth": ("waveFrequency", [0.01, 0.01]),
                "gpsLatitude": ("gpsTime", [-14.0]),
                "gpsLongitude": ("gpsTime", [-170.0]),
            },
            coords={"waveTime": t, "waveFrequency": freq, "gpsTime": t[[0]]},
        )
        bulk = cdip_hourly(ds)
        out = cdip_hourly_spectra(ds, np.arange(0.0, 360.0, 5.0))
        self.assertEqual(out["efth"].dims, ("time", "freq", "dir"))
        np.testing.assert_array_equal(out.time.values, bulk.time.values)
        m0 = float((out["efth"].sum("dir") * 5.0 * 0.01).sum())
        self.assertAlmostEqual(4 * np.sqrt(m0), float(bulk["hs"][0]), places=4)
        self.assertEqual(float(out["efth"].isel(time=0, freq=0).idxmax("dir")), 250.0)
        self.assertAlmostEqual(float(bulk["dir"][0]), 250.0, places=6)  # bulk dir is the mean direction, not the spectrum axis

    def test_urls(self):
        self.assertTrue(cdip_url(189).endswith("archive/189p1/189p1_historic.nc"))
        self.assertTrue(cdip_url("273", realtime=True).endswith("realtime/273p1_rt.nc"))


if __name__ == "__main__":
    unittest.main()
