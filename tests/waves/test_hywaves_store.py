import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from bluemath_tk.waves.hywaves.store import ReconstructionStore

# hourly, 2 Jan -> 3 Feb 2020, with the nanosecond jitter real hindcasts carry
TIMES = pd.date_range("2020-01-02", "2020-02-03 23:00", freq="h") + pd.to_timedelta(
    np.tile([0, 1], 792), "ns"
)[: len(pd.date_range("2020-01-02", "2020-02-03 23:00", freq="h"))]
SITES = ["a", "b", "c"]


def _bulk(times, sites, value):
    return xr.Dataset(
        {"hs": (("time", "sites"), np.full((len(times), len(sites)), value))},
        coords={"time": times, "sites": sites},
    )


class TestReconstructionStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "s.zarr"

    def store(self, **kw):
        return ReconstructionStore(self.path, TIMES, SITES, {"hs": ()}, key={"n": 1}, fingerprint="f1", **kw)

    def test_months_missing_write_read(self):
        store = self.store()
        self.assertEqual(store.months, ["2020-01", "2020-02"])
        self.assertEqual(store.months_of("2020-01-31"), ["2020-01"])
        self.assertEqual(store.missing(slice("2020-01-31", "2020-02-01"), ["b"]),
                         {"2020-01": ["b"], "2020-02": ["b"]})
        jan = TIMES[TIMES.month == 1]
        store.write(_bulk(jan[:-5], ["a", "c"], 2.0), ["2020-01"])  # last 5 h lacking -> NaN
        self.assertEqual(store.missing("2020-01", SITES), {"2020-01": ["b"]})
        day = store.read("2020-01-10", ["c"])
        self.assertEqual(day.sizes["time"], 24)
        self.assertTrue((day.hs == 2.0).all())
        tail = store.read("2020-01-31", ["a"])
        self.assertEqual(int(tail.hs.isnull().sum()), 5)
        with self.assertRaises(KeyError):
            store.read("2020-01", ["b"])
        self.assertEqual(store.coverage().loc["2020-01", "sites"], 2)

    def test_reopen_keeps_and_mismatch_rebuilds(self):
        self.store().write(_bulk(TIMES[TIMES.month == 2], SITES, 1.0), ["2020-02"])
        self.assertEqual(self.store().missing("2020-02", SITES), {})
        changed = ReconstructionStore(self.path, TIMES, SITES, {"hs": ()}, key={"n": 1}, fingerprint="f2")
        self.assertEqual(changed.missing("2020-02", ["a"]), {"2020-02": ["a"]})

    def test_invalidate(self):
        store = self.store()
        store.write(_bulk(TIMES[TIMES.month == 2], SITES, 1.0), ["2020-02"])
        store.invalidate("2020-02-01", ["b"])
        self.assertEqual(store.missing("2020-02", SITES), {"2020-02": ["b"]})

    def test_spectra_variable(self):
        freq, dirs = np.array([0.1, 0.2]), np.array([0.0, 90.0, 180.0])
        store = ReconstructionStore(
            self.path, TIMES, SITES, {"efth": ("freq", "dir")}, coords={"freq": freq, "dir": dirs}
        )
        feb = TIMES[TIMES.month == 2]
        ds = xr.Dataset(
            {"efth": (("time", "sites", "freq", "dir"), np.ones((len(feb), 2, 2, 3)))},
            coords={"time": feb, "sites": ["a", "b"], "freq": freq, "dir": dirs},
        )
        store.write(ds, ["2020-02"])
        out = store.read("2020-02-02", ["b"])
        self.assertEqual(out.efth.dims, ("time", "sites", "freq", "dir"))
        self.assertTrue((out.efth == 1).all())


if __name__ == "__main__":
    unittest.main()
