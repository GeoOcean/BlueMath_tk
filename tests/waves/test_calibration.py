import unittest

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from bluemath_tk.waves.calibration import CalVal, QuantileMapping, sector_index


def _pairs(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Hindcast underestimating the westerly wind sea by 10 %, swell right."""

    rng = np.random.default_rng(seed)
    sea = rng.uniform(0.5, 4.0, n)
    swell = rng.uniform(0.2, 2.0, n)
    dir_sea = rng.choice([270.0, 0.0], n)
    dir_swell = np.full(n, 315.0)
    factor = np.where(dir_sea == 270.0, 1.1, 1.0)
    ref = np.sqrt((factor * sea) ** 2 + swell**2) * rng.normal(1.0, 0.02, n)

    return pd.DataFrame(
        {
            "hs": np.sqrt(sea**2 + swell**2),
            "phs0": sea,
            "pdir0": dir_sea,
            "phs1": swell,
            "pdir1": dir_swell,
            "sat_hs": ref,
        },
        index=pd.date_range("2000", periods=n, freq="6h", name="time"),
    )


class TestCalVal(unittest.TestCase):
    """Directional calibration against a reference."""

    def test_sectors(self):
        """Sectors are centred on North; NaN gives -1."""

        np.testing.assert_array_equal(
            sector_index(np.array([0.0, 11.0, 11.3, 349.0, 270.0, np.nan]), 22.5),
            [0, 0, 1, 0, 12, -1],
        )

    def test_fit_recovers_factor(self):
        """The westerly sea sector gets ~1.1, the rest stays ~1 or unfitted."""

        cv = CalVal().fit(_pairs(), train_fraction=0.75)
        table = cv.coefficients.set_index(["family", "sector"])
        self.assertAlmostEqual(table.loc[("sea", 12), "factor"], 1.1, delta=0.02)
        self.assertAlmostEqual(table.loc[("sea", 0), "factor"], 1.0, delta=0.02)
        self.assertAlmostEqual(table.loc[("swell", 14), "factor"], 1.0, delta=0.03)
        self.assertFalse(table.loc[("sea", 4), "fitted"])  # no data from the east
        self.assertEqual(len(cv.train_index), 1500)
        self.assertEqual(len(cv.test_index), 500)

        metrics = cv.metrics().set_index(["set", "stage"])
        self.assertEqual(int(metrics.loc[("test", "raw"), "n"]), 500)
        self.assertLess(
            metrics.loc[("test", "calibrated"), "rmse"],
            metrics.loc[("test", "raw"), "rmse"],
        )
        self.assertLess(abs(metrics.loc[("test", "calibrated"), "bias"]), 0.02)

    def test_correct_dataset_and_time_split(self):
        """Correction applies to a goal dataset; the time split keeps order."""

        pairs = _pairs()
        cv = CalVal().fit(pairs, split="time", train_fraction=0.5)
        self.assertEqual(int(cv.train_index.max()), 999)
        ds = xr.Dataset.from_dataframe(pairs.drop(columns="sat_hs"))
        out = cv.correct(ds)
        west = pairs["pdir0"].to_numpy() == 270.0
        ratio = out["phs0"].to_numpy() / pairs["phs0"].to_numpy()
        self.assertAlmostEqual(float(np.mean(ratio[west])), 1.1, delta=0.02)
        self.assertTrue(np.allclose(out["pdir0"], ds["pdir0"]))
        energy = np.sqrt(out["phs0"] ** 2 + out["phs1"] ** 2)
        np.testing.assert_allclose(out["hs"], energy, rtol=1e-6)

    def test_bulk_only(self):
        """Without partitions, hs/dir form one family."""

        pairs = (
            _pairs()[["hs", "phs0", "pdir0", "sat_hs"]]
            .rename(columns={"pdir0": "dir"})
            .drop(columns="phs0")
        )
        cv = CalVal().fit(pairs)
        self.assertEqual(list(cv.families), ["total"])
        self.assertGreater(cv.coefficients.query("sector == 12")["factor"].item(), 1.0)



class TestQuantileMapping(unittest.TestCase):
    """Second calibration step: the Hs distribution, tail included."""

    def _fit(self, n: int = 20000) -> tuple[QuantileMapping, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        hs = rng.weibull(1.3, n) * 1.5
        reference = hs + np.maximum(hs - 3.0, 0.0) * 0.3
        return QuantileMapping().fit(hs, reference), hs, reference

    def test_recovers_distribution(self):
        """Mapped quantiles match the reference, up to the tail quantile."""

        mapping, hs, reference = self._fit()
        probabilities = [0.5, 0.9, 0.99, 0.999]
        np.testing.assert_allclose(
            np.quantile(mapping.map(hs), probabilities),
            np.quantile(reference, probabilities),
            atol=0.02,
        )

    def test_constant_beyond_tail_and_monotonic(self):
        """Beyond the last quantile the offset is constant; the map never decreases."""

        mapping, hs, _ = self._fit()
        last = mapping.table.iloc[-1]
        offset = last["reference"] - last["hindcast"]
        self.assertAlmostEqual(mapping.map(np.array([20.0]))[0], 20.0 + offset)
        self.assertTrue((np.diff(mapping.map(np.sort(hs))) >= -1e-9).all())

    def test_correct_scales_partitions(self):
        """Partition Hs follow the bulk ratio; other variables and NaN stay."""

        mapping, _, _ = self._fit()
        data = pd.DataFrame(
            {"hs": [0.0, 5.0, np.nan], "phs0": [0.0, 4.0, np.nan], "pdir0": [10.0, 20.0, 30.0]}
        )
        out = mapping.correct(data)
        ratio = out.loc[1, "hs"] / 5.0
        self.assertGreater(ratio, 1.0)
        self.assertAlmostEqual(out.loc[1, "phs0"], 4.0 * ratio)
        self.assertEqual(out.loc[0, "hs"], 0.0)
        self.assertTrue(np.isnan(out.loc[2, "hs"]))
        pd.testing.assert_series_equal(out["pdir0"], data["pdir0"])

    def test_dataset_and_too_few_pairs(self):
        """Datasets give the same result; a tail on too few pairs is refused."""

        mapping, _, _ = self._fit()
        table = pd.DataFrame({"hs": [1.0, 5.0], "phs0": [1.0, 4.0]})
        dataset = xr.Dataset({k: ("time", table[k].to_numpy()) for k in table})
        np.testing.assert_allclose(mapping.correct(dataset)["phs0"], mapping.correct(table)["phs0"])
        with self.assertRaises(ValueError):
            QuantileMapping().fit(np.ones(1000), np.ones(1000))

    def test_quantiles_and_plot(self):
        """Diagnostics on the fitted pairs; the samples are not saved with the model."""

        mapping, hs, _ = self._fit()
        table = mapping.quantiles(raw=hs * 0.9)
        self.assertEqual(list(table.columns), ["reference", "raw", "hindcast", "mapped", "n_above"])
        np.testing.assert_allclose(table["mapped"], table["reference"], atol=0.02)
        self.assertEqual(table.loc[0.999, "n_above"], 20)
        fig = mapping.plot(raw=hs * 0.9, labels={"reference": "satellite"})
        self.assertEqual(len(fig.axes), 5)
        plt.close(fig)
        with self.assertRaises(ValueError):
            mapping.evaluated_samples(raw=hs[:10])
        self.assertNotIn("_samples", mapping.__getstate__())


if __name__ == "__main__":
    unittest.main()
