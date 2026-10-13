"""
Directional calibration of hindcast significant wave height against a
reference (satellite altimetry, buoys).

Method (Mínguez et al., 2011, *Directional calibration of wave reanalysis
databases using instrumental data*): the reference energy is fitted as a sum
of the hindcast energy of each family (wind sea, swell) in each direction
sector,

.. math::

    H_{s,ref}^2 = \\sum_k a^{sea}_k E^{sea}_k + \\sum_k a^{swell}_k E^{swell}_k,
    \\qquad E_k = \\sum_{i \\in k} H_{s,i}^2,

without intercept, so calm seas stay calm. A partition contributes to the
sector of its mean direction; partition 0 is the wind sea and the others are
swells (the HyWaves ``phs{i}`` / ``pdir{i}`` naming). Without partitions the
bulk ``hs`` / ``dir`` is calibrated with one family. Each partition Hs is then
multiplied by :math:`\\sqrt{a}` of its family and sector; the bulk Hs is
scaled by the change in total partition energy.

Coefficients that are not supported by the data (fewer than ``min_samples``
pairs in the sector, p-value above ``p_value`` or not positive) are fixed at
1 (no change) and the others refitted, until all are accepted.

The pairs (e.g. from :func:`bluemath_tk.waves.altimetry.collocate`) are split
into a training part (``train_fraction``, 75 % by default) used for the fit
and a test part used only to validate it (:meth:`CalVal.metrics`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from ..core.metrics import compute_metrics
from ..core.models import BlueMathModel
from ..core.plotting.scatter import validation_scatter

#: Metrics reported by :meth:`CalVal.metrics` (reference first: bias = hindcast - reference).
CALIBRATION_METRICS: tuple[str, ...] = ("bias", "rmse", "si", "r2", "slope")


def sector_index(direction: np.ndarray, sector_size: float) -> np.ndarray:
    """
    Direction sector of nautical directions, sectors centred on North.

    Sector 0 is ``[360 - size/2, size/2)``, sector 1 the next one clockwise, ...

    Parameters
    ----------
    direction : np.ndarray
        Directions (degrees); NaN gives -1.
    sector_size : float
        Sector width (degrees), dividing 360.

    Returns
    -------
    np.ndarray
        Integer sector per value (-1 where the direction is NaN).
    """

    direction = np.asarray(direction, dtype=float)
    n_sectors = int(round(360.0 / sector_size))
    idx = np.floor(
        ((np.nan_to_num(direction) + sector_size / 2.0) % 360.0) / sector_size
    )
    idx = np.where(np.isfinite(direction), idx % n_sectors, -1)

    return idx.astype(int)


class CalVal(BlueMathModel):
    """
    Directional calibration of hindcast Hs against reference Hs.

    Parameters
    ----------
    sector_size : float, optional
        Direction sector width (degrees). Default is 22.5 (16 sectors).
    hs_prefix, dir_prefix : str, optional
        Partition variable prefixes. Default "phs" / "pdir" (``phs0`` = wind
        sea, ``phs1``, ... = swells).
    min_samples : int, optional
        Pairs with energy in a sector needed to fit its coefficient. Default 20.
    p_value : float, optional
        Largest accepted p-value of a coefficient. Default 0.05.

    Attributes
    ----------
    coefficients : pd.DataFrame
        One row per family and sector: ``family``, ``sector``, ``center``,
        ``n`` (training pairs with energy there), ``a`` (energy factor),
        ``factor`` (Hs factor, ``sqrt(a)``), ``fitted`` (False when fixed at
        1) and ``p_value``.
    """

    def __init__(
        self,
        sector_size: float = 22.5,
        hs_prefix: str = "phs",
        dir_prefix: str = "pdir",
        min_samples: int = 20,
        p_value: float = 0.05,
    ) -> None:
        super().__init__()
        self.set_logger_name(name="CalVal", level="INFO", console=True)
        if (360.0 / sector_size) % 1:
            raise ValueError(f"sector_size must divide 360, got {sector_size}")
        self.sector_size = float(sector_size)
        self.n_sectors = int(round(360.0 / sector_size))
        self.hs_prefix = hs_prefix
        self.dir_prefix = dir_prefix
        self.min_samples = int(min_samples)
        self.p_value = float(p_value)

        self.families: dict[str, list[tuple[str, str]]] = {}
        self.reference: str = ""
        self.coefficients: pd.DataFrame | None = None
        self.train_index: pd.Index | None = None
        self.test_index: pd.Index | None = None
        self._pairs: pd.DataFrame | None = None
        self._exclude_attributes += ["_pairs"]

    # ------------------------------------------------------------------
    # Families and design matrix
    # ------------------------------------------------------------------

    def _families(self, columns) -> dict[str, list[tuple[str, str]]]:
        """``{"sea": [(phs0, pdir0)], "swell": [(phs1, pdir1), ...]}`` or bulk."""

        columns = list(columns)
        n = len(self.hs_prefix)
        parts = sorted(
            int(c[n:])
            for c in columns
            if c.startswith(self.hs_prefix)
            and c[n:].isdigit()
            and f"{self.dir_prefix}{c[n:]}" in columns
        )
        if parts:
            pair = lambda i: (f"{self.hs_prefix}{i}", f"{self.dir_prefix}{i}")  # noqa: E731
            families = {"sea": [pair(parts[0])]}
            if len(parts) > 1:
                families["swell"] = [pair(i) for i in parts[1:]]
            return families
        if "hs" in columns and "dir" in columns:
            return {"total": [("hs", "dir")]}
        raise ValueError(
            f"No partitions ({self.hs_prefix}N/{self.dir_prefix}N) nor hs/dir in {columns}"
        )

    def _energy(self, data: pd.DataFrame) -> np.ndarray:
        """Energy per family and sector, ``(n, n_families * n_sectors)``."""

        blocks = []
        for members in self.families.values():
            block = np.zeros((len(data), self.n_sectors))
            for hs_name, dir_name in members:
                hs = np.nan_to_num(data[hs_name].to_numpy(float))
                idx = sector_index(data[dir_name].to_numpy(float), self.sector_size)
                ok = (idx >= 0) & (hs > 0)
                np.add.at(block, (np.flatnonzero(ok), idx[ok]), hs[ok] ** 2)
            blocks.append(block)

        return np.hstack(blocks)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        pairs: pd.DataFrame,
        reference: str = "sat_hs",
        train_fraction: float = 0.75,
        split: str = "random",
        seed: int = 0,
    ) -> CalVal:
        """
        Fit the directional coefficients on part of the pairs.

        Parameters
        ----------
        pairs : pd.DataFrame
            Hindcast variables (``hs`` and partitions ``phs{i}``, ``pdir{i}``,
            or ``hs``, ``dir``) and the reference column, one row per
            collocation (e.g. :func:`bluemath_tk.waves.altimetry.collocate`).
        reference : str, optional
            Reference Hs column. Default is "sat_hs".
        train_fraction : float, optional
            Share of the pairs used for the fit; the rest validates it.
            Default is 0.75. Use 1.0 to fit on everything.
        split : {"random", "time"}, optional
            "random" draws the training pairs at random (seeded); "time" takes
            the first *train_fraction* of the record. Default "random".
        seed : int, optional
            Random seed for ``split="random"``. Default is 0.

        Returns
        -------
        CalVal
            The fitted instance.
        """

        if not 0 < train_fraction <= 1:
            raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")
        pairs = pairs[np.isfinite(pairs[reference]) & np.isfinite(pairs["hs"])]
        if pairs.empty:
            raise ValueError("No finite pairs to fit")
        pairs = pairs.reset_index()
        self.reference = reference
        self.families = self._families(pairs.columns)
        self._pairs = pairs

        n_train = int(round(train_fraction * len(pairs)))
        if split == "random":
            order = np.random.default_rng(seed).permutation(len(pairs))
        elif split == "time":
            order = np.argsort(pairs.iloc[:, 0].to_numpy(), kind="stable")
        else:
            raise ValueError(f"split must be 'random' or 'time', got {split!r}")
        self.train_index = pd.Index(np.sort(order[:n_train]))
        self.test_index = pd.Index(np.sort(order[n_train:]))
        train = pairs.loc[self.train_index]

        X = self._energy(train)
        y = train[reference].to_numpy(float) ** 2
        n_obs = (X > 0).sum(axis=0)
        a = np.ones(X.shape[1])
        fitted = n_obs >= self.min_samples
        p_values = np.full(X.shape[1], np.nan)
        for _ in range(X.shape[1]):
            if not fitted.any():
                break
            offset = X[:, ~fitted] @ a[~fitted]
            result = sm.OLS(y - offset, X[:, fitted]).fit()
            params, pvals = np.asarray(result.params), np.asarray(result.pvalues)
            bad = (params <= 0) | ~(pvals < self.p_value)
            a[fitted] = np.where(bad, 1.0, params)
            p_values[fitted] = pvals
            if not bad.any():
                break
            fitted[np.flatnonzero(fitted)[bad]] = False
            a[~fitted] = 1.0

        rows = []
        for f, family in enumerate(self.families):
            for k in range(self.n_sectors):
                j = f * self.n_sectors + k
                rows.append(
                    {
                        "family": family,
                        "sector": k,
                        "center": k * self.sector_size,
                        "n": int(n_obs[j]),
                        "a": float(a[j]),
                        "factor": float(np.sqrt(a[j])),
                        "fitted": bool(fitted[j]),
                        "p_value": float(p_values[j]) if fitted[j] else np.nan,
                    }
                )
        self.coefficients = pd.DataFrame(rows)
        self.logger.info(
            "Fitted %d of %d coefficients on %d pairs (%d kept for validation)",
            int(fitted.sum()),
            fitted.size,
            len(train),
            len(self.test_index),
        )

        return self

    # ------------------------------------------------------------------
    # Correction
    # ------------------------------------------------------------------

    def _factor(self, family: str, direction: np.ndarray) -> np.ndarray:
        """Hs factor of *family* at each direction (1 where unknown)."""

        table = self.coefficients
        factors = table.loc[table["family"] == family, "factor"].to_numpy()
        idx = sector_index(direction, self.sector_size)

        return np.where(idx >= 0, factors[np.clip(idx, 0, None)], 1.0)

    def correct(self, data: pd.DataFrame | xr.Dataset) -> pd.DataFrame | xr.Dataset:
        """
        Apply the calibration: partition Hs times its factor, bulk Hs rescaled.

        Parameters
        ----------
        data : pd.DataFrame or xr.Dataset
            Hindcast with the variables used in :meth:`fit` (any dimensions
            for a Dataset, e.g. ``time``).

        Returns
        -------
        pd.DataFrame or xr.Dataset
            A copy with corrected ``hs`` and partition Hs; other variables
            unchanged.
        """

        if self.coefficients is None:
            raise ValueError("Run fit() first")
        out = data.copy()
        if "total" in self.families:
            out["hs"] = data["hs"] * self._factor("total", np.asarray(data["dir"]))
            return out

        energy_before = 0.0
        energy_after = 0.0
        for family, members in self.families.items():
            for hs_name, dir_name in members:
                hs = data[hs_name]
                corrected = hs * self._factor(family, np.asarray(data[dir_name]))
                out[hs_name] = corrected
                energy_before = energy_before + np.nan_to_num(np.asarray(hs)) ** 2
                energy_after = energy_after + np.nan_to_num(np.asarray(corrected)) ** 2
        energy_before = np.asarray(energy_before, dtype=float)
        ratio = np.sqrt(
            np.divide(
                np.asarray(energy_after, dtype=float),
                energy_before,
                out=np.ones_like(energy_before),
                where=energy_before > 0,
            )
        )
        out["hs"] = data["hs"] * ratio

        return out

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def evaluated_pairs(self) -> pd.DataFrame:
        """
        The fitted pairs with the calibrated Hs (``hs_cal``) and their ``set``.

        Returns
        -------
        pd.DataFrame
            The pairs given to :meth:`fit`, plus ``hs_cal`` and ``set``
            ("train" or "test").
        """

        if self._pairs is None:
            raise ValueError("Run fit() first (the pairs are not stored after loading)")
        pairs = self._pairs.copy()
        pairs["hs_cal"] = np.asarray(self.correct(pairs)["hs"])
        pairs["set"] = np.where(pairs.index.isin(self.test_index), "test", "train")

        return pairs

    def metrics(self, metrics: tuple[str, ...] = CALIBRATION_METRICS) -> pd.DataFrame:
        """
        Hindcast against the reference, before and after the calibration.

        Parameters
        ----------
        metrics : tuple of str, optional
            Metric names (see :func:`bluemath_tk.core.metrics.compute_metrics`).

        Returns
        -------
        pd.DataFrame
            Rows ``set`` (train, test, all) x ``stage`` (raw, calibrated):
            ``n`` and the metrics (reference first: bias = hindcast - reference).
        """

        pairs = self.evaluated_pairs()
        rows = []
        for name, subset in (
            ("train", pairs[pairs["set"] == "train"]),
            ("test", pairs[pairs["set"] == "test"]),
            ("all", pairs),
        ):
            for stage, column in (("raw", "hs"), ("calibrated", "hs_cal")):
                values = compute_metrics(
                    reference=subset[self.reference],
                    estimate=subset[column],
                    metrics=metrics,
                )
                rows.append({"set": name, "stage": stage, **values})

        return pd.DataFrame(rows)

    def plot(self, title: str | None = None) -> Figure:
        """
        Test pairs before and after the calibration, and the factors per sector.

        Parameters
        ----------
        title : str, optional
            Figure title.

        Returns
        -------
        Figure
            Two scatters (reference on x) and one polar panel per family.
        """

        pairs = self.evaluated_pairs()
        test = pairs[pairs["set"] == "test"]
        label_set = "test"
        if test.empty:
            test, label_set = pairs, "all"
        n_fam = len(self.families)
        fig = plt.figure(figsize=(4.5 * (2 + n_fam), 4.8))
        for i, (column, label) in enumerate((("hs", "raw"), ("hs_cal", "calibrated"))):
            ax = fig.add_subplot(1, 2 + n_fam, i + 1)
            validation_scatter(
                ax,
                reference=test[self.reference].to_numpy(),
                estimate=test[column].to_numpy(),
                reference_label="Reference Hs",
                estimate_label=f"Hindcast Hs ({label})",
                title=f"{label_set} pairs · {label}",
                units="m",
            )
        theta = np.deg2rad(np.arange(self.n_sectors) * self.sector_size)
        width = np.deg2rad(self.sector_size) * 0.9
        for j, family in enumerate(self.families):
            ax = fig.add_subplot(1, 2 + n_fam, 3 + j, projection="polar")
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            table = self.coefficients[self.coefficients["family"] == family]
            colors = np.where(
                ~table["fitted"].to_numpy(),
                "0.75",
                np.where(table["factor"].to_numpy() >= 1, "tab:red", "tab:blue"),
            )
            ax.bar(
                theta,
                table["factor"].to_numpy() - 1.0,
                width=width,
                bottom=1.0,
                color=colors,
            )
            ax.plot(np.linspace(0, 2 * np.pi, 100), np.ones(100), "k-", lw=0.8)
            ax.set_title(f"{family}: Hs factor (grey = not fitted)", fontsize=10)
        if title:
            fig.suptitle(title)
        fig.tight_layout()

        return fig
