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

A constant factor per sector corrects the level of a hindcast, not a storm tail
that is too short. :class:`QuantileMapping` is the second step: it maps the Hs
quantiles of the directionally calibrated hindcast onto those of the reference,
over the pairs of a whole area (a tail needs many storms: altimeter passes
around one point hold too few), and keeps the correction constant beyond
``tail_quantile``, where the quantiles rest on a handful of pairs.
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

#: Default labels and colours of the :class:`QuantileMapping` stages.
MAPPING_STAGES: dict[str, tuple[str, str]] = {
    "reference": ("reference", "black"),
    "raw": ("raw", "0.6"),
    "hindcast": ("hindcast", "tab:blue"),
    "mapped": ("mapped", "tab:orange"),
}


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


class QuantileMapping(BlueMathModel):
    """
    Monotonic map of hindcast Hs onto the Hs distribution of a reference.

    Each pair of samples gives the quantiles of both at the same probabilities:
    below the first one the correction is a factor, between them linear, and
    above ``tail_quantile`` a constant offset.

    Parameters
    ----------
    tail_quantile : float, optional
        Last quantile mapped. Default is 0.999.
    n_tail : int, optional
        Quantiles between 0.99 and ``tail_quantile``. Default is 12.
    min_tail_samples : int, optional
        Pairs needed above ``tail_quantile``. Default is 5.
    hs_prefix : str, optional
        Partition Hs prefix. Default "phs".

    Attributes
    ----------
    table : pd.DataFrame
        ``probability``, ``hindcast`` and ``reference`` quantiles.

    Notes
    -----
    The map aligns the two distributions, not each pair: the mapped Hs of a
    pair can move away from its reference even when the quantiles agree
    (:meth:`plot`, last panel). The samples given to :meth:`fit` are kept for
    :meth:`quantiles` and :meth:`plot` but not saved with the model.
    """

    def __init__(
        self,
        tail_quantile: float = 0.999,
        n_tail: int = 12,
        min_tail_samples: int = 5,
        hs_prefix: str = "phs",
    ) -> None:
        super().__init__()
        self.set_logger_name(name="QuantileMapping", level="INFO", console=True)
        if not 0.99 <= tail_quantile < 1:
            raise ValueError(f"tail_quantile must be in [0.99, 1), got {tail_quantile}")
        self.tail_quantile = float(tail_quantile)
        self.n_tail = int(n_tail)
        self.min_tail_samples = int(min_tail_samples)
        self.hs_prefix = hs_prefix
        self.table: pd.DataFrame | None = None
        self._samples: pd.DataFrame | None = None
        self._exclude_attributes += ["_samples"]

    def fit(self, hindcast: np.ndarray, reference: np.ndarray) -> QuantileMapping:
        """
        Fit the map on paired Hs samples.

        Parameters
        ----------
        hindcast, reference : np.ndarray
            Hs of the hindcast (already directionally calibrated) and of the
            reference, same shape; pairs with a NaN are dropped.

        Returns
        -------
        QuantileMapping
            The fitted instance.
        """

        hindcast, reference = np.asarray(hindcast, float), np.asarray(reference, float)
        self._samples = pd.DataFrame({"hindcast": hindcast, "reference": reference})
        finite = np.isfinite(hindcast) & np.isfinite(reference)
        hindcast, reference = hindcast[finite], reference[finite]
        tail_samples = hindcast.size * (1.0 - self.tail_quantile)
        if tail_samples < self.min_tail_samples:
            raise ValueError(
                f"{hindcast.size} pairs leave {tail_samples:.1f} above quantile "
                f"{self.tail_quantile}; {self.min_tail_samples} are needed"
            )
        probability = np.unique(
            np.r_[np.linspace(0.01, 0.99, 99), np.linspace(0.99, self.tail_quantile, self.n_tail)]
        )
        self.table = pd.DataFrame(
            {
                "probability": probability,
                "hindcast": np.maximum.accumulate(np.quantile(hindcast, probability)),
                "reference": np.maximum.accumulate(np.quantile(reference, probability)),
            }
        )
        last = self.table.iloc[-1]
        self.logger.info(
            "Fitted on %d pairs: %+.2f m above %.2f m",
            hindcast.size,
            last["reference"] - last["hindcast"],
            last["hindcast"],
        )

        return self

    def map(self, hs: np.ndarray) -> np.ndarray:
        """Mapped Hs (NaN stays NaN)."""

        if self.table is None:
            raise ValueError("Run fit() first")
        hs = np.asarray(hs, float)
        low, high = self.table["hindcast"].to_numpy(), self.table["reference"].to_numpy()
        mapped = np.interp(hs, low, high)
        mapped = np.where(hs < low[0], hs * high[0] / max(low[0], 1e-6), mapped)
        mapped = np.where(hs > low[-1], hs + high[-1] - low[-1], mapped)

        return np.where(np.isnan(hs), np.nan, mapped)

    def correct(self, data: pd.DataFrame | xr.Dataset) -> pd.DataFrame | xr.Dataset:
        """
        Map the bulk ``hs`` and scale the partition Hs by the same ratio.

        Parameters
        ----------
        data : pd.DataFrame or xr.Dataset
            Hindcast with ``hs`` and any partition Hs.

        Returns
        -------
        pd.DataFrame or xr.Dataset
            A copy with corrected Hs; other variables unchanged.
        """

        hs = np.asarray(data["hs"], float)
        ratio = np.divide(self.map(hs), hs, out=np.ones_like(hs), where=hs > 0)
        factor = ratio if isinstance(data, pd.DataFrame) else data["hs"].copy(data=ratio)
        out = data.copy()
        names = data.columns if isinstance(data, pd.DataFrame) else data.data_vars
        for name in names:
            if name == "hs" or str(name).startswith(self.hs_prefix):
                out[name] = data[name] * factor

        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def evaluated_samples(self, raw: np.ndarray | None = None) -> pd.DataFrame:
        """
        The fitted pairs with their mapped Hs.

        Parameters
        ----------
        raw : np.ndarray, optional
            Hs before any earlier correction, aligned with the samples given
            to :meth:`fit` (e.g. before the directional step).

        Returns
        -------
        pd.DataFrame
            ``reference``, ``raw`` (if given), ``hindcast`` and ``mapped``,
            pairs with a NaN dropped.
        """

        if self._samples is None:
            raise ValueError("Run fit() first (the samples are not stored after loading)")
        samples = self._samples.copy()
        if raw is not None:
            raw = np.asarray(raw, float)
            if raw.shape != (len(samples),):
                raise ValueError(f"raw has shape {raw.shape}; the fit had {len(samples)} pairs")
            samples["raw"] = raw
        samples["mapped"] = self.map(samples["hindcast"].to_numpy())
        columns = [c for c in ("reference", "raw", "hindcast", "mapped") if c in samples]

        return samples[columns].dropna().reset_index(drop=True)

    def quantiles(
        self,
        probabilities: tuple[float, ...] = (0.5, 0.9, 0.99, 0.999),
        raw: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Hs quantiles of the reference and of each stage, on the fitted pairs.

        Parameters
        ----------
        probabilities : tuple of float, optional
            Non-exceedance probabilities.
        raw : np.ndarray, optional
            See :meth:`evaluated_samples`.

        Returns
        -------
        pd.DataFrame
            Index ``probability``; one column per stage and ``n_above``, the
            pairs above the quantile (what the value rests on).
        """

        samples = self.evaluated_samples(raw)
        table = samples.quantile(list(probabilities))
        table.index.name = "probability"
        table["n_above"] = np.round(len(samples) * (1.0 - table.index.to_numpy())).astype(int)

        return table

    def plot(
        self,
        raw: np.ndarray | None = None,
        labels: dict[str, str] | None = None,
        title: str | None = None,
    ) -> Figure:
        """
        What the map does: the correction, the quantiles, the tail and the pairs.

        Panels: (a) correction (mapped - hindcast) against hindcast Hs, with the
        fitted quantiles, the three regions (factor below the first quantile,
        interpolated, constant offset beyond ``tail_quantile``), how many pairs
        each holds and their histogram; (b) Q-Q against the reference; (c)
        exceedance probability; (d) mapped Hs of each pair against its
        reference.

        Parameters
        ----------
        raw : np.ndarray, optional
            See :meth:`evaluated_samples`; adds the raw stage to (b) and (c).
        labels : dict, optional
            Stage labels (keys ``reference``, ``raw``, ``hindcast``,
            ``mapped``), e.g. ``{"reference": "satellite"}``.
        title : str, optional
            Figure title.

        Returns
        -------
        Figure
            The 2 x 2 panels.
        """

        samples = self.evaluated_samples(raw)
        names = {key: (labels or {}).get(key, label) for key, (label, _) in MAPPING_STAGES.items()}
        colors = {key: color for key, (_, color) in MAPPING_STAGES.items()}
        stages = [c for c in ("raw", "hindcast", "mapped") if c in samples]
        low, high = self.table["hindcast"].to_numpy(), self.table["reference"].to_numpy()
        hindcast = samples["hindcast"].to_numpy()
        n = len(samples)
        fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))

        ax = axes[0, 0]
        x_max = max(hindcast.max(), low[-1]) * 1.1
        x = np.linspace(0.0, x_max, 500)
        ax_hist = ax.twinx()
        ax_hist.hist(hindcast, bins=60, range=(0, x_max), color="0.85", log=True)
        ax_hist.set_ylabel("pairs per bin (log)", color="0.5")
        ax.set_zorder(ax_hist.get_zorder() + 1)
        ax.patch.set_visible(False)
        regions = (
            (0.0, low[0], "factor", "0.6"),
            (low[0], low[-1], "interpolated", "tab:blue"),
            (low[-1], x_max, f"offset {high[-1] - low[-1]:+.2f} m", "tab:red"),
        )
        for k, (start, end, label, color) in enumerate(regions):
            ax.axvspan(start, end, color=color, alpha=0.07)
            inside = int(((hindcast >= start) & (hindcast < end)).sum())
            ax.text(0.01 * x_max if k == 0 else (start + end) / 2, 0.97, f"{label}\n{inside} pairs",
                    transform=ax.get_xaxis_transform(), ha="left" if k == 0 else "center", va="top",
                    fontsize=8, color=color)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.plot(x, self.map(x) - x, color=colors["mapped"], lw=2, label="mapped - hindcast")
        ax.plot(low, high - low, "o", ms=3, color="k", label="fitted quantiles")
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top + 0.3 * (top - bottom))
        ax.set(xlim=(0, x_max), xlabel=f"{names['hindcast']} Hs (m)", ylabel="correction (m)",
               title=f"(a) the map · tail at p{100 * self.tail_quantile:g} = {low[-1]:.2f} m")
        ax.legend(loc="lower right", fontsize=8)

        ax = axes[0, 1]
        probability = np.unique(np.r_[np.linspace(0.01, 0.99, 99), 1.0 - np.logspace(-2, np.log10(1.0 / n), 40)])
        reference_q = np.quantile(samples["reference"], probability)
        for stage in stages:
            ax.plot(reference_q, np.quantile(samples[stage], probability), ".-", ms=3, lw=1,
                    color=colors[stage], label=names[stage])
        limit = max(reference_q.max(), samples[stages].quantile(probability[-1]).max()) * 1.05
        ax.plot([0, limit], [0, limit], "k--", lw=0.8)
        tail = np.quantile(samples["reference"], self.tail_quantile)
        ax.axvline(tail, color="tab:red", ls=":", lw=1)
        ax.text(tail, 0.02, f" p{100 * self.tail_quantile:g}", transform=ax.get_xaxis_transform(),
                color="tab:red", fontsize=8)
        ax.set(xlim=(0, limit), ylim=(0, limit), aspect="equal",
               xlabel=f"{names['reference']} Hs quantile (m)", ylabel="Hs quantile (m)",
               title="(b) Q-Q: on the 1:1 line the distributions agree")
        ax.legend(fontsize=8)

        ax = axes[1, 0]
        exceedance = 1.0 - np.arange(1, n + 1) / (n + 1)
        for stage in ("reference", *stages):
            ax.semilogy(np.sort(samples[stage].to_numpy()), exceedance, lw=1.2 + (stage == "reference"),
                        color=colors[stage], label=names[stage])
        ax.axhline(1.0 - self.tail_quantile, color="tab:red", ls=":", lw=1)
        ax.text(0.01, 1.0 - self.tail_quantile, f"p{100 * self.tail_quantile:g}", color="tab:red",
                fontsize=8, va="bottom", transform=ax.get_yaxis_transform())
        ax.set(xlabel="Hs (m)", ylabel="exceedance probability",
               title=f"(c) tail · {n} pairs, {int((hindcast > low[-1]).sum())} beyond the last quantile")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)

        validation_scatter(
            axes[1, 1],
            reference=samples["reference"].to_numpy(),
            estimate=samples["mapped"].to_numpy(),
            reference_label=f"{names['reference']} Hs",
            estimate_label=f"{names['mapped']} Hs",
            title="(d) pairs: the map aligns distributions, not pairs",
            units="m",
        )
        if title:
            fig.suptitle(title)
        fig.tight_layout()

        return fig
