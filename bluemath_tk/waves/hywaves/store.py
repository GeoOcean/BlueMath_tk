"""
Project: BlueMath_tk
Sub-Module: waves.hywaves.store
Author: GeoOcean Research Group, Universidad de Cantabria
Repository: https://github.com/GeoOcean/BlueMath_tk.git
Status: Under development (Working)

Cache of reconstructed waves: one zarr store per reconstruction setup,
filled one site x calendar month at a time.

Every (site, hour) of a HyWaves reconstruction is independent of the others,
so results can be computed in any pieces and kept together. The store holds
the whole time axis of the offshore hindcast and every site of the setup:

- arrays on ``(month, step, sites)`` (plus e.g. ``freq``, ``dir`` for
  spectra), ``step`` = position inside the calendar month (padded with NaT to
  the longest month), so **one chunk is one site and one calendar month**;
  bulk arrays put all the sites of a month in one file (zarr sharding);
- ``done`` on ``(month, sites)``: which site-months are stored;
- the setup (``key``) and a ``fingerprint`` of its input files in the
  attributes: a store whose setup, inputs, sites or time axis changed is
  rebuilt, never reused.

A request reads the site-months it needs; :meth:`ReconstructionStore.missing`
lists the ones to compute (always whole months), and
:meth:`ReconstructionStore.write` stores them.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from filelock import FileLock

logger = logging.getLogger(__name__)

_NAT = np.iinfo(np.int64).min


class ReconstructionStore:
    """
    Site x month cache of a reconstruction on the time axis of its hindcast.

    Parameters
    ----------
    path : str or Path
        The zarr store (created if missing, rebuilt if it does not match).
    times : array-like of datetime64
        The full time axis of the offshore hindcast.
    sites : sequence of str
        Every site the store can hold.
    variables : mapping
        Variable name -> names of its extra dimensions (``()`` for bulk
        parameters, ``("freq", "dir")`` for spectra).
    coords : mapping, optional
        Values of the extra dimensions, e.g. ``{"freq": ..., "dir": ...}``.
    key : mapping, optional
        The setup (JSON-serialisable); part of the validity check.
    fingerprint : str, optional
        Fingerprint of the input files; part of the validity check.
    attrs : mapping or callable, optional
        More attributes to record when the store is created (not checked),
        e.g. the PCs per goal; a callable is only called then.
    """

    def __init__(
        self,
        path: str | Path,
        times: Any,
        sites: Iterable[str],
        variables: Mapping[str, tuple[str, ...]],
        *,
        coords: Mapping[str, Any] | None = None,
        key: Mapping[str, Any] | None = None,
        fingerprint: str = "",
        attrs: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.path = Path(path)
        self.sites = [str(s) for s in sites]
        self.variables = {name: tuple(dims) for name, dims in variables.items()}
        self.coords = {name: np.asarray(v) for name, v in (coords or {}).items()}
        index = pd.DatetimeIndex(times)
        self.month_labels = index.to_period("M")
        self.months = [str(m) for m in self.month_labels.unique()]
        steps = pd.Series(np.arange(len(index))).groupby(np.asarray(self.month_labels)).cumcount()
        self._month_of = pd.Index(self.months).get_indexer(self.month_labels.astype(str))
        self._step_of = steps.to_numpy()
        n_steps = int(self._step_of.max()) + 1
        grid = np.full((len(self.months), n_steps), _NAT, dtype=np.int64)
        grid[self._month_of, self._step_of] = index.asi8
        self._time_grid = grid
        self._site_index = pd.Index(self.sites)
        meta = {
            "key": json.dumps(key or {}, sort_keys=True, default=str),
            "fingerprint": fingerprint,
            "sites": self.sites,
            "variables": {k: list(v) for k, v in self.variables.items()},
        }
        self.group = self._open(meta, attrs)

    # ------------------------------------------------------------------
    # Creation and validity
    # ------------------------------------------------------------------

    def _matches(self, group: zarr.Group, meta: Mapping[str, Any]) -> bool:
        stored = dict(group.attrs)
        return all(stored.get(k) == v for k, v in meta.items()) and np.array_equal(
            group["time"][:], self._time_grid
        )

    def _open(self, meta: dict[str, Any], attrs: Any) -> zarr.Group:
        if self.path.exists():
            group = zarr.open_group(self.path, mode="r+")
            if self._matches(group, meta):
                return group
            logger.warning("%s does not match this setup or its inputs: rebuilt", self.path.name)
            shutil.rmtree(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        group = zarr.open_group(self.path, mode="w")
        n_months, n_steps = self._time_grid.shape
        n_sites = len(self.sites)
        group.create_array("time", data=self._time_grid, chunks=self._time_grid.shape)
        group.create_array(
            "done", shape=(n_months, n_sites), chunks=(n_months, n_sites), dtype="bool", fill_value=False
        )
        for name, values in self.coords.items():
            group.create_array(name, data=values)
        for name, extra in self.variables.items():
            extra_shape = tuple(len(self.coords[d]) for d in extra)
            shape = (n_months, n_steps, n_sites, *extra_shape)
            chunks = (1, n_steps, 1, *extra_shape)
            group.create_array(
                name,
                shape=shape,
                chunks=chunks,
                shards=None if extra else (1, n_steps, n_sites),
                dtype="float32",
                fill_value=np.nan,
            )
        extra = attrs() if callable(attrs) else (attrs or {})
        group.attrs.update({**meta, **extra})
        return group

    def _lock(self) -> FileLock:
        return FileLock(self.path.with_suffix(".lock"))

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------

    def months_of(self, time: Any) -> list[str]:
        """Calendar months (``"2013-12"``) touched by a time selection."""
        index = pd.DatetimeIndex(self._time_grid[self._time_grid != _NAT].view("datetime64[ns]"))
        selected = xr.DataArray(index, dims="time", coords={"time": index}).sel(time=time)
        return [str(m) for m in pd.DatetimeIndex(np.atleast_1d(selected.values)).to_period("M").unique()]

    def missing(self, time: Any, sites: Iterable[str]) -> dict[str, list[str]]:
        """Sites not stored yet for each month of the selection (months with none left omitted)."""
        site_idx = self._site_index.get_indexer(list(sites))
        if (site_idx < 0).any():
            raise KeyError("Sites not in this store")
        done = self.group["done"][:]
        out = {}
        for month in self.months_of(time):
            m = self.months.index(month)
            todo = [self.sites[i] for i in site_idx if not done[m, i]]
            if todo:
                out[month] = todo
        return out

    def write(self, ds: xr.Dataset, months: Iterable[str]) -> None:
        """
        Store whole months of *ds* (on ``time``, ``sites``) and mark them done.

        *ds* must cover every step of each month in *months* for its sites;
        steps it lacks (e.g. hours with no wave energy) are stored as NaN.
        """
        site_idx = self._site_index.get_indexer([str(s) for s in ds.sites.values])
        with self._lock():
            done = self.group["done"][:]
            for month in months:
                m = self.months.index(month)
                steps = self._time_grid[m][self._time_grid[m] != _NAT].view("datetime64[ns]")
                part = ds.reindex(time=steps)
                for name, extra in self.variables.items():
                    values = part[name].transpose("time", "sites", *extra).to_numpy().astype("float32", copy=False)
                    if extra:  # one chunk per site: write each site's month
                        for j, i in enumerate(site_idx):
                            self.group[name][m, : len(steps), i] = values[:, j]
                    else:  # one shard per month: read, update, write back
                        block = self.group[name][m]
                        block[: len(steps)][:, site_idx] = values
                        self.group[name][m] = block
                done[m, site_idx] = True
            self.group["done"][:] = done

    def invalidate(self, time: Any, sites: Iterable[str]) -> None:
        """Mark the months of *time* as not stored for *sites* (they are recomputed)."""
        site_idx = self._site_index.get_indexer(list(sites))
        with self._lock():
            done = self.group["done"][:]
            for month in self.months_of(time):
                done[self.months.index(month), site_idx] = False
            self.group["done"][:] = done

    def read(self, time: Any, sites: Iterable[str]) -> xr.Dataset:
        """
        Stored values of *sites* for a time selection, on ``(time, sites)``.

        Raises
        ------
        KeyError
            If a requested site-month is not stored.
        """
        sites = [str(s) for s in sites]
        if self.missing(time, sites):
            raise KeyError("Requested site-months are not stored; compute them first")
        site_idx = self._site_index.get_indexer(sites)
        month_idx = [self.months.index(m) for m in self.months_of(time)]
        times = self._time_grid[month_idx]
        valid = times != _NAT
        data = {}
        for name, extra in self.variables.items():
            values = np.stack([self.group[name].oindex[m, :, site_idx] for m in month_idx])
            data[name] = (("time", "sites", *extra), values[valid])
        coords = {"time": times[valid].view("datetime64[ns]"), "sites": sites}
        coords.update({d: self.coords[d] for extra in self.variables.values() for d in extra})
        return xr.Dataset(data, coords=coords).sel(time=time)

    def coverage(self) -> pd.DataFrame:
        """Stored sites per month (months with at least one)."""
        done = self.group["done"][:]
        counts = pd.Series(done.sum(axis=1), index=self.months, name="sites")
        return counts[counts > 0].to_frame()


__all__ = ["ReconstructionStore"]
