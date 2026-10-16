"""
HyWaves reconstruction: linear summation of goal contributions.

Each goal (and spectral partition) contributes nearshore ``hs``, ``tp``,
``dir``, ``spr`` per site and time step (see
:func:`~bluemath_tk.waves.hywaves.metamodel.predict_fields`). Every
contribution becomes a JONSWAP x Cartwright spectrum, the spectra are summed
over goals and partitions, and bulk parameters are computed from the sum
(:func:`sum_partition_spectra`). With a wind correction, one more
contribution, the local wind sea, is added per block before the sum
(see :mod:`.wind_correction`).

The sum reproduces a model run forced by every goal at once only if each
metamodel case imposes a fraction ``w(goal, point)`` of its goal's *energy*
along the open boundary and the fractions of all goals add up to 1. Both
models do: SnapWave mixes the spectra of the two nearest goals at every
boundary node with inverse-distance weights; SWAN (``BOUNDSPEC ... VARIABLE
PAR``) interpolates the spectra linearly along the boundary, so the energy is
linear and Hs falls as ``sqrt(w)`` between goals, and the boundary ends take
their nearest goal in full (see
:func:`bluemath_tk.wrappers.swan.swan_utils.boundary_par_table`).

The spectral construction allocates the full ``(time, sites, goal x partition,
freq, dir)`` array (its peak was measured at 6.55x the float32 array, see
:func:`site_sum_bytes`). Every site and hour is independent, so the sum runs
in site x time blocks sized to a memory budget (:func:`batch_shape`),
optionally across worker processes (:func:`sum_partition_spectra_batched`);
the budget and the workers come from :mod:`.resources`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

from ...core.resources import child_threads

logger = logging.getLogger(__name__)

#: Frequency (Hz) and direction (deg) grids for the summed spectra.
SPECTRAL_FREQ = np.logspace(np.log10(0.035), np.log10(0.5), 29)
SPECTRAL_DIR = np.arange(0.0, 360.0, 5)

#: Spectral partitions expected in an offshore dataset (``phs0``...).
SPECTRAL_PARTITIONS = (0, 1, 2, 3)

# wavespectra's construct_partition (jonswap x cartwright, then the sum) keeps
# several temporaries of the size of the float32 (time, sites, parts, freq,
# dir) array alive at once, and the stats add a few of the summed spectrum:
# measured peaks (2026-10-15) were 6.35x per part + 2.3x per summed spectrum,
# sized here with a margin.
PEAK_MEMORY_PER_PART = 8
PEAK_MEMORY_PER_SUM = 4


def partition_ids(offshore: xr.Dataset, spectral: bool) -> list[int]:
    """
    Partitions to reconstruct: bulk only, or every spectral partition.

    Parameters
    ----------
    offshore : xr.Dataset
        Offshore dataset (checked for ``phs{i}`` in spectral mode).
    spectral : bool
        Spectral (partitions :data:`SPECTRAL_PARTITIONS`) or bulk (``-1``).

    Returns
    -------
    list[int]
        Partition ids.

    Raises
    ------
    ValueError
        If spectral mode is requested and partition variables are missing.
    """

    from .metamodel import BULK_PARTITION

    if not spectral:
        return [BULK_PARTITION]
    missing = [i for i in SPECTRAL_PARTITIONS if f"phs{i}" not in offshore.data_vars]
    if missing:
        raise ValueError(
            "Offshore dataset missing partition variables: "
            + ", ".join(f"phs{i}" for i in missing)
        )

    return list(SPECTRAL_PARTITIONS)


def site_sum_bytes(n_times: int, n_parts: int) -> int:
    """
    Peak memory of :func:`sum_partition_spectra` per site (bytes).

    :data:`PEAK_MEMORY_PER_PART` float32 spectra per contribution plus
    :data:`PEAK_MEMORY_PER_SUM` for the summed spectrum and its stats.

    Parameters
    ----------
    n_times : int
        Time steps summed.
    n_parts : int
        Contributions summed per site.

    Returns
    -------
    int
        Bytes.
    """

    spectrum = max(1, n_times) * len(SPECTRAL_FREQ) * len(SPECTRAL_DIR) * 4
    return spectrum * (PEAK_MEMORY_PER_PART * max(1, n_parts) + PEAK_MEMORY_PER_SUM)


def spectra_bytes(n_times: int, n_sites: int) -> int:
    """Size of the summed float32 spectra ``efth`` of *n_sites* over *n_times* steps."""

    return n_times * n_sites * len(SPECTRAL_FREQ) * len(SPECTRAL_DIR) * 4


def batch_shape(n_times: int, n_parts: int, budget_bytes: int) -> tuple[int, int]:
    """
    Sites and time steps per :func:`sum_partition_spectra` call within a memory budget.

    Whole time series of as many sites as fit; when one site does not fit,
    one site over as many steps as fit (every step is independent).

    Parameters
    ----------
    n_times : int
        Time steps to sum.
    n_parts : int
        Contributions summed per site.
    budget_bytes : int
        Memory budget.

    Returns
    -------
    tuple[int, int]
        Sites and steps per call, each at least 1.
    """

    per_site = site_sum_bytes(n_times, n_parts)
    if per_site <= budget_bytes:
        return budget_bytes // per_site, max(1, n_times)

    return 1, max(1, budget_bytes // site_sum_bytes(1, n_parts))


def sum_partition_spectra(
    cube: xr.Dataset,
    site_ids: list[str],
    variables: Mapping[str, str],
    save_spectra: bool = False,
    load: bool = True,
    freq: np.ndarray = SPECTRAL_FREQ,
    dirs: np.ndarray = SPECTRAL_DIR,
) -> tuple[xr.Dataset, xr.Dataset | None]:
    """
    Sum goal x partition contributions as spectra and compute bulk parameters.

    Parameters
    ----------
    cube : xr.Dataset
        ``hs``, ``tp``, ``dir``, ``spr`` on ``(goal, partition, time, sites)``
        (missing contributions as NaN).
    site_ids : list of str
        Sites to process.
    variables : mapping
        Output name -> wavespectra stat, e.g. ``{"hs": "hs", "tp": "tp"}``.
    save_spectra : bool, optional
        Also return the summed spectrum ``efth`` on ``(time, sites, freq, dir)``.
    load : bool, optional
        Load the results into memory. Default is True.
    freq, dirs : np.ndarray, optional
        Spectral grid. Default :data:`SPECTRAL_FREQ`, :data:`SPECTRAL_DIR`.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset | None]
        Bulk parameters on ``(time, sites)`` and the optional spectra.
    """

    from wavespectra.construct import construct_partition

    parts = cube.sel(sites=site_ids).stack(part=("goal", "partition"))
    # contributions never predicted (e.g. a goal x partition combination
    # created by merging, such as the wind-sea contribution) carry no energy
    present = parts.hs.notnull().any([d for d in parts.hs.dims if d != "part"])
    parts = parts.isel(part=np.flatnonzero(present.values))
    # float32 halves the peak array (twice the sites per batch); the relative
    # difference from float64 was ~1e-6 on a real reconstruction.
    spec = construct_partition(
        freq_name="jonswap",
        freq_kwargs={
            "freq": np.asarray(freq, dtype="float32"),
            "fp": (1.0 / parts.tp).astype("float32"),
            "hs": parts.hs.astype("float32"),
        },
        dir_name="cartwright",
        dir_kwargs={
            "dir": np.asarray(dirs, dtype="float32"),
            "dm": parts.dir.astype("float32"),
            "dspr": parts.spr.astype("float32"),
        },
    )
    combined = spec.sum(dim="part")
    if "sites" not in combined.dims:
        combined = combined.expand_dims(sites=list(site_ids))

    spectra = None
    if save_spectra:
        spectra = (
            combined
            if isinstance(combined, xr.Dataset)
            else combined.to_dataset(name="efth")
        )
        if load:
            spectra = spectra.load()

    stats = combined.spec.stats(sorted(set(variables.values())))
    result = xr.Dataset({name: stats[stat] for name, stat in variables.items()})
    if load:
        result = result.load()

    return result, spectra


def _add_wind_sea(
    merged: xr.Dataset, wind_correction: Mapping[str, Any]
) -> tuple[xr.Dataset, xr.DataArray | None]:
    """Merge the wind-sea contribution of one site batch; also return its ``K``."""

    from .wind_correction import wind_sea_contribution

    contrib = wind_sea_contribution(merged, **wind_correction)
    if contrib is None:
        return merged, None
    k = contrib["k"].squeeze(["goal", "partition"], drop=True)

    return xr.merge([merged, contrib.drop_vars("k")]), k


def _time_axis(chunks: list[xr.Dataset]) -> np.ndarray:
    return np.unique(np.concatenate([c["time"].values for c in chunks]))


def _n_parts(chunks: list[xr.Dataset], wind_correction: Any) -> int:
    return sum(c.sizes["goal"] * c.sizes["partition"] for c in chunks) + int(
        wind_correction is not None
    )


def _sum_blocks(
    chunks: list[xr.Dataset],
    site_ids: list[str],
    variables: Mapping[str, str],
    save_spectra: bool,
    budget_bytes: int,
    load: bool = True,
    label: str = "",
    wind_correction: Mapping[str, Any] | None = None,
) -> tuple[xr.Dataset, xr.Dataset | None]:
    """Merge contribution chunks and sum them, in site x time blocks within the budget."""

    times = _time_axis(chunks)
    n_sites, n_steps = batch_shape(len(times), _n_parts(chunks, wind_correction), budget_bytes)
    n_blocks = -(-len(site_ids) // n_sites) * -(-len(times) // n_steps)
    results, spectras = [], []
    t0 = time.monotonic()
    k = 0
    for i in range(0, len(site_ids), n_sites):
        sites = site_ids[i : i + n_sites]
        merged = xr.merge(
            [c.sel(sites=sites) for c in chunks], join="outer", compat="no_conflicts"
        )
        site_results, site_spectra = [], []
        for j in range(0, merged.sizes["time"], n_steps):
            t_block = time.monotonic()
            block = merged.isel(time=slice(j, j + n_steps))
            wind_k = None
            if wind_correction is not None:
                block, wind_k = _add_wind_sea(block, wind_correction)
            result, spectra = sum_partition_spectra(
                block, sites, variables, save_spectra=save_spectra, load=load
            )
            if wind_k is not None:
                result["wind_k"] = wind_k
            site_results.append(result)
            if save_spectra and spectra is not None:
                site_spectra.append(spectra)
            k += 1
            logger.info(
                "%ssum block %d/%d (%d sites x %d steps) in %.1fs (%.1fs elapsed)",
                f"[{label}] " if label else "",
                k,
                n_blocks,
                len(sites),
                block.sizes["time"],
                time.monotonic() - t_block,
                time.monotonic() - t0,
            )
        results.append(xr.concat(site_results, dim="time"))
        if site_spectra:
            spectras.append(xr.concat(site_spectra, dim="time"))
        del merged

    return (
        xr.concat(results, dim="sites"),
        xr.concat(spectras, dim="sites") if spectras else None,
    )


# ---------------------------------------------------------------------------
# Parallel blocks
# ---------------------------------------------------------------------------
#
# Summation is independent per site and per time step, and pure CPU. Worker
# processes use the "spawn" start method: by this point the caller has usually
# run GP inference on CUDA, and fork()ed children inheriting a CUDA context
# hang. Each worker is limited to one BLAS/OpenMP thread (the parallelism is
# the processes), and gets an equal share of the memory budget.


def _init_worker(log_file: str | None) -> None:
    """Send this module's log records to *log_file* in a worker process."""

    if log_file:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


def _work_items(
    site_ids: list[str], times: np.ndarray, workers: int
) -> list[tuple[list[str], np.ndarray]]:
    """Site groups x time spans for *workers*: sites first, time too when sites are fewer."""

    groups = [g.tolist() for g in np.array_split(np.array(site_ids, dtype=object), min(workers, len(site_ids)))]
    spans = np.array_split(times, max(1, min(len(times), workers // len(groups))))

    return [(g, span) for g in groups for span in spans if len(span)]


def sum_partition_spectra_batched(
    chunks: list[xr.Dataset],
    site_ids: list[str],
    variables: Mapping[str, str],
    *,
    budget_bytes: int,
    workers: int = 1,
    save_spectra: bool = False,
    load: bool = True,
    log_file: str | None = None,
    wind_correction: Mapping[str, Any] | None = None,
) -> tuple[xr.Dataset, xr.Dataset | None]:
    """
    :func:`sum_partition_spectra` over many sites and steps, in memory-sized blocks.

    The work is split into site groups (and time spans when there are fewer
    sites than workers), one per worker; each worker sums its share in site x
    time blocks of its part of the budget (:func:`batch_shape`).

    Parameters
    ----------
    chunks : list of xr.Dataset
        One contribution per goal x partition, each on
        ``(goal, partition, time, sites)``; merged per block.
    site_ids : list of str
        All sites.
    variables : mapping
        Output name -> wavespectra stat.
    budget_bytes : int
        Memory budget of the sum, shared by the workers.
    workers : int, optional
        Worker processes; 1 (default) runs in this process.
    save_spectra : bool, optional
        Also return summed spectra.
    load : bool, optional
        Load results into memory (always True in workers).
    log_file : str, optional
        Log file for worker progress messages.
    wind_correction : mapping, optional
        Keyword arguments of
        :func:`~bluemath_tk.waves.hywaves.wind_correction.wind_sea_contribution`
        (``correction``, ``fetch``, optional ``wind``, ``spr``, ``tp``): adds
        the local wind sea per block, and ``wind_k`` (``K``, 1 where there is
        no correction) to the bulk output. Chunks then also carry
        ``ws_weight`` (and ``u10``, ``u10dir`` without ``wind``).

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset | None]
        Bulk parameters on ``(time, sites)`` and optional spectra.
    """

    if workers <= 1:
        return _sum_blocks(
            chunks, site_ids, variables, save_spectra, budget_bytes, load,
            wind_correction=wind_correction,
        )

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    items = _work_items(site_ids, _time_axis(chunks), workers)
    per_worker = max(1, budget_bytes // len(items))
    logger.info(
        "sum: %d sites across %d workers (%.2f GiB each)",
        len(site_ids),
        len(items),
        per_worker / 1024**3,
    )
    out: dict[int, tuple[xr.Dataset, xr.Dataset | None]] = {}
    with (
        child_threads(1),
        ProcessPoolExecutor(
            max_workers=len(items),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(log_file,),
        ) as executor,
    ):
        futures = {}
        for k, (sites, span) in enumerate(items):
            span_slice = slice(span[0], span[-1])
            future = executor.submit(
                _sum_blocks,
                [c.sel(sites=sites, time=span_slice) for c in chunks],
                sites,
                variables,
                save_spectra,
                per_worker,
                True,
                f"item {k + 1}/{len(items)}",
                wind_correction,
            )
            futures[future] = k
        for future in as_completed(futures):
            out[futures[future]] = future.result()

    by_group: dict[str, list[int]] = {}
    for k, (sites, _) in enumerate(items):
        by_group.setdefault(sites[0], []).append(k)
    results, spectras = [], []
    for ks in by_group.values():
        results.append(xr.concat([out[k][0] for k in ks], dim="time"))
        if save_spectra:
            spectras.append(xr.concat([out[k][1] for k in ks], dim="time"))

    return (
        xr.concat(results, dim="sites"),
        xr.concat(spectras, dim="sites") if spectras else None,
    )
