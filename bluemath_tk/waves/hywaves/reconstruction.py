"""
HyWaves reconstruction: linear summation of goal contributions.

Each goal (and spectral partition) contributes nearshore ``hs``, ``tp``,
``dir``, ``spr`` per site and time step (see
:func:`~bluemath_tk.waves.hywaves.metamodel.predict_fields`). Every
contribution becomes a JONSWAP x Cartwright spectrum, the spectra are summed
over goals and partitions, and bulk parameters are computed from the sum
(:func:`sum_partition_spectra`).

The spectral construction allocates the full ``(time, sites, goal x partition,
freq, dir)`` array, so long periods or many sites are processed in site
batches sized to a memory budget (:func:`site_batch_size`), optionally across
worker processes (:func:`sum_partition_spectra_batched`).
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

#: Frequency (Hz) and direction (deg) grids for the summed spectra.
SPECTRAL_FREQ = np.logspace(np.log10(0.035), np.log10(0.5), 29)
SPECTRAL_DIR = np.arange(0.0, 360.0, 5)

#: Spectral partitions expected in an offshore dataset (``phs0``...).
SPECTRAL_PARTITIONS = (0, 1, 2, 3)

#: Default memory budget for one :func:`sum_partition_spectra` call.
DEFAULT_MEMORY_BUDGET_BYTES = 12 * 1024**3

# wavespectra's construct_partition (jonswap x cartwright, then the sum) keeps
# several temporaries of the size of the full (time, sites, parts, freq, dir)
# array alive at once; peaks of at least ~5x that array were measured (a run
# sized for 12 GiB reached 55+ GiB), so batches are sized with this margin.
PEAK_MEMORY_MULTIPLIER = 8


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


def site_batch_size(
    n_times: int,
    n_parts: int,
    budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
    n_freq: int = len(SPECTRAL_FREQ),
    n_dir: int = len(SPECTRAL_DIR),
) -> int:
    """
    Sites per :func:`sum_partition_spectra` call that fit a memory budget.

    Parameters
    ----------
    n_times : int
        Time steps in the call.
    n_parts : int
        Goals x partitions summed.
    budget_bytes : int, optional
        Memory budget. Default is :data:`DEFAULT_MEMORY_BUDGET_BYTES`.
    n_freq, n_dir : int, optional
        Spectral grid size.

    Returns
    -------
    int
        At least 1.
    """

    # float32 spectra (see sum_partition_spectra), times the peak multiplier
    bytes_per_site = (
        max(1, n_times) * max(1, n_parts) * n_freq * n_dir * 4 * PEAK_MEMORY_MULTIPLIER
    )

    return max(1, budget_bytes // bytes_per_site)


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

    from wavespectra.construct import STATS, construct_partition

    parts = cube.sel(sites=site_ids).stack(part=("goal", "partition"))
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

    stats = combined.spec.stats(list(STATS))
    result = xr.Dataset({name: stats[stat] for name, stat in variables.items()})
    if load:
        result = result.load()

    return result, spectra


def _sum_site_batches(
    chunks: list[xr.Dataset],
    site_ids: list[str],
    n_times: int,
    n_parts: int,
    variables: Mapping[str, str],
    save_spectra: bool,
    budget_bytes: int,
    load: bool = True,
    label: str = "",
) -> tuple[xr.Dataset, xr.Dataset | None]:
    """Merge contribution chunks and sum them, site batch by site batch."""

    batch = site_batch_size(n_times, n_parts, budget_bytes)
    n_batches = -(-len(site_ids) // batch)
    results, spectras = [], []
    t0 = time.monotonic()
    for k, i in enumerate(range(0, len(site_ids), batch), start=1):
        t_batch = time.monotonic()
        sites = site_ids[i : i + batch]
        merged = xr.merge([c.sel(sites=sites) for c in chunks])
        result, spectra = sum_partition_spectra(
            merged, sites, variables, save_spectra=save_spectra, load=load
        )
        results.append(result)
        if save_spectra and spectra is not None:
            spectras.append(spectra)
        del merged
        logger.info(
            "%ssum batch %d/%d (%d sites) in %.1fs (%.1fs elapsed)",
            f"[{label}] " if label else "",
            k,
            n_batches,
            len(sites),
            time.monotonic() - t_batch,
            time.monotonic() - t0,
        )

    return (
        xr.concat(results, dim="sites"),
        xr.concat(spectras, dim="sites") if spectras else None,
    )


# ---------------------------------------------------------------------------
# Parallel site batches
# ---------------------------------------------------------------------------
#
# Summation is embarrassingly parallel across sites and pure CPU. Worker
# processes use the "spawn" start method: by this point the caller has usually
# run GP inference on CUDA, and fork()ed children inheriting a CUDA context
# hang. Each worker is limited to one BLAS/OpenMP thread (the parallelism is
# the processes), and gets an equal share of the memory budget.

_BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@contextlib.contextmanager
def _single_threaded_blas_for_children():
    """Set BLAS/OpenMP threads to 1 in the environment workers inherit."""

    previous = {name: os.environ.get(name) for name in _BLAS_THREAD_ENV_VARS}
    try:
        for name in _BLAS_THREAD_ENV_VARS:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _init_worker(log_file: str | None) -> None:
    """Send this module's log records to *log_file* in a worker process."""

    if log_file:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


def sum_partition_spectra_batched(
    chunks: list[xr.Dataset],
    site_ids: list[str],
    n_times: int,
    n_parts: int,
    variables: Mapping[str, str],
    save_spectra: bool = False,
    budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
    workers: int = 1,
    load: bool = True,
    log_file: str | None = None,
) -> tuple[xr.Dataset, xr.Dataset | None]:
    """
    :func:`sum_partition_spectra` over many sites, in memory-sized batches.

    Parameters
    ----------
    chunks : list of xr.Dataset
        One contribution per goal x partition, each on
        ``(goal, partition, time, sites)``; merged per site batch.
    site_ids : list of str
        All sites.
    n_times, n_parts : int
        See :func:`site_batch_size`.
    variables : mapping
        Output name -> wavespectra stat.
    save_spectra : bool, optional
        Also return summed spectra.
    budget_bytes : int, optional
        Total memory budget, shared by the workers.
    workers : int, optional
        Worker processes; 1 (default) runs in this process.
    load : bool, optional
        Load results into memory (always True in workers).
    log_file : str, optional
        Log file for worker progress messages.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset | None]
        Bulk parameters on ``(time, sites)`` and optional spectra.
    """

    if workers <= 1:
        return _sum_site_batches(
            chunks,
            site_ids,
            n_times,
            n_parts,
            variables,
            save_spectra,
            budget_bytes,
            load,
        )

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    per_worker = max(1, budget_bytes // workers)
    groups = [
        g.tolist()
        for g in np.array_split(np.array(site_ids, dtype=object), workers)
        if len(g)
    ]
    logger.info(
        "sum: %d sites across %d workers (%.2f GiB each)",
        len(site_ids),
        len(groups),
        per_worker / 1024**3,
    )
    results: list[xr.Dataset] = []
    spectras: list[xr.Dataset] = []
    with (
        _single_threaded_blas_for_children(),
        ProcessPoolExecutor(
            max_workers=len(groups),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(log_file,),
        ) as executor,
    ):
        futures: dict[Any, str] = {}
        for k, sites in enumerate(groups, start=1):
            label = f"group {k}/{len(groups)}"
            future = executor.submit(
                _sum_site_batches,
                [c.sel(sites=sites) for c in chunks],
                sites,
                n_times,
                n_parts,
                variables,
                save_spectra,
                per_worker,
                True,
                label,
            )
            futures[future] = label
        for future in as_completed(futures):
            result, spectra = future.result()
            results.append(result)
            if save_spectra and spectra is not None:
                spectras.append(spectra)
            logger.info("%s done", futures[future])

    return (
        xr.concat(results, dim="sites"),
        xr.concat(spectras, dim="sites") if spectras else None,
    )
