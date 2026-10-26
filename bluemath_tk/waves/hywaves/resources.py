"""
Project: BlueMath_tk
Sub-Module: waves.hywaves.resources
Author: GeoOcean Research Group, Universidad de Cantabria
Repository: https://github.com/GeoOcean/BlueMath_tk.git
Status: Under development (Working)

How much of the machine a HyWaves fit or reconstruction uses.

The plans are **proposals**: :func:`plan_fit` and :func:`plan_reconstruction`
size the work to at most :data:`~bluemath_tk.core.resources.USAGE` (75%) of
the cores and free memory of this machine
(:func:`~bluemath_tk.core.resources.machine`) and return a small dataclass.
The caller shows it, may edit any field, and passes it to the run::

    res = plan_reconstruction(sites=4971, months=12, parts=40)
    res.workers = 20                     # the user decides
    # ... goal contributions with res.device / res.threads,
    # sum_partition_spectra_batched(..., workers=res.workers,
    #                               budget_bytes=sum_budget(res.memory, held, res.workers))

How the work is split, and why (measured on a 48-core Xeon Gold 5418Y node,
2026-10-15):

- **Fit**: every (goal, metamodel set, MDA size) PCA + GP fit is independent
  and small (one exact GP per principal component on a few hundred points).
  PyTorch uses threads poorly on such small problems (36 threads were 1.9x
  faster than one), so on CPU the fits run as parallel single-thread
  processes (:class:`FitResources`); a fit needs <= ~1.2 GiB.
- **Reconstruction**, one call = *months* x *sites*:

  1. goal contributions: the GP of every goal x partition in its own
     process (``threads // 2`` processes of 2 threads, 512-row kernels;
     ~5x faster than one threaded process), then the PCA inverse to the
     sites, threaded, in the calling process (``device``, ``threads``);
  2. spectral sum: JONSWAP x Cartwright spectra summed per site and hour in
     ``workers`` single-thread processes. Every site and hour is
     independent, so the sum is split into site x time blocks
     (:func:`~bluemath_tk.waves.hywaves.reconstruction.batch_shape`) and
     each worker needs 0.5-2 GiB whatever the call. The sum is bound by
     memory bandwidth: small blocks are fastest, and 8 / 18 / 36 workers
     are 7.1 / 10.9 / 14.1x faster than one.

  :func:`call_memory` is the memory of one call, :func:`hours` a rough wall
  time from the measured rates.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.resources import GIB, USAGE, Machine, machine
from .reconstruction import spectra_bytes

#: Longest calendar month of an hourly hindcast (time steps).
STEPS_PER_MONTH = 744

#: Most months per reconstruction call: a year on CPU (one GP call and one
#: worker pool, ~6 s to start, per call); on a GPU, the value checked in
#: spectral mode with all goals.
MONTHS_PER_CALL = {"cpu": 12, "cuda": 4}

# Goal contributions (hs, tp, dir, spr; float64) are held by this process and,
# split by site, by the sum workers: twice their size.
_CONTRIBUTION_BYTES = 4 * 8 * 2

# Saved spectra: the workers' results, their concatenation and the copy made
# when stored.
_SPECTRA_COPIES = 3

#: Spectral-sum block memory per worker when there is room: 1-5 sites per
#: block summed fastest (one month, 10 contributions; 17 sites was 40% slower).
SUM_BYTES_PER_WORKER = 2 * GIB

#: The least a worker gets (one site over ~190 steps of 40 contributions).
MIN_SUM_BYTES_PER_WORKER = GIB // 2

#: Memory of a sum worker process itself (imports, its share of the inputs).
WORKER_BYTES = 3 * GIB // 4

# Xeon Gold 5418Y. Sum (2026-10-15), one core: building and summing the
# spectra per step x site x contribution, plus the bulk stats per step x site;
# workers slow each other down (memory bandwidth): speedup = w / (1 +
# _CONTENTION w), measured 7.1 / 10.9 / 14.1 with 8 / 18 / 36 workers.
# GP step (2026-10-16, MDA 550, 18 processes x 2 threads, 512-row kernels):
# ~6 s per call to start the processes, then per month x contribution (GP)
# and per site x month x contribution (PCA inverse, 36 threads).
_SUM_SECONDS_PER_PART = 2.2e-5
_SUM_SECONDS_PER_SITE = 1.0e-4
_CONTENTION = 0.04
_GP_SECONDS_PER_CALL = 6.0
_GP_SECONDS_PER_MONTH_PART = 0.05
_GP_SECONDS_PER_SITE_MONTH_PART = 2.9e-4

#: Cost of a sum worker pool (s): starting the processes (~6 s) and sending
#: them the contributions; one site-year sum took 11-18 s in a pool for
#: ~0.5 s of work.
POOL_SECONDS = 10.0


@dataclass
class ReconstructionResources:
    """
    Resources of a reconstruction, from :func:`plan_reconstruction`.

    Edit any field before using it, e.g. fewer ``workers`` to leave cores to
    other users of a shared machine.

    Attributes
    ----------
    device : str
        PyTorch device of the GP prediction (``"cpu"`` or ``"cuda"``).
    threads : int
        Threads of the GP prediction and PCA inverse (calling process).
    workers : int
        Processes of the spectral sum, one thread each.
    months : int
        Calendar months per call (more months: fewer GP calls and worker
        pools, more memory for the contributions).
    sites : int or None
        Sites per call; ``None`` for every site (fewer only when memory is
        short, e.g. many sites with ``save_spectra``).
    memory : int
        Most memory a call may take (bytes). The contributions and saved
        spectra are held first; the sum workers get up to
        :data:`SUM_BYTES_PER_WORKER` each of what is left.
    """

    device: str = "cpu"
    threads: int = 1
    workers: int = 1
    months: int = 1
    sites: int | None = None
    memory: int = 8 * GIB

    def __str__(self) -> str:
        sites = "every site" if self.sites is None else f"{self.sites} sites"
        return (
            f"device {self.device}, {self.threads} threads (GP + PCA), {self.workers} workers "
            f"(spectral sum), {self.months} month(s) x {sites} per call, "
            f"{self.memory / GIB:.0f} GiB"
        )


@dataclass
class FitResources:
    """
    Resources of the metamodel fits, from :func:`plan_fit`.

    Edit any field before using it.

    Attributes
    ----------
    device : str
        PyTorch device of the GP fits (``"cpu"`` or ``"cuda"``).
    workers : int
        Fits run at the same time, one process each.
    threads : int
        Threads per process (1 on CPU: the parallelism is the processes).
    """

    device: str = "cpu"
    workers: int = 1
    threads: int = 1

    def __str__(self) -> str:
        return f"device {self.device}, {self.workers} process(es) x {self.threads} thread(s)"


def call_memory(
    months: int,
    sites: int,
    parts: int,
    *,
    workers: int,
    save_spectra: bool = False,
    sum_per_worker: int = MIN_SUM_BYTES_PER_WORKER,
) -> dict[str, int]:
    """
    Memory of one reconstruction call (bytes), by part.

    - ``contributions``: the predicted ``hs``, ``tp``, ``dir``, ``spr`` of
      every contribution (float64), held by the caller and, split, by the
      workers;
    - ``spectra``: the summed spectra when kept (float32; workers' results,
      their concatenation and the stored copy);
    - ``sum``: each worker process and its spectral blocks.

    Parameters
    ----------
    months, sites : int
        Calendar months and sites of the call.
    parts : int
        Contributions summed per site (goals x partitions, + 1 with the wind
        correction).
    workers : int
        Sum processes.
    save_spectra : bool, optional
        The summed spectra are kept.
    sum_per_worker : int, optional
        Block memory per worker; default the least it can get.

    Returns
    -------
    dict
        ``contributions``, ``spectra``, ``sum`` (workers and their blocks)
        and their ``total``.
    """

    steps = months * STEPS_PER_MONTH
    out = {
        "contributions": steps * sites * parts * _CONTRIBUTION_BYTES,
        "spectra": _SPECTRA_COPIES * spectra_bytes(steps, sites) if save_spectra else 0,
        "sum": workers * (WORKER_BYTES + sum_per_worker),
    }
    out["total"] = sum(out.values())

    return out


def sum_budget(memory: int, held: int, workers: int) -> int:
    """
    Memory budget of one spectral sum (bytes).

    Up to :data:`SUM_BYTES_PER_WORKER` per worker (blocks this size summed
    fastest), within what *held* and the worker processes leave of *memory*.

    Parameters
    ----------
    memory : int
        Memory of the whole call (``ReconstructionResources.memory``).
    held : int
        Memory already taken by the contributions (and saved spectra).
    workers : int
        Sum processes.

    Returns
    -------
    int
        Bytes for ``sum_partition_spectra_batched(budget_bytes=...)``; 0
        when nothing is left (the blocks then shrink to one site-hour).
    """

    return max(0, min(workers * SUM_BYTES_PER_WORKER, memory - held - workers * WORKER_BYTES))


def _single_core_seconds(sites: int, steps: int, parts: int) -> float:
    return sites * steps * (_SUM_SECONDS_PER_PART * parts + _SUM_SECONDS_PER_SITE)


def _speedup(workers: int) -> float:
    return workers / (1 + _CONTENTION * workers)


def sum_workers(sites: int, steps: int, parts: int, workers: int) -> int:
    """
    Workers worth using for one spectral sum.

    *workers* when the time they save beats :data:`POOL_SECONDS`, else 1
    (the sum runs in the calling process; e.g. one site over a year).

    Parameters
    ----------
    sites, steps : int
        Sites and time steps summed.
    parts : int
        Contributions per site.
    workers : int
        Workers available (``ReconstructionResources.workers``).

    Returns
    -------
    int
        *workers* or 1.
    """

    single = _single_core_seconds(sites, steps, parts)
    saved = single - single / _speedup(max(1, workers))

    return workers if saved > POOL_SECONDS else 1


def hours(site_months: int, months: int, parts: int, workers: int) -> float:
    """
    Rough wall time (hours) of a reconstruction.

    From rates measured on one 48-core Xeon Gold 5418Y node (GP processes and
    PCA, the sum slowed by memory bandwidth as workers grow); other machines
    differ, so take it as an order of magnitude.

    Parameters
    ----------
    site_months : int
        Site-months to compute.
    months : int
        Calendar months they span.
    parts : int
        Contributions per site (goals x partitions, + 1 with wind).
    workers : int
        Sum processes.

    Returns
    -------
    float
        Hours.
    """

    per_site_month = _single_core_seconds(1, STEPS_PER_MONTH, parts)
    seconds = site_months * (per_site_month / _speedup(max(1, workers)) + _GP_SECONDS_PER_SITE_MONTH_PART * parts)
    calls = -(-months // MONTHS_PER_CALL["cpu"])
    seconds += calls * _GP_SECONDS_PER_CALL + months * parts * _GP_SECONDS_PER_MONTH_PART

    return seconds / 3600


def _halvings(n: int) -> list[int]:
    out = [n]
    while out[-1] > 1:
        out.append(-(-out[-1] // 2))
    return out


def plan_reconstruction(
    sites: int,
    months: int,
    parts: int,
    *,
    save_spectra: bool = False,
    host: Machine | None = None,
    usage: float = USAGE,
) -> ReconstructionResources:
    """
    Proposed resources to reconstruct *months* x *sites* on this machine.

    One sum worker per usable core, and the largest call that fits *usage*
    of the free memory (:func:`call_memory`, with the least block memory per
    worker): every site and up to :data:`MONTHS_PER_CALL` months, then fewer
    months, then fewer sites, and only then fewer workers.

    Parameters
    ----------
    sites, months : int
        Sites and calendar months to compute.
    parts : int
        Contributions per site (see :func:`call_memory`).
    save_spectra : bool, optional
        The summed spectra are kept.
    host : Machine, optional
        The machine; default :func:`bluemath_tk.core.resources.machine`.
    usage : float, optional
        Fraction of the machine to use. Default :data:`USAGE`.

    Returns
    -------
    ReconstructionResources

    Raises
    ------
    MemoryError
        If one site and one month do not fit.
    """

    host = host or machine()
    cores, memory = host.usable(usage)
    device = "cuda" if host.gpu else "cpu"
    sites, months = max(1, sites), max(1, months)

    def fits(n_sites: int, n_months: int, workers: int) -> bool:
        need = call_memory(n_months, n_sites, parts, workers=workers, save_spectra=save_spectra)
        return need["total"] <= memory

    for n_sites in _halvings(sites):
        for n_months in range(min(MONTHS_PER_CALL[device], months), 0, -1):
            if fits(n_sites, n_months, cores):
                return ReconstructionResources(
                    device, cores, cores, n_months, None if n_sites == sites else n_sites, memory
                )
    for workers in range(cores - 1, 0, -1):
        if fits(1, 1, workers):
            return ReconstructionResources(device, cores, workers, 1, 1, memory)
    need = call_memory(1, 1, parts, workers=1, save_spectra=save_spectra)["total"]
    raise MemoryError(
        f"One site and one month need {need / GIB:.1f} GiB; {memory / GIB:.1f} GiB usable"
    )


def plan_fit(
    jobs: int, job_memory: int, *, host: Machine | None = None, usage: float = USAGE
) -> FitResources:
    """
    Proposed resources for *jobs* independent fits of *job_memory* bytes each.

    On CPU: one single-thread process per fit, as many at a time as *usage*
    of the cores and of the free memory allow. With a GPU: one process on it,
    with *usage* of the cores as threads.

    Parameters
    ----------
    jobs : int
        Fits to run.
    job_memory : int
        Memory of one fit (bytes).
    host : Machine, optional
        The machine; default :func:`bluemath_tk.core.resources.machine`.
    usage : float, optional
        Fraction of the machine to use. Default :data:`USAGE`.

    Returns
    -------
    FitResources
    """

    host = host or machine()
    cores, memory = host.usable(usage)
    if host.gpu:
        return FitResources("cuda", 1, cores)

    return FitResources("cpu", max(1, min(cores, jobs, memory // max(1, job_memory))), 1)


__all__ = [
    "MONTHS_PER_CALL",
    "STEPS_PER_MONTH",
    "FitResources",
    "MIN_SUM_BYTES_PER_WORKER",
    "POOL_SECONDS",
    "ReconstructionResources",
    "SUM_BYTES_PER_WORKER",
    "WORKER_BYTES",
    "call_memory",
    "hours",
    "plan_fit",
    "plan_reconstruction",
    "sum_budget",
    "sum_workers",
]
