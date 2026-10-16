"""
Project: BlueMath_tk
Sub-Module: core.resources
Author: GeoOcean Research Group, Universidad de Cantabria
Repository: https://github.com/GeoOcean/BlueMath_tk.git
Status: Under development (Working)

What this machine has, and how to share it out: cores, free memory and GPU
(:func:`machine`), the part a job may take (:data:`USAGE`), thread limits for
this process and for child processes, and a pool of single-purpose processes
(:func:`run_processes`).
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import psutil

logger = logging.getLogger(__name__)

#: Fraction of the machine a job may take (cores, free memory, GPU memory).
USAGE = 0.75

GIB = 1024**3

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class Machine:
    """
    Resources of this machine, as seen now.

    Attributes
    ----------
    cores : int
        CPU cores this process may run on.
    memory : int
        Free memory (bytes).
    memory_total : int
        Total memory (bytes).
    gpu : str or None
        CUDA device name, ``None`` without one.
    gpu_memory : int
        Memory of the CUDA device (bytes), 0 without one.
    """

    cores: int
    memory: int
    memory_total: int
    gpu: str | None = None
    gpu_memory: int = 0

    def usable(self, usage: float = USAGE) -> tuple[int, int]:
        """
        Cores and memory a job may take.

        Parameters
        ----------
        usage : float, optional
            Fraction of the cores and of the *free* memory. Default
            :data:`USAGE`.

        Returns
        -------
        tuple[int, int]
            Cores (at least 1) and memory (bytes).
        """
        return max(1, int(self.cores * usage)), int(self.memory * usage)

    def __str__(self) -> str:
        gpu = f"GPU {self.gpu} ({self.gpu_memory / GIB:.0f} GiB)" if self.gpu else "no GPU"
        return (
            f"{self.cores} cores, {self.memory / GIB:.0f} GiB free of "
            f"{self.memory_total / GIB:.0f} GiB, {gpu}"
        )


def machine() -> Machine:
    """
    Cores, free memory and CUDA device of this machine.

    Returns
    -------
    Machine
        The resources seen now (cores this process may use, e.g. inside a
        SLURM allocation).
    """

    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    vm = psutil.virtual_memory()
    gpu, gpu_memory = None, 0
    with contextlib.suppress(ImportError):
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu, gpu_memory = props.name, int(props.total_memory)

    return Machine(int(cores or 1), int(vm.available), int(vm.total), gpu, gpu_memory)


@contextlib.contextmanager
def child_threads(n: int) -> Iterator[None]:
    """Limit the BLAS/OpenMP threads of the processes started inside the block to *n*."""

    previous = {name: os.environ.get(name) for name in _THREAD_ENV_VARS}
    try:
        for name in _THREAD_ENV_VARS:
            os.environ[name] = str(n)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextlib.contextmanager
def limit_threads(n: int) -> Iterator[None]:
    """Limit the BLAS/OpenMP (and PyTorch, if loaded) threads of this process to *n* inside the block."""

    from threadpoolctl import threadpool_limits

    torch = None
    with contextlib.suppress(ImportError):
        import torch
    previous = torch.get_num_threads() if torch else None
    try:
        with threadpool_limits(limits=n):
            if torch:
                torch.set_num_threads(n)
            yield
    finally:
        if torch:
            torch.set_num_threads(previous)


def _init_child(threads: int, log_file: str | None) -> None:
    """Thread limit (PyTorch included) and log file of a pool process."""

    with contextlib.suppress(ImportError):
        import torch

        torch.set_num_threads(threads)
    if log_file:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)


def run_processes(
    fn: Callable[..., Any],
    jobs: Sequence[tuple[Any, ...]],
    *,
    workers: int,
    threads: int = 1,
    log_file: str | None = None,
) -> list[Any]:
    """
    ``fn(*job)`` for every job, in *workers* processes of *threads* threads each.

    Processes are started with ``spawn`` (safe after CUDA or threads were
    used in this process); *fn* must be importable (a module-level function).
    With one worker the jobs run here, in order, limited to *threads*.

    ``spawn`` re-imports the main module in every process: a *script* that
    calls this (directly or through a library) must run its work under
    ``if __name__ == "__main__":``, or each process runs the script again.
    Notebooks need nothing.

    Parameters
    ----------
    fn : callable
        Module-level function.
    jobs : sequence of tuple
        Positional arguments of each call.
    workers : int
        Processes.
    threads : int, optional
        BLAS/OpenMP/PyTorch threads per process. Default is 1.
    log_file : str, optional
        File the processes log to (``INFO``).

    Returns
    -------
    list
        Results, in the order of *jobs*.
    """

    if workers <= 1 or len(jobs) <= 1:
        with limit_threads(threads):
            return [fn(*job) for job in jobs]

    results: list[Any] = [None] * len(jobs)
    with (
        child_threads(threads),
        ProcessPoolExecutor(
            max_workers=min(workers, len(jobs)),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_child,
            initargs=(threads, log_file),
        ) as pool,
    ):
        futures = {pool.submit(fn, *job): i for i, job in enumerate(jobs)}
        for done, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            logger.info("%s: %d/%d jobs done", getattr(fn, "__name__", "job"), done, len(jobs))

    return results


__all__ = ["GIB", "USAGE", "Machine", "child_threads", "limit_threads", "machine", "run_processes"]
