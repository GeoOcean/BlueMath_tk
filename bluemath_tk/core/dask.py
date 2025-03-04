import os

import psutil
from dask.distributed import Client, LocalCluster


def get_total_ram() -> int:
    """
    Get the total RAM in the system.

    Returns
    -------
    int
        The total RAM in bytes.
    """

    return psutil.virtual_memory().total


def get_available_ram() -> int:
    """
    Get the available RAM in the system.

    Returns
    -------
    int
        The available RAM in bytes.
    """

    return psutil.virtual_memory().available


def setup_dask_client(n_workers: int = None, memory_limit: str = 0.5):
    """
    Setup a Dask client with controlled resources.

    Parameters
    ----------
    n_workers : int, optional
        Number of workers. Default is None.
    memory_limit : str, optional
        Memory limit per worker. Default is 0.5.

    Returns
    -------
    Client
        Dask distributed client

    Notes
    -----
    - Resources might vary depending on the hardware and the load of the machine.
      Be very careful when setting the number of workers and memory limit, as it
      might affect the performance of the machine, or in the worse case scenario,
      the performance of other users in the same machine (cluster case).
    """

    if n_workers is None:
        n_workers = int(os.environ.get("BLUEMATH_NUM_WORKERS", "1"))
    if isinstance(memory_limit, float):
        memory_limit *= get_available_ram() / get_total_ram()

    cluster = LocalCluster(
        n_workers=n_workers, threads_per_worker=1, memory_limit=memory_limit
    )
    client = Client(cluster)

    return client
