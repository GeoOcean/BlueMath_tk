import numpy as np
import psutil
import xarray as xr
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


def get_available_cpus() -> int:
    """
    Get the available CPU cores in the system.

    Returns
    -------
    int
        The number of available CPU cores.
    """

    return int(psutil.cpu_count() * 0.5)


def calculate_optimal_chunks(
    dataset: xr.Dataset,
    cpu_cores_to_use: int = None,
    total_ram_percentage_to_use: float = 0.5,
    full_dims: list = None,
) -> dict:
    """
    Calculate optimal chunk sizes for each variable in an xarray Dataset.
    NOTE: This function is not beign used in the code, as it is a first
          approximation to test how we could chunk given the hardware.

    Parameters
    ----------
    dataset : xr.Dataset
        The input dataset containing multiple variables.
    cpu_cores_to_use : int, optional
        Number of CPU cores to use. If None, half of available cores are used.
        Default is None.
    total_ram_percentage_to_use : float, optional
        Fraction of total RAM to use for chunking. Default is 0.5.
    full_dims : list, optional
        List of dimension names that should use all values (not be chunked).
        Default is None.

    Returns
    -------
    dict
        Dictionary with variable names as keys and chunk dictionaries as values.
        Example: {'var1': {'time': 1000, 'lat': 50, 'lon': 50}}
    """

    # Get number of available CPU cores if not specified
    cpu_cores_to_use = cpu_cores_to_use or get_available_cpus()

    # Get available memory for chunking
    available_mem = get_available_ram()
    target_bytes = (available_mem * total_ram_percentage_to_use) / cpu_cores_to_use

    full_dims = full_dims or []
    chunks_dict = {}

    # Process each variable in the dataset
    for var_name, da in dataset.data_vars.items():
        # Get shape and dtype info
        shape = da.shape
        dims = da.dims
        dtype = da.dtype
        bytes_per_elem = np.dtype(dtype).itemsize

        # Separate chunked and full dimensions
        chunk_dims = [d for d in dims if d not in full_dims]

        # Calculate elements for chunked dimensions only
        if chunk_dims:
            # Calculate total elements considering full dimensions
            full_dims_size = np.prod(
                [s for d, s in zip(dims, shape) if d in full_dims], dtype=np.float64
            )
            total_chunk_elements = target_bytes / (bytes_per_elem * full_dims_size)

            # Calculate base chunk size for remaining dimensions
            chunk_size = int(np.power(total_chunk_elements, 1 / len(chunk_dims)))
        else:
            chunk_size = 0  # Not used if all dimensions are full

        # Create chunks dictionary for this variable
        var_chunks = {}
        for dim_name, dim_size in zip(dims, shape):
            if dim_name in full_dims:
                var_chunks[dim_name] = dim_size  # Use full dimension
            else:
                var_chunks[dim_name] = min(chunk_size, dim_size)

        chunks_dict[var_name] = var_chunks

    return chunks_dict


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
        n_workers = get_available_cpus()
    if isinstance(memory_limit, float):
        memory_limit *= get_available_ram() / get_total_ram()

    cluster = LocalCluster(
        n_workers=n_workers, threads_per_worker=1, memory_limit=memory_limit
    )
    client = Client(cluster)

    return client
