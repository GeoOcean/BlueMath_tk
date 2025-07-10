import os.path as op
from typing import Dict

from siphon.catalog import TDSCatalog

GEOOCEAN_CLUSTER_DATA = "/lustre/geocean/DATA/"
GEOOCEAN_THREDDS_DATA = "https://geoocean.sci.unican.es/thredds/dodsC/geoceanData/"

# Default paths dictionary
PATHS = {
    "SHYTCWAVES_COEFS": op.join(
        GEOOCEAN_THREDDS_DATA,
        "GEOOCEAN/SHyTCWaves_bulk/ibtracs_coef_pmin_wmax.nc",
    ),
    "SHYTCWAVES_BULK": op.join(
        GEOOCEAN_THREDDS_DATA,
        "GEOOCEAN/SHyTCWaves_bulk/library_shytcwaves_bulk_params_int32.nc",
    ),
    "SHYTCWAVES_MDA": op.join(
        GEOOCEAN_THREDDS_DATA,
        "GEOOCEAN/SHyTCWaves_bulk/shytcwaves_mda.nc",
    ),
    "SHYTCWAVES_MDA_INDICES": op.join(
        GEOOCEAN_THREDDS_DATA,
        "GEOOCEAN/SHyTCWaves_bulk/shytcwaves_mda_indices.nc",
    ),
    "SHYTCWAVES_MDA_MASK_INDICES": op.join(
        GEOOCEAN_THREDDS_DATA,
        "GEOOCEAN/SHyTCWaves_bulk/mda_mask_indices.nc",
    ),
    "SHYTCWAVES_MDA_MASK_INDICES_LOWRES": op.join(
        GEOOCEAN_THREDDS_DATA,
        "GEOOCEAN/SHyTCWaves_bulk/mda_mask_indices_lowres.nc",
    ),
    "SHYTCWAVES_SPECTRA": op.join(
        GEOOCEAN_CLUSTER_DATA,
        "GEOOCEAN/SHyTCWaves/",
    ),
}


def update_paths(new_paths: dict) -> None:
    """
    Update the paths dictionary with new values.

    Parameters
    ----------
    new_paths : dict
        Dictionary containing new path values to update.

    Examples
    --------
    >>> update_paths({"MY_PATH": "/new/path/to/data"})
    """

    PATHS.update(new_paths)


def get_paths(verbose: bool = True) -> dict:
    """
    Get the paths dictionary.

    Parameters
    ----------
    verbose : bool, optional
        Whether to print warnings about Thredds paths. Default is True.

    Returns
    -------
    dict
        Dictionary containing the paths.
    """

    if verbose:
        for file_name, file_path in PATHS.items():
            if "thredds" in file_path:
                print(f"WARNING: {file_name} is a Thredds path.")
            elif "lustre" in file_path:
                print(f"WARNING: {file_name} is a GeoOcean cluster path.")
                print("* Data access can be requested to bluemath@unican.es")
        print(
            "You can update any path or add new paths with the update_paths function,"
            " from bluemath_tk.config.paths."
        )
        print("Example: update_paths({'SHYTCWAVES_COEFS': '/new/path/to/data'})")

    return PATHS


def get_thredds_catalog() -> TDSCatalog:
    """
    Get the Thredds catalog object.

    Returns
    -------
    TDSCatalog
        Siphon TDSCatalog object containing the catalog information.
    """

    catalog_url = (
        "https://geoocean.sci.unican.es/thredds/catalog/geoceanData/catalog.xml"
    )

    return TDSCatalog(catalog_url)


def get_catalog_folders() -> Dict[str, str]:
    """
    Get a dictionary of folder names and their links from the first level of the catalog.

    Returns
    -------
    Dict[str, str]
        Dictionary with folder names as keys and their catalog URLs as values.
    """

    catalog = get_thredds_catalog()
    folders = {}

    for name, ref in catalog.catalog_refs.items():
        folders[ref.title] = ref.href

    return folders


def print_catalog_table() -> None:
    """
    Print a formatted table of available folders in the catalog.
    """

    folders = get_catalog_folders()

    # Print header
    print("\nAvailable Folders in GeoOcean Thredds Catalog:")
    print("-" * 80)
    print(f"{'Folder Name':<20} | {'Catalog URL':<40}")
    print("-" * 80)

    # Print each folder
    for name, url in sorted(folders.items()):
        print(f"{name:<20} | {url:<40}")

    print("-" * 80)


if __name__ == "__main__":
    print_catalog_table()
