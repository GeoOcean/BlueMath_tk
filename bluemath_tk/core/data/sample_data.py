import numpy as np
import xarray as xr


def get_2d_dataset():
    # Define the coordinates
    coord1 = np.linspace(-100, 100, 20)
    coord2 = np.linspace(-100, 100, 20)
    coord3 = np.arange(1, 50)

    # Create a meshgrid
    coord1, coord2, coord3 = np.meshgrid(coord1, coord2, coord3, indexing="ij")

    # Create a 3D dataset
    X = (
        np.sin(np.radians(coord1)) * np.cos(np.radians(coord2)) * np.sin(coord3)
        + np.sin(2 * np.radians(coord1))
        * np.cos(2 * np.radians(coord2))
        * np.sin(2 * coord3)
        + np.sin(3 * np.radians(coord1))
        * np.cos(3 * np.radians(coord2))
        * np.sin(3 * coord3)
    )
    # Create a 3D dataset
    Y = -np.sin(X)

    # Create an xarray dataset
    ds = xr.Dataset(
        {
            "X": (["coord1", "coord2", "coord3"], X),
            "Y": (["coord1", "coord2", "coord3"], Y),
        },
        coords={
            "coord1": coord1[:, 0, 0],
            "coord2": coord2[0, :, 0],
            "coord3": coord3[0, 0, :],
        },
    )

    return ds
