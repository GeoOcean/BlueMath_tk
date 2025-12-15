import numpy as np
import xarray as xr


def generate_grid_parameters(
    bathy_data: xr.DataArray,
    alpc: float = 0,
    xpc: float = None,
    ypc: float = None,
    xlenc: float = None,
    ylenc: float = None,
    buffer_distance: float = None,
) -> dict:
    """
    Generate grid parameters for the SWAN model based on bathymetry.

    Parameters
    ----------
    bathy_data : xr.DataArray
                Bathymetry data.
        Must have the following dimensions:
        - lon/x: longitude or x coordinate
        - lat/y: latitude or y coordinate
    alpc: float
        Computational Grid Rotation angle in degrees.
    xpc: float
        X origin.
    ypc: float
        Y origin.

    Returns
    -------
    dict
        Dictionary with grid configuration for SWAN input.
    """

    # Determine coordinate system based on coordinate names
    coord_names = list(bathy_data.coords)

    # Get coordinate variables
    if any(name in ["lon", "longitude"] for name in coord_names):
        x_coord = next(name for name in coord_names if name in ["lon", "longitude"])
        y_coord = next(name for name in coord_names if name in ["lat", "latitude"])
    else:
        x_coord = next(
            name for name in coord_names if name in ["x", "X", "cx", "easting"]
        )
        y_coord = next(
            name for name in coord_names if name in ["y", "Y", "cy", "northing"]
        )

    # Get resolution from cropped data
    grid_resolution_x = abs(
        bathy_data[x_coord][1].values - bathy_data[x_coord][0].values
    )
    grid_resolution_y = abs(
        bathy_data[y_coord][1].values - bathy_data[y_coord][0].values
    )

    if alpc != 0:
        angle_rad = np.radians(alpc)
        # Create rotation matrix
        R = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )

        # Create unrotated rectangle corners
        dx = np.array([0, xlenc, xlenc, 0, 0])
        dy = np.array([0, 0, ylenc, ylenc, 0])
        points = np.column_stack([dx, dy])

        # Rotate points
        rotated = np.dot(points, R.T)

        # Translate to corner position
        x = rotated[:, 0] + xpc
        y = rotated[:, 1] + ypc
        corners = np.column_stack([x, y])

        x_min = np.min(x) - buffer_distance
        x_max = np.max(x) + buffer_distance
        y_min = np.min(y) - buffer_distance
        y_max = np.max(y) + buffer_distance

        print(f"Cropping bounds: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")

        # Crop bathymetry
        cropped = bathy_data.sel(
            {
                x_coord: slice(x_min, x_max),
                y_coord: slice(y_max, y_min),
            }  # Note: slice from max to min for descending coordinates
        )

        grid_parameters = {
            "xpc": xpc,
            "ypc": ypc,
            "alpc": alpc,
            "xlenc": xlenc,
            "ylenc": ylenc,
            "mxc": int(np.round(xlenc / grid_resolution_x) - 1),
            "myc": int(np.round(ylenc / grid_resolution_y) - 1),
            "xpinp": np.nanmin(cropped[x_coord]),  # x origin from cropped data
            "ypinp": np.nanmin(cropped[y_coord]),  # y origin from cropped data
            "alpinp": 0,  # x-axis direction
            "mxinp": len(cropped[x_coord]) - 1,  # number mesh x from cropped data
            "myinp": len(cropped[y_coord]) - 1,  # number mesh y from cropped data
            "dxinp": grid_resolution_x,  # resolution from cropped data
            "dyinp": grid_resolution_y,  # resolution from cropped data
        }
        return grid_parameters, cropped, corners

    else:
        # Compute parameters from full bathymetry
        grid_parameters = {
            "xpc": float(np.nanmin(bathy_data[x_coord])),  # origin x
            "ypc": float(np.nanmin(bathy_data[y_coord])),  # origin y
            "alpc": alpc,  # x-axis direction
            "xlenc": float(
                np.nanmax(bathy_data[x_coord]) - np.nanmin(bathy_data[x_coord])
            ),  # grid length x
            "ylenc": float(
                np.nanmax(bathy_data[y_coord]) - np.nanmin(bathy_data[y_coord])
            ),  # grid length y
            "mxc": len(bathy_data[x_coord]) - 1,  # num mesh x
            "myc": len(bathy_data[y_coord]) - 1,  # num mesh y
            "xpinp": float(np.nanmin(bathy_data[x_coord])),  # origin x
            "ypinp": float(np.nanmin(bathy_data[y_coord])),  # origin y
            "alpinp": 0,  # x-axis direction
            "mxinp": len(bathy_data[x_coord]) - 1,  # num mesh x
            "myinp": len(bathy_data[y_coord]) - 1,  # num mesh y
            "dxinp": float(
                abs(bathy_data[x_coord][1].values - bathy_data[x_coord][0].values)
            ),  # resolution x
            "dyinp": float(
                abs(bathy_data[y_coord][1].values - bathy_data[y_coord][0].values)
            ),  # resolution y
        }
        return grid_parameters
