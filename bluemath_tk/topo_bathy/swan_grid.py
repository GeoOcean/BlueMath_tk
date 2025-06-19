import numpy as np
import xarray as xr


def generate_grid_parameters(
    bathy_data: xr.DataArray,
    buffer_distance: float = None,
) -> dict:
    """
    Generate grid parameters for the SWAN model based on bathymetry.

    Parameters
    ----------
    bathy_data : xr.DataArray
        Bathymetry data with dimensions 'lat' and 'lon'.


    Returns
    -------
    dict
        Dictionary with grid configuration for SWAN input.

    Raises
    ------
    ValueError
        If coord_type is not 'geographic' or 'cartesian'.

    Contact
    -------
    @bellidog on GitHub
    """

    """
    Generate the grid parameters for the SWAN model.

    Parameters
    ----------
    bathy_data : xr.DataArray
        Bathymetry data.
        Must have the following dimensions:
        - lon/x: longitude or x coordinate
        - lat/y: latitude or y coordinate

    Returns
    -------
    dict
        Grid parameters for the SWAN model.
    """

    # Determine coordinate system based on coordinate names
    coord_names = list(bathy_data.coords)

    # Get coordinate variables
    if any(name in ["lon", "longitude"] for name in coord_names):
        x_coord = next(name for name in coord_names if name in ["lon", "longitude"])
        y_coord = next(name for name in coord_names if name in ["lat", "latitude"])
        # coord_type = 'geographic'
    else:
        x_coord = next(
            name for name in coord_names if name in ["x", "X", "cx", "easting"]
        )
        y_coord = next(
            name for name in coord_names if name in ["y", "Y", "cy", "northing"]
        )
    #     coord_type = 'cartesian'

    # if coord_type not in ["geographic", "cartesian"]:
    #     raise ValueError("coord_type must be 'geographic' or 'cartesian'.")

    # use_int = coord_type == "cartesian"
    # cast = int if use_int else float

    # Get the main parameters from user input
    print("Please enter the following parameters:")
    alpc = float(input("Enter rotation angle in degrees (alpc): "))

    # Get resolution from cropped data
    grid_resolution_x = abs(
        bathy_data[x_coord][1].values - bathy_data[x_coord][0].values
    )
    grid_resolution_y = abs(
        bathy_data[y_coord][1].values - bathy_data[y_coord][0].values
    )

    if alpc != 0:
        xpc = float(input("Enter x origin (xpc): "))
        ypc = float(input("Enter y origin (ypc): "))
        xlenc = float(input("Enter grid length in x (xlenc): "))
        ylenc = float(input("Enter grid length in y (ylenc): "))

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

        # TODO: This is a temporary buffer distance, it should be adjusted to the actual grid size
        # Get bounds with buffer
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

        fixed_parameters = {
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
        return fixed_parameters, cropped

    else:
        # Compute parameters from full bathymetry
        return {
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
            "alpinp": alpc,  # x-axis direction
            "mxinp": len(bathy_data[x_coord]) - 1,  # num mesh x
            "myinp": len(bathy_data[y_coord]) - 1,  # num mesh y
            "dxinp": float(
                abs(bathy_data[x_coord][1].values - bathy_data[x_coord][0].values)
            ),  # resolution x
            "dyinp": float(
                abs(bathy_data[y_coord][1].values - bathy_data[y_coord][0].values)
            ),  # resolution y
        }
