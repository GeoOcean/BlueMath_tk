import numpy as np
import matplotlib.pyplot as plt
import rasterio
import cartopy.crs as ccrs
import xarray as xr
from rasterio.transform import from_bounds
from bluemath_tk.core.plotting.colors import hex_colors_land, hex_colors_water
from bluemath_tk.core.plotting.utils import join_colormaps

def plot_bathymetry(rasters, figsize=(10, 8), cbar=False, ax=None):
    """
    Plot a bathymetry map from either a raster file or a NetCDF dataset.

    Parameters
    ----------
    rasters : str or xarray.Dataset
        Either a path to a raster file readable by rasterio or an xarray Dataset.
    figsize : tuple of float, optional
        Figure size in inches, by default ``(10, 8)``.
    cbar : bool, optional
        If ``True``, display a colorbar.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    ax : matplotlib.axes._subplots.AxesSubplot
        The map axis with ``PlateCarree`` projection.

    Examples
    --------
    >>> import xarray as xr
    >>> ds = xr.open_dataset("GEBCO_sample.nc")
    >>> fig, ax = plot_bathymetry(ds)

    >>> fig, ax = plot_bathymetry("path/to/raster.tif")
    """
    if isinstance(rasters, str):
        data = []
        with rasterio.open(rasters) as src:
            raster_data = src.read(1)
            no_data_value = src.nodata
            if no_data_value is not None:
                raster_data = np.ma.masked_equal(raster_data, no_data_value)
            data.append(raster_data)
            transform = src.transform
        height, width = data[0].shape

    elif isinstance(rasters, xr.Dataset):
        data, transform, height, width = nc_to_raster_like(rasters, var_name=None)
    else:
        raise TypeError("Input must be a raster path or an xarray Dataset.")

    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(transform, rows, cols)

    vmin = np.nanmin(data[0])
    vmax = np.nanmax(data[0])

    cmap, norm = join_colormaps(
        cmap1=hex_colors_water,
        cmap2=hex_colors_land,
        value_range1=(vmin, 0.0),
        value_range2=(0.0, vmax),
        name="raster_cmap",
    )

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=figsize)
    else:
        fig = ax.figure
    im = ax.imshow(
        data[0],
        cmap=cmap,
        norm=norm,
        extent=(np.min(xs), np.max(xs), np.min(ys), np.max(ys)),
    )
    if cbar:
        plt.colorbar(im, ax=ax, orientation="vertical", label="Elevation (m)")

    return fig, ax

def find_main_data_variable(ds):
    """
    Find the first variable in the dataset that depends on the coordinate axes.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.

    Returns
    -------
    str or None
        Name of the main data variable if found, otherwise ``None``.
    """
    coord_names = list(ds.coords)
    for var_name, var in ds.data_vars.items():
        var_dims = set(var.dims)
        if all(c in var_dims for c in coord_names):
            return var_name

    best_var = None
    best_score = -1
    for var_name, var in ds.data_vars.items():
        score = len(set(var.dims).intersection(coord_names))
        if score > best_score:
            best_score = score
            best_var = var_name
    return best_var


def nc_to_raster_like(ds, var_name=None):
    """
    Convert an xarray Dataset to a raster-like structure equivalent to a rasterio read.

    This function extracts the main data variable, determines the coordinate
    system (lon/lat), flips the data if needed (north-south inversion),
    and builds the affine transform.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    var_name : str, optional
        Variable name to extract. If ``None``, the first variable depending
        on coordinates will be automatically detected.

    Returns
    -------
    data : list of numpy.ndarray
        List containing the raster-like array.
    transform : affine.Affine
        Affine transform equivalent to rasterio geotransform.
    height : int
        Number of rows.
    width : int
        Number of columns.

    Raises
    ------
    ValueError
        If no suitable variable or coordinate system is found.

    Examples
    --------
    >>> import xarray as xr
    >>> ds = xr.open_dataset("GEBCO_sample.nc")
    >>> data, transform, height, width = nc_to_raster_like(ds)
    """
    if var_name is None:
        var_name = find_main_data_variable(ds)
        if var_name is None:
            raise ValueError("No variable found depending on lat/lon coordinates.")

    da = ds[var_name]
    raster_data = da.values
    if np.isnan(raster_data).any():
        raster_data = np.ma.masked_invalid(raster_data)

    coords_and_vars = list(ds.coords)
    lon_name = next((n for n in coords_and_vars if "lon" in n.lower()), None)
    lat_name = next((n for n in coords_and_vars if "lat" in n.lower()), None)
    if lon_name is None or lat_name is None:
        raise ValueError("Could not detect latitude/longitude coordinates.")

    lon = ds[lon_name].values
    lat = ds[lat_name].values
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()
    width = len(lon)
    height = len(lat)

    if lat[0] < lat[-1]:
        raster_data = raster_data[::-1, :]
        lat_min, lat_max = lat_max, lat_min

    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    data = [raster_data]

    return data, transform, height, width


# =======================
# Example usage
# =======================
if __name__ == "__main__":
    GEBCO = xr.open_dataset(
        "https://dap.ceda.ac.uk/thredds/dodsC/bodc/gebco/global/gebco_2025/ice_surface_elevation/netcdf/GEBCO_2025.nc"
    )
    GEBCO_sel = GEBCO.sel(lon=slice(-4, -3), lat=slice(43, 44))

    EMODnet = xr.open_dataset(
        "https://geoocean.sci.unican.es/thredds/dodsC/geoocean/emodnet-bathy-2024"
    )
    EMODnet_sel = EMODnet.sel(lon=slice(-4, -3), lat=slice(43, 44))

    fig, axes = plt.subplots(
        1, 2,
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(14, 6)
    )

    plot_bathymetry(GEBCO_sel, ax=axes[0])
    axes[0].set_title("GEBCO 2025 Bathymetry")

    plot_bathymetry(EMODnet_sel, ax=axes[1])
    axes[1].set_title("EMODnet 2024 Bathymetry")

    # For raster files:
    # plot_bathymetry("path/to/raster.tif", cbar=True)
