import datetime as datetime
import warnings
from typing import Tuple

import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.path import Path
from tqdm import tqdm

from ..core.plotting.colors import hex_colors_land, hex_colors_water
from ..core.plotting.utils import join_colormaps


def get_regular_grid(
    node_computation_longitude: np.ndarray,
    node_computation_latitude: np.ndarray,
    node_computation_elements: np.ndarray,
    factor: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a regular grid based on the node computation longitude and latitude.
    The grid is defined by the minimum and maximum longitude and latitude values,
    and the minimum distance between nodes in both dimensions.
    The grid is generated with a specified factor to adjust the resolution.

    Parameters
    ----------
    node_computation_longitude : np.ndarray
        1D array of longitudes for the nodes.
    node_computation_latitude : np.ndarray
        1D array of latitudes for the nodes.
    node_computation_elements : np.ndarray
        2D array of indices defining the elements (triangles).
    factor : float, optional
        A scaling factor to adjust the resolution of the grid.

    Returns
    -------
    lon_grid : np.ndarray
        1D array of longitudes defining the grid.
    lat_grid : np.ndarray
        1D array of latitudes defining the grid.
    """

    lon_min, lon_max = (
        node_computation_longitude.min(),
        node_computation_longitude.max(),
    )
    lat_min, lat_max = node_computation_latitude.min(), node_computation_latitude.max()

    lon_tri = node_computation_longitude[node_computation_elements]
    lat_tri = node_computation_latitude[node_computation_elements]

    dlon01 = np.abs(lon_tri[:, 0] - lon_tri[:, 1])
    dlon12 = np.abs(lon_tri[:, 1] - lon_tri[:, 2])
    dlon20 = np.abs(lon_tri[:, 2] - lon_tri[:, 0])
    min_dx = np.min(np.stack([dlon01, dlon12, dlon20], axis=1).max(axis=1)) * factor

    dlat01 = np.abs(lat_tri[:, 0] - lat_tri[:, 1])
    dlat12 = np.abs(lat_tri[:, 1] - lat_tri[:, 2])
    dlat20 = np.abs(lat_tri[:, 2] - lat_tri[:, 0])
    min_dy = np.min(np.stack([dlat01, dlat12, dlat20], axis=1).max(axis=1)) * factor

    lon_grid = np.arange(lon_min, lon_max + min_dx, min_dx)
    lat_grid = np.arange(lat_min, lat_max + min_dy, min_dy)

    return lon_grid, lat_grid


def generate_structured_points(
    triangle_connectivity: np.ndarray,
    node_lon: np.ndarray,
    node_lat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate structured points for each triangle in the mesh.
    Each triangle will have 10 points: vertices, centroid, midpoints of edges,
    and midpoints of vertex-centroid segments.

    Parameters
    ----------
    triangle_connectivity : np.ndarray
        Array of shape (n_triangles, 3) containing indices of the vertices for each triangle.
    node_lon : np.ndarray
        Array of shape (n_nodes,) containing the longitudes of the nodes.
    node_lat : np.ndarray
        Array of shape (n_nodes,) containing the latitudes of the nodes.

    Returns
    -------
    lon_all : np.ndarray
        Array of shape (n_triangles, 10) containing the longitudes of the structured points for each triangle.
    lat_all : np.ndarray
        Array of shape (n_triangles, 10) containing the latitudes of the structured points for each triangle.
    """

    n_tri = triangle_connectivity.shape[0]
    lon_all = np.empty((n_tri, 10))
    lat_all = np.empty((n_tri, 10))

    for i, tri in enumerate(triangle_connectivity):
        A = np.array([node_lon[tri[0]], node_lat[tri[0]]])
        B = np.array([node_lon[tri[1]], node_lat[tri[1]]])
        C = np.array([node_lon[tri[2]], node_lat[tri[2]]])

        G = (A + B + C) / 3
        M_AB = (A + B) / 2
        M_BC = (B + C) / 2
        M_CA = (C + A) / 2
        M_AG = (A + G) / 2
        M_BG = (B + G) / 2
        M_CG = (C + G) / 2

        points = [A, B, C, G, M_AB, M_BC, M_CA, M_AG, M_BG, M_CG]
        lon_all[i, :] = [pt[0] for pt in points]
        lat_all[i, :] = [pt[1] for pt in points]

    return lon_all, lat_all


def plot_GS_input_wind_partition(
    xds_vortex_GS: xr.Dataset,
    xds_vortex_interp: xr.Dataset,
    ds_GFD_info: xr.Dataset,
    i_time: int = 0,
    figsize=(10, 8),
) -> None:
    """
    Plot the wind partition for GreenSurge input data.

    Parameters
    ----------
    xds_vortex_GS : xr.Dataset
        Dataset containing the vortex model data for GreenSurge.
    xds_vortex_interp : xr.Dataset
        Dataset containing the interpolated vortex model data.
    ds_GFD_info : xr.Dataset
        Dataset containing the GreenSurge forcing information.
    i_time : int, optional
        Index of the time step to plot. Default is 0.
    figsize : tuple, optional
        Figure size. Default is (10, 8).
    """

    simple_quiver = 20
    scale = 30
    width = 0.003

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    time = xds_vortex_GS.time.isel(time=i_time)
    fig.suptitle(
        f"Wind partition for {time.values.astype('datetime64[s]').astype(str)}",
        fontsize=16,
    )
    ax1.set_title("Vortex wind")
    ax2.set_title("Wind partition (GreenSurge)")

    # Plotting the wind speed
    W = xds_vortex_interp.W.isel(time=i_time)
    Dir = (270 - xds_vortex_interp.Dir.isel(time=i_time)) % 360

    triangle_forcing_connectivity = ds_GFD_info.triangle_forcing_connectivity.values
    node_forcing_longitude = ds_GFD_info.node_forcing_longitude.values
    node_forcing_latitude = ds_GFD_info.node_forcing_latitude.values

    Lon = xds_vortex_GS.lon
    Lat = xds_vortex_GS.lat

    W_reg = xds_vortex_GS.W.isel(time=i_time)
    Dir_reg = (270 - xds_vortex_GS.Dir.isel(time=i_time)) % 360

    vmin = np.min((W.min(), W_reg.min()))
    vmax = np.max((W.max(), W_reg.max()))

    cmap = join_colormaps("plasma_r", "viridis", name="wind_partition_cmap")
    # 100, "plasma_r", 0.05, 0.9, "viridis", 0.2, 1

    ax2.tripcolor(
        node_forcing_longitude,
        node_forcing_latitude,
        triangle_forcing_connectivity,
        facecolors=W,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolor="white",
        shading="flat",
        transform=ccrs.PlateCarree(),
    )

    pm1 = ax1.pcolormesh(
        Lon,
        Lat,
        W_reg,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        transform=ccrs.PlateCarree(),
    )
    cbar = fig.colorbar(
        pm1, ax=(ax1, ax2), orientation="horizontal", pad=0.03, aspect=50
    )
    cbar.set_label(
        "{0} ({1})".format("Wind", "m.s⁻¹"),
        rotation=0,
        va="bottom",
        fontweight="bold",
        labelpad=15,
    )

    ax2.quiver(
        np.mean(node_forcing_longitude[triangle_forcing_connectivity], axis=1),
        np.mean(node_forcing_latitude[triangle_forcing_connectivity], axis=1),
        np.cos(np.deg2rad(Dir)),
        np.sin(np.deg2rad(Dir)),
        color="black",
        scale=scale,
        width=width,
        transform=ccrs.PlateCarree(),
    )

    ax1.quiver(
        Lon[::simple_quiver],
        Lat[::simple_quiver],
        (np.cos(np.deg2rad(Dir_reg)))[::simple_quiver, ::simple_quiver],
        (np.sin(np.deg2rad(Dir_reg)))[::simple_quiver, ::simple_quiver],
        color="black",
        scale=scale,
        width=width,
        transform=ccrs.PlateCarree(),
    )

    ax1.set_extent([Lon.min(), Lon.max(), Lat.min(), Lat.max()], crs=ccrs.PlateCarree())
    ax2.set_extent([Lon.min(), Lon.max(), Lat.min(), Lat.max()], crs=ccrs.PlateCarree())

    ax1.coastlines()
    ax2.coastlines()


def plot_greensurge_setup(
    info_ds: xr.Dataset, figsize: tuple = (10, 10)
) -> Tuple[Figure, Axes]:
    """
    Plot the GreenSurge mesh setup from the provided dataset.

    Parameters
    ----------
    info_ds : xr.Dataset
        Dataset containing the mesh information.
    figsize : tuple, optional
        Figure size. Default is (10, 10).

    Returns
    -------
    fig : Figure
        Figure object.
    ax : Axes
        Axes object.
    """

    # Extracting data from the dataset
    Conectivity = info_ds.triangle_forcing_connectivity.values
    node_forcing_longitude = info_ds.node_forcing_longitude.values
    node_forcing_latitude = info_ds.node_forcing_latitude.values
    node_computation_longitude = info_ds.node_computation_longitude.values
    node_computation_latitude = info_ds.node_computation_latitude.values

    num_elements = len(Conectivity)

    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=figsize,
        constrained_layout=True,
    )

    ax.triplot(
        node_computation_longitude,
        node_computation_latitude,
        info_ds.triangle_computation_connectivity.values,
        color="grey",
        linestyle="-",
        marker="",
        linewidth=1 / 2,
        label="Computational mesh",
    )
    ax.triplot(
        node_forcing_longitude,
        node_forcing_latitude,
        Conectivity,
        color="green",
        linestyle="-",
        marker="",
        linewidth=1,
        label=f"Forcing mesh ({num_elements} elements)",
    )

    for t in range(num_elements):
        node0, node1, node2 = Conectivity[t]
        x = (
            node_forcing_longitude[int(node0)]
            + node_forcing_longitude[int(node1)]
            + node_forcing_longitude[int(node2)]
        ) / 3
        y = (
            node_forcing_latitude[int(node0)]
            + node_forcing_latitude[int(node1)]
            + node_forcing_latitude[int(node2)]
        ) / 3
        plt.text(
            x, y, f"T{t}", fontsize=10, ha="center", va="center", fontweight="bold"
        )

    bnd = [
        min(node_computation_longitude.min(), node_forcing_longitude.min()),
        max(node_computation_longitude.max(), node_forcing_longitude.max()),
        min(node_computation_latitude.min(), node_forcing_latitude.min()),
        max(node_computation_latitude.max(), node_forcing_latitude.max()),
    ]
    ax.set_extent([*bnd], crs=ccrs.PlateCarree())
    plt.legend()
    ax.set_title("GreenSurge Mesh Setup")
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    return fig, ax


def create_triangle_mask_from_points(
    lon: np.ndarray, lat: np.ndarray, triangle: np.ndarray
) -> np.ndarray:
    """
    Create a mask indicating which scattered points are inside a triangle.

    Parameters
    ----------
    lon : np.ndarray
        1D array of longitudes of the points.
    lat : np.ndarray
        1D array of latitudes of the points.
    triangle : np.ndarray
        (3, 2) array containing the triangle vertices as (lon, lat) pairs.

    Returns
    -------
    mask : np.ndarray
        1D boolean array of same length as lon/lat indicating points inside the triangle.
    """

    points = np.column_stack((lon, lat))  # Shape (N, 2)
    triangle_path = Path(triangle)
    mask = triangle_path.contains_points(points)

    return mask


def plot_GS_vs_dynamic_windsetup_swath(
    ds_WL_GS_WindSetUp: xr.Dataset,
    ds_WL_dynamic_WindSetUp: xr.Dataset,
    ds_gfd_metadata: xr.Dataset,
    vmin: float = None,
    vmax: float = None,
    figsize: tuple = (10, 8),
) -> None:
    """
    Plot the GreenSurge and dynamic wind setup from the provided datasets.

    Parameters
    ----------
    ds_WL_GS_WindSetUp : xr.Dataset
        Dataset containing the GreenSurge wind setup data.
    ds_WL_dynamic_WindSetUp : xr.Dataset
        Dataset containing the dynamic wind setup data.
    ds_gfd_metadata : xr.Dataset
        Dataset containing the metadata for the GFD mesh.
    vmin : float, optional
        Minimum value for the color scale. Default is None.
    vmax : float, optional
        Maximum value for the color scale. Default is None.
    figsize : tuple, optional
        Figure size. Default is (10, 8).
    """

    warnings.filterwarnings("ignore", message="All-NaN slice encountered")

    X = ds_gfd_metadata.node_computation_longitude.values
    Y = ds_gfd_metadata.node_computation_latitude.values
    triangles = ds_gfd_metadata.triangle_computation_connectivity.values

    Longitude_dynamic = ds_WL_dynamic_WindSetUp.mesh2d_node_x.values
    Latitude_dynamic = ds_WL_dynamic_WindSetUp.mesh2d_node_y.values

    xds_GS = np.nanmax(ds_WL_GS_WindSetUp["WL"].values, axis=0)
    xds_DY = np.nanmax(ds_WL_dynamic_WindSetUp["mesh2d_s1"].values, axis=0)

    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = float(np.nanmax(xds_GS))

    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )

    axs[0].tripcolor(
        Longitude_dynamic,
        Latitude_dynamic,
        ds_WL_dynamic_WindSetUp.mesh2d_face_nodes.values - 1,
        facecolors=xds_DY,
        cmap="CMRmap_r",
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
    )

    pm = axs[1].tripcolor(
        X,
        Y,
        triangles,
        facecolors=xds_GS,
        cmap="CMRmap_r",
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
    )
    cbar = fig.colorbar(pm, ax=axs, orientation="horizontal", pad=0.03, aspect=50)
    cbar.set_label(
        "WL ({})".format("m"), rotation=0, va="bottom", fontweight="bold", labelpad=15
    )
    fig.suptitle("SWATH", fontsize=18, fontweight="bold")

    axs[0].set_title("Dynamic Wind SetUp", fontsize=14)
    axs[1].set_title("GreenSurge Wind SetUp", fontsize=14)

    lon_min = np.nanmin(Longitude_dynamic)
    lon_max = np.nanmax(Longitude_dynamic)
    lat_min = np.nanmin(Latitude_dynamic)
    lat_max = np.nanmax(Latitude_dynamic)
    for ax in axs:
        ax.set_extent([lon_min, lon_max, lat_min, lat_max])


def GS_windsetup_reconstruction_with_postprocess(
    greensurge_dataset: xr.Dataset,
    ds_gfd_metadata: xr.Dataset,
    wind_direction_input: xr.Dataset,
    velocity_thresholds: np.ndarray = np.array([0, 100, 100]),
    drag_coefficients: np.ndarray = np.array([0.00063, 0.00723, 0.00723]),
) -> xr.Dataset:
    """
    Reconstructs the GreenSurge wind setup using the provided wind direction input and metadata.

    Parameters
    ----------
    greensurge_dataset : xr.Dataset
        xarray Dataset containing the GreenSurge mesh and forcing data.
    ds_gfd_metadata: xr.Dataset
        xarray Dataset containing metadata for the GFD mesh.
    wind_direction_input: xr.Dataset
        xarray Dataset containing wind direction and speed data.
    velocity_thresholds : np.ndarray
        Array of velocity thresholds for drag coefficient calculation.
    drag_coefficients : np.ndarray
        Array of drag coefficients corresponding to the velocity thresholds.

    Returns
    -------
    xr.Dataset
        xarray Dataset containing the reconstructed wind setup.
    """

    velocity_thresholds = np.asarray(velocity_thresholds)
    drag_coefficients = np.asarray(drag_coefficients)

    direction_bins = ds_gfd_metadata.wind_directions.values
    forcing_cell_indices = greensurge_dataset.forcing_cell.values
    wind_speed_reference = ds_gfd_metadata.wind_speed.values.item()
    base_drag_coeff = GS_LinearWindDragCoef_mat(
        wind_speed_reference, drag_coefficients, velocity_thresholds
    )
    time_step_hours = ds_gfd_metadata.time_step_hours.values

    time_start = wind_direction_input.time.values.min()
    time_end = wind_direction_input.time.values.max()
    duration_in_steps = (
        int((ds_gfd_metadata.simulation_duration_hours.values) / time_step_hours) + 1
    )
    output_time_vector = np.arange(
        time_start, time_end, np.timedelta64(int(60 * time_step_hours.item()), "m")
    )
    num_output_times = len(output_time_vector)

    direction_data = wind_direction_input.Dir.values
    wind_speed_data = wind_direction_input.W.values

    n_faces = greensurge_dataset["mesh2d_s1"].isel(forcing_cell=0, direction=0).shape
    wind_setup_output = np.zeros((num_output_times, n_faces[1]))
    water_level_accumulator = np.zeros(n_faces)

    for time_index in tqdm(range(num_output_times), desc="Processing time steps"):
        water_level_accumulator[:] = 0
        for cell_index in forcing_cell_indices.astype(int):
            current_dir = direction_data[cell_index, time_index] % 360
            adjusted_bins = np.where(direction_bins == 0, 360, direction_bins)
            closest_direction_index = np.abs(adjusted_bins - current_dir).argmin()

            water_level_case = (
                greensurge_dataset["mesh2d_s1"]
                .sel(forcing_cell=cell_index, direction=closest_direction_index)
                .values
            )
            water_level_case = np.nan_to_num(water_level_case, nan=0)

            wind_speed_value = wind_speed_data[cell_index, time_index]
            drag_coeff_value = GS_LinearWindDragCoef_mat(
                wind_speed_value, drag_coefficients, velocity_thresholds
            )

            scaling_factor = (wind_speed_value**2 / wind_speed_reference**2) * (
                drag_coeff_value / base_drag_coeff
            )
            water_level_accumulator += water_level_case * scaling_factor

        step_window = min(duration_in_steps, num_output_times - time_index)
        if (num_output_times - time_index) > step_window:
            wind_setup_output[time_index : time_index + step_window] += (
                water_level_accumulator
            )
        else:
            shift_counter = step_window - (num_output_times - time_index)
            wind_setup_output[
                time_index : time_index + step_window - shift_counter
            ] += water_level_accumulator[: step_window - shift_counter]

    ds_wind_setup = xr.Dataset(
        {"WL": (["time", "nface"], wind_setup_output)},
        coords={
            "time": output_time_vector,
            "nface": np.arange(wind_setup_output.shape[1]),
        },
    )
    ds_wind_setup.attrs["description"] = "Wind setup from GreenSurge methodology"

    return ds_wind_setup


def GS_LinearWindDragCoef_mat(
    Wspeed: np.ndarray, CD_Wl_abc: np.ndarray, Wl_abc: np.ndarray
) -> np.ndarray:
    """
    Calculate the linear drag coefficient based on wind speed and specified thresholds.

    Parameters
    ----------
    Wspeed : np.ndarray
        Wind speed values (1D array).
    CD_Wl_abc : np.ndarray
        Coefficients for the drag coefficient calculation, should be a 1D array of length 3.
    Wl_abc : np.ndarray
        Wind speed thresholds for the drag coefficient calculation, should be a 1D array of length 3.

    Returns
    -------
    np.ndarray
        Calculated drag coefficient values based on the input wind speed.
    """

    Wspeed = np.atleast_1d(Wspeed).astype(np.float64)
    was_scalar = Wspeed.ndim == 1 and Wspeed.size == 1

    Wla, Wlb, Wlc = Wl_abc
    CDa, CDb, CDc = CD_Wl_abc

    if Wla != Wlb:
        a_ab = (CDa - CDb) / (Wla - Wlb)
        b_ab = CDb - a_ab * Wlb
    else:
        a_ab = 0
        b_ab = CDa

    if Wlb != Wlc:
        a_bc = (CDb - CDc) / (Wlb - Wlc)
        b_bc = CDc - a_bc * Wlc
    else:
        a_bc = 0
        b_bc = CDb

    a_cinf = 0
    b_cinf = CDc

    CD = a_cinf * Wspeed + b_cinf
    CD[Wspeed <= Wlb] = a_ab * Wspeed[Wspeed <= Wlb] + b_ab
    mask_bc = (Wspeed > Wlb) & (Wspeed <= Wlc)
    CD[mask_bc] = a_bc * Wspeed[mask_bc] + b_bc

    return CD.item() if was_scalar else CD


def plot_GS_vs_dynamic_windsetup(
    ds_WL_GS_WindSetUp: xr.Dataset,
    ds_WL_dynamic_WindSetUp: xr.Dataset,
    ds_gfd_metadata: xr.Dataset,
    time: datetime.datetime,
    vmin: float = None,
    vmax: float = None,
    figsize: tuple = (10, 8),
) -> None:
    """
    Plot the GreenSurge and dynamic wind setup from the provided datasets.

    Parameters
    ----------
    ds_WL_GS_WindSetUp: xr.Dataset
        xarray Dataset containing the GreenSurge wind setup data.
    ds_WL_dynamic_WindSetUp: xr.Dataset
        xarray Dataset containing the dynamic wind setup data.
    ds_gfd_metadata: xr.Dataset
        xarray Dataset containing the metadata for the GFD mesh.
    time: datetime.datetime
        The time point at which to plot the data.
    vmin: float, optional
        Minimum value for the color scale. Default is None.
    vmax: float, optional
        Maximum value for the color scale. Default is None.
    figsize: tuple, optional
        Tuple specifying the figure size. Default is (10, 8).
    """

    warnings.filterwarnings("ignore", message="All-NaN slice encountered")

    X = ds_gfd_metadata.node_computation_longitude.values
    Y = ds_gfd_metadata.node_computation_latitude.values
    triangles = ds_gfd_metadata.triangle_computation_connectivity.values

    Longitude_dynamic = ds_WL_dynamic_WindSetUp.mesh2d_node_x.values
    Latitude_dynamic = ds_WL_dynamic_WindSetUp.mesh2d_node_y.values

    xds_GS = ds_WL_GS_WindSetUp["WL"].sel(time=time).values
    xds_DY = ds_WL_dynamic_WindSetUp["mesh2d_s1"].sel(time=time).values
    if vmin is None or vmax is None:
        vmax = float(np.nanmax(xds_GS)) * 0.5
        vmin = -vmax

    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )

    axs[0].tripcolor(
        Longitude_dynamic,
        Latitude_dynamic,
        ds_WL_dynamic_WindSetUp.mesh2d_face_nodes.values - 1,
        facecolors=xds_DY,
        cmap="bwr",
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
    )

    pm = axs[1].tripcolor(
        X,
        Y,
        triangles,
        facecolors=xds_GS,
        cmap="bwr",
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
    )
    cbar = fig.colorbar(pm, ax=axs, orientation="horizontal", pad=0.03, aspect=50)
    cbar.set_label(
        "WL ({})".format("m"), rotation=0, va="bottom", fontweight="bold", labelpad=15
    )
    fig.suptitle(
        f"Wind SetUp for {time.astype('datetime64[s]').astype(str)}",
        fontsize=16,
        fontweight="bold",
    )

    axs[0].set_title("Dynamic Wind SetUp")
    axs[1].set_title("GreenSurge Wind SetUp")

    lon_min = np.nanmin(Longitude_dynamic)
    lon_max = np.nanmax(Longitude_dynamic)
    lat_min = np.nanmin(Latitude_dynamic)
    lat_max = np.nanmax(Latitude_dynamic)
    for ax in axs:
        ax.set_extent([lon_min, lon_max, lat_min, lat_max])


def plot_GS_TG_validation_timeseries(
    ds_WL_GS_WindSetUp: xr.Dataset,
    ds_WL_GS_IB: xr.Dataset,
    ds_WL_dynamic_WindSetUp: xr.Dataset,
    tide_gauge: xr.Dataset,
    ds_GFD_info: xr.Dataset,
    figsize: tuple = (15, 7),
    WLmin: float = None,
    WLmax: float = None,
) -> None:
    """
    Plot a time series comparison of GreenSurge, dynamic wind setup, and tide gauge data with a bathymetry map.

    Parameters
    ----------
    ds_WL_GS_WindSetUp : xr.Dataset
        Dataset containing GreenSurge wind setup data with dimensions (nface, time).
    ds_WL_GS_IB : xr.Dataset
        Dataset containing inverse barometer data with dimensions (lat, lon, time).
    ds_WL_dynamic_WindSetUp : xr.Dataset
        Dataset containing dynamic wind setup data with dimensions (mesh2d_nFaces, time).
    tide_gauge : xr.Dataset
        Dataset containing tide gauge data with dimensions (time).
    ds_GFD_info : xr.Dataset
        Dataset containing grid information with longitude and latitude coordinates.
    figsize : tuple, optional
        Size of the figure for the plot. Default is (15, 7).
    WLmin : float, optional
        Minimum water level for the plot. Default is None.
    WLmax : float, optional
        Maximum water level for the plot. Default is None.
    """

    lon_obs = tide_gauge.lon.values
    lat_obs = tide_gauge.lat.values
    lon_obs = np.where(lon_obs > 180, lon_obs - 360, lon_obs)

    nface_index = int(
        extract_pos_nearest_points_tri(ds_GFD_info, lon_obs, lat_obs).item()
    )
    mesh2d_nFaces = int(
        extract_pos_nearest_points_tri(ds_WL_dynamic_WindSetUp, lon_obs, lat_obs).item()
    )
    pos_lon_IB, pos_lat_IB = extract_pos_nearest_points(ds_WL_GS_IB, lon_obs, lat_obs)

    time = ds_WL_GS_WindSetUp.WL.time
    ds_WL_dynamic_WindSetUp = ds_WL_dynamic_WindSetUp.sel(time=time)
    ds_WL_GS_IB = ds_WL_GS_IB.interp(time=time)

    WL_GS = ds_WL_GS_WindSetUp.WL.sel(nface=nface_index).values
    WL_dyn = ds_WL_dynamic_WindSetUp.mesh2d_s1.sel(mesh2d_nFaces=mesh2d_nFaces).values
    WL_IB = ds_WL_GS_IB.IB.values[int(pos_lat_IB.item()), int(pos_lon_IB.item()), :]
    WL_TG = tide_gauge.SS.values

    WL_SS_dyn = WL_dyn + WL_IB
    WL_SS_GS = WL_GS + WL_IB

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3], figure=fig)

    ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    X = ds_WL_dynamic_WindSetUp.mesh2d_node_x.values
    Y = ds_WL_dynamic_WindSetUp.mesh2d_node_y.values
    triangles = ds_WL_dynamic_WindSetUp.mesh2d_face_nodes.values.astype(int) - 1
    Z = np.mean(ds_WL_dynamic_WindSetUp.mesh2d_node_z.values[triangles], axis=1)

    ax_map.tripcolor(
        X,
        Y,
        triangles,
        facecolors=Z,
        cmap="RdYlBu_r",
        shading="flat",
        transform=ccrs.PlateCarree(),
    )
    ax_map.scatter(
        lon_obs,
        lat_obs,
        color="red",
        marker="x",
        transform=ccrs.PlateCarree(),
        label="Tide Gauge",
    )
    ax_map.set_extent([X.min(), X.max(), Y.min(), Y.max()], crs=ccrs.PlateCarree())
    ax_map.set_title("Bathymetry Map")
    ax_map.legend(loc="upper right", fontsize="small")

    ax_ts = fig.add_subplot(gs[1])
    time_vals = time.values
    ax_ts.plot(time_vals, WL_SS_dyn, c="blue", label="Dynamic simulation")
    ax_ts.plot(time_vals, WL_SS_GS, c="tomato", label="GreenSurge")
    ax_ts.plot(tide_gauge.time.values, WL_TG, c="green", label="Tide Gauge")
    ax_ts.plot(time_vals, WL_GS, c="grey", label="GS WindSetup")
    ax_ts.plot(time_vals, WL_IB, c="black", label="Inverse Barometer")

    if WLmin is None or WLmax is None:
        WLmax = max(np.nanmax(WL_SS_dyn), np.nanmax(WL_SS_GS), np.nanmax(WL_TG))
        WLmin = min(np.nanmin(WL_SS_dyn), np.nanmin(WL_SS_GS), np.nanmin(WL_TG))

    ax_ts.set_xlim(time_vals[0], time_vals[-1])
    ax_ts.set_ylim(WLmin, WLmax)
    ax_ts.set_ylabel("Water Level (m)")
    ax_ts.set_title("Tide Gauge Validation")
    ax_ts.legend()

    plt.tight_layout()
    plt.show()


def extract_pos_nearest_points_tri(
    ds_mesh_info: xr.Dataset, lon_points: np.ndarray, lat_points: np.ndarray
) -> np.ndarray:
    """
    Extract the nearest triangle index for given longitude and latitude points.

    Parameters
    ----------
    ds_mesh_info : xr.Dataset
        Dataset containing mesh information with longitude and latitude coordinates.
    lon_points : np.ndarray
        Array of longitudes for which to find the nearest triangle index.
    lat_points : np.ndarray
        Array of latitudes for which to find the nearest triangle index.
    Returns
    -------
    np.ndarray
        Array of nearest triangle indices corresponding to the input longitude and latitude points.
    """

    if "node_forcing_latitude" in ds_mesh_info.variables:
        lon_mesh = ds_mesh_info.node_computation_longitude.values
        lat_mesh = ds_mesh_info.node_computation_latitude.values
        type_ds = 0
    else:
        lon_mesh = ds_mesh_info.mesh2d_face_x.values
        lat_mesh = ds_mesh_info.mesh2d_face_y.values
        type_ds = 1

    nface_index = np.zeros(len(lon_points))

    for i in range(len(lon_points)):
        lon = lon_points[i]
        lat = lat_points[i]

        distances = np.sqrt((lon_mesh - lon) ** 2 + (lat_mesh - lat) ** 2)
        min_idx = np.argmin(distances)

        if type_ds == 0:
            nface_index[i] = ds_mesh_info.node_cumputation_index.values[min_idx]
        elif type_ds == 1:
            nface_index[i] = ds_mesh_info.mesh2d_nFaces.values[min_idx]

    return nface_index


def extract_pos_nearest_points(
    ds_mesh_info: xr.Dataset, lon_points: np.ndarray, lat_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the nearest point indices for given longitude and latitude points in a mesh dataset.

    Parameters
    ----------
    ds_mesh_info : xr.Dataset
        Dataset containing mesh information with longitude and latitude coordinates.
    lon_points : np.ndarray
        Array of longitudes for which to find the nearest point indices.
    lat_points : np.ndarray
        Array of latitudes for which to find the nearest point indices.

    Returns
    -------
    pos_lon_points_mesh : np.ndarray
        Array of longitude indices corresponding to the input longitude points in the mesh.
    pos_lat_points_mesh : np.ndarray
        Array of latitude indices corresponding to the input latitude points in the mesh.
    """

    lon_mesh = ds_mesh_info.lon.values
    lat_mesh = ds_mesh_info.lat.values

    pos_lon_points_mesh = np.zeros(len(lon_points))
    pos_lat_points_mesh = np.zeros(len(lat_points))

    for i in range(len(lon_points)):
        lon = lon_points[i]
        lat = lat_points[i]

        lat_index = np.nanargmin((lat - lat_mesh) ** 2)
        lon_index = np.nanargmin((lon - lon_mesh) ** 2)

        pos_lon_points_mesh[i] = lon_index
        pos_lat_points_mesh[i] = lat_index

    return pos_lon_points_mesh, pos_lat_points_mesh


def pressure_to_IB(xds_presure: xr.Dataset) -> xr.Dataset:
    """
    Convert pressure data in a dataset to inverse barometer (IB) values.

    Parameters
    ----------
    xds_presure : xr.Dataset
        Dataset containing pressure data with dimensions (lat, lon, time).

    Returns
    -------
    xr.Dataset
        Dataset with an additional variable 'IB' representing the inverse barometer values.
    """

    p = xds_presure.p.values
    IB = (p - 1013.25) * -1 / 100  # Convert pressure (hPa) to inverse barometer (m)

    xds_presure_modified = xds_presure.copy()
    xds_presure_modified["IB"] = (("lat", "lon", "time"), IB)

    return xds_presure_modified
