import warnings
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.path import Path
from netCDF4 import Dataset
from tqdm import tqdm

from ..core.operations import get_degrees_from_uv
from ..core.plotting.colors import hex_colors_land, hex_colors_water
from ..core.plotting.utils import join_colormaps

# from ..topo_bathy.mesh_utils import read_adcirc_grd


def read_adcirc_grd(grd_file: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Reads the ADCIRC grid file and returns the node and element data.

    Parameters
    ----------
    grd_file : str
        Path to the ADCIRC grid file.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str]]
        A tuple containing:
        - Nodes (np.ndarray): An array of shape (nnodes, 3) containing the coordinates of each node.
        - Elmts (np.ndarray): An array of shape (nelmts, 3) containing the element connectivity,
            with node indices adjusted (decremented by 1).
        - lines (List[str]): The remaining lines in the file after reading the nodes and elements.

    Examples
    --------
    >>> nodes, elmts, lines = read_adcirc_grd("path/to/grid.grd")
    >>> print(nodes.shape, elmts.shape, len(lines))
    (1000, 3) (500, 3) 10
    """

    with open(grd_file, "r") as f:
        _header0 = f.readline()
        header1 = f.readline()
        header_nums = list(map(float, header1.split()))
        nelmts = int(header_nums[0])
        nnodes = int(header_nums[1])

        Nodes = np.loadtxt(f, max_rows=nnodes)
        Elmts = np.loadtxt(f, max_rows=nelmts) - 1
        lines = f.readlines()

    return Nodes, Elmts, lines


def calculate_edges(Elmts: np.ndarray) -> np.ndarray:
    """
    Calculates the unique edges from the given triangle elements.

    Parameters
    ----------
    Elmts : np.ndarray
        A 2D array of shape (nelmts, 3) containing the node indices for each triangle element.

    Returns
    -------
    np.ndarray
        A 2D array of shape (n_edges, 2) containing the unique edges,
        each represented by a pair of node indices.
    """

    perc = 0
    Links = np.zeros((len(Elmts) * 3, 2), dtype=int)
    tel = 0
    for ii, elmt in enumerate(Elmts):
        if round(100 * (ii / len(Elmts))) != perc:
            perc = round(100 * (ii / len(Elmts)))
        Links[tel] = [elmt[0], elmt[1]]
        tel += 1
        Links[tel] = [elmt[1], elmt[2]]
        tel += 1
        Links[tel] = [elmt[2], elmt[0]]
        tel += 1

    Links_sorted = np.sort(Links, axis=1)
    Links_unique = np.unique(Links_sorted, axis=0)

    return Links_unique


def adcirc2DFlowFM(Path_grd: str, netcdf_path: str) -> None:
    """
    Converts ADCIRC grid data to a NetCDF Delft3DFM format.

    Parameters
    ----------
    Path_grd : str
        Path to the ADCIRC grid file.
    netcdf_path : str
        Path where the resulting NetCDF file will be saved.

    Examples
    --------
    >>> adcirc2DFlowFM("path/to/grid.grd", "path/to/output.nc")
    >>> print("NetCDF file created successfully.")
    """

    Nodes_full, Elmts_full, lines = read_adcirc_grd(Path_grd)
    NODE = Nodes_full[:, [1, 2, 3]]
    EDGE = Elmts_full[:, [2, 3, 4]]
    edges = calculate_edges(EDGE) + 1
    EDGE_S = np.sort(EDGE, axis=1)
    EDGE_S = EDGE_S[EDGE_S[:, 2].argsort()]
    EDGE_S = EDGE_S[EDGE_S[:, 1].argsort()]
    face_node = np.array(EDGE_S[EDGE_S[:, 0].argsort()], dtype=np.int32)
    edge_node = np.zeros([len(edges), 2], dtype="i4")
    edge_face = np.zeros([len(edges), 2], dtype=np.double)
    edge_x = np.zeros(len(edges))
    edge_y = np.zeros(len(edges))

    edge_node = np.array(
        edge_node,
        dtype=np.int32,
    )

    face_x = (
        NODE[EDGE[:, 0].astype(int), 0]
        + NODE[EDGE[:, 1].astype(int), 0]
        + NODE[EDGE[:, 2].astype(int), 0]
    ) / 3
    face_y = (
        NODE[EDGE[:, 0].astype(int), 1]
        + NODE[EDGE[:, 1].astype(int), 1]
        + NODE[EDGE[:, 2].astype(int), 1]
    ) / 3

    edge_x = (NODE[edges[:, 0] - 1, 0] + NODE[edges[:, 1] - 1, 0]) / 2
    edge_y = (NODE[edges[:, 0] - 1, 1] + NODE[edges[:, 1] - 1, 1]) / 2

    face_node_dict = {}

    for idx, face in enumerate(face_node):
        for node in face:
            if node not in face_node_dict:
                face_node_dict[node] = []
            face_node_dict[node].append(idx)

    for i, edge in enumerate(edges):
        node1, node2 = map(int, edge)

        edge_node[i, 0] = node1
        edge_node[i, 1] = node2

        faces_node1 = face_node_dict.get(node1 - 1, [])
        faces_node2 = face_node_dict.get(node2 - 1, [])

        faces = list(set(faces_node1) & set(faces_node2))

        if len(faces) < 2:
            edge_face[i, 0] = faces[0] + 1 if faces else 0
            edge_face[i, 1] = 0
        else:
            edge_face[i, 0] = faces[0] + 1
            edge_face[i, 1] = faces[1] + 1

    face_x = np.array(face_x, dtype=np.double)
    face_y = np.array(face_y, dtype=np.double)

    node_x = np.array(NODE[:, 0], dtype=np.double)
    node_y = np.array(NODE[:, 1], dtype=np.double)
    node_z = np.array(NODE[:, 2], dtype=np.double)

    face_x_bnd = np.array(node_x[face_node], dtype=np.double)
    face_y_bnd = np.array(node_y[face_node], dtype=np.double)

    num_nodes = NODE.shape[0]
    num_faces = EDGE.shape[0]
    num_edges = edges.shape[0]

    with Dataset(netcdf_path, "w", format="NETCDF4") as dataset:
        _mesh2d_nNodes = dataset.createDimension("mesh2d_nNodes", num_nodes)
        _mesh2d_nEdges = dataset.createDimension("mesh2d_nEdges", num_edges)
        _mesh2d_nFaces = dataset.createDimension("mesh2d_nFaces", num_faces)
        _mesh2d_nMax_face_nodes = dataset.createDimension("mesh2d_nMax_face_nodes", 3)
        _two_dim = dataset.createDimension("Two", 2)

        mesh2d_node_x = dataset.createVariable(
            "mesh2d_node_x", "f8", ("mesh2d_nNodes",)
        )
        mesh2d_node_x.standard_name = "projection_x_coordinate"
        mesh2d_node_x.long_name = "x-coordinate of mesh nodes"

        mesh2d_node_y = dataset.createVariable(
            "mesh2d_node_y", "f8", ("mesh2d_nNodes",)
        )
        mesh2d_node_y.standard_name = "projection_y_coordinate"
        mesh2d_node_y.long_name = "y-coordinate of mesh nodes"

        mesh2d_node_z = dataset.createVariable(
            "mesh2d_node_z", "f8", ("mesh2d_nNodes",)
        )
        mesh2d_node_z.units = "m"
        mesh2d_node_z.standard_name = "altitude"
        mesh2d_node_z.long_name = "z-coordinate of mesh nodes"

        mesh2d_edge_x = dataset.createVariable(
            "mesh2d_edge_x", "f8", ("mesh2d_nEdges",)
        )
        mesh2d_edge_x.standard_name = "projection_x_coordinate"
        mesh2d_edge_x.long_name = (
            "Characteristic x-coordinate of the mesh edge (e.g., midpoint)"
        )

        mesh2d_edge_y = dataset.createVariable(
            "mesh2d_edge_y", "f8", ("mesh2d_nEdges",)
        )
        mesh2d_edge_y.standard_name = "projection_y_coordinate"
        mesh2d_edge_y.long_name = (
            "Characteristic y-coordinate of the mesh edge (e.g., midpoint)"
        )

        mesh2d_edge_nodes = dataset.createVariable(
            "mesh2d_edge_nodes", "i4", ("mesh2d_nEdges", "Two")
        )
        mesh2d_edge_nodes.cf_role = "edge_node_connectivity"
        mesh2d_edge_nodes.long_name = "Start and end nodes of mesh edges"
        mesh2d_edge_nodes.start_index = 1

        mesh2d_edge_faces = dataset.createVariable(
            "mesh2d_edge_faces", "f8", ("mesh2d_nEdges", "Two")
        )
        mesh2d_edge_faces.cf_role = "edge_face_connectivity"
        mesh2d_edge_faces.long_name = "Start and end nodes of mesh edges"
        mesh2d_edge_faces.start_index = 1

        mesh2d_face_nodes = dataset.createVariable(
            "mesh2d_face_nodes", "i4", ("mesh2d_nFaces", "mesh2d_nMax_face_nodes")
        )
        mesh2d_face_nodes.long_name = "Vertex node of mesh face (counterclockwise)"
        mesh2d_face_nodes.start_index = 1

        mesh2d_face_x = dataset.createVariable(
            "mesh2d_face_x", "f8", ("mesh2d_nFaces",)
        )
        mesh2d_face_x.standard_name = "projection_x_coordinate"
        mesh2d_face_x.long_name = "characteristic x-coordinate of the mesh face"
        mesh2d_face_x.start_index = 1

        mesh2d_face_y = dataset.createVariable(
            "mesh2d_face_y", "f8", ("mesh2d_nFaces",)
        )
        mesh2d_face_y.standard_name = "projection_y_coordinate"
        mesh2d_face_y.long_name = "characteristic y-coordinate of the mesh face"
        mesh2d_face_y.start_index = 1

        mesh2d_face_x_bnd = dataset.createVariable(
            "mesh2d_face_x_bnd", "f8", ("mesh2d_nFaces", "mesh2d_nMax_face_nodes")
        )
        mesh2d_face_x_bnd.long_name = (
            "x-coordinate bounds of mesh faces (i.e. corner coordinates)"
        )

        mesh2d_face_y_bnd = dataset.createVariable(
            "mesh2d_face_y_bnd", "f8", ("mesh2d_nFaces", "mesh2d_nMax_face_nodes")
        )
        mesh2d_face_y_bnd.long_name = (
            "y-coordinate bounds of mesh faces (i.e. corner coordinates)"
        )

        mesh2d_node_x.units = "longitude"
        mesh2d_node_y.units = "latitude"
        mesh2d_edge_x.units = "longitude"
        mesh2d_edge_y.units = "latitude"
        mesh2d_face_x.units = "longitude"
        mesh2d_face_y.units = "latitude"
        mesh2d_face_x_bnd.units = "grados"
        mesh2d_face_y_bnd.units = "grados"
        mesh2d_face_x_bnd.standard_name = "longitude"
        mesh2d_face_y_bnd.standard_name = "latitude"
        mesh2d_face_nodes.coordinates = "mesh2d_node_x mesh2d_node_y"

        wgs84 = dataset.createVariable("wgs84", "int32")
        wgs84.setncatts(
            {
                "name": "WGS 84",
                "epsg": np.int32(4326),
                "grid_mapping_name": "latitude_longitude",
                "longitude_of_prime_meridian": 0.0,
                "semi_major_axis": 6378137.0,
                "semi_minor_axis": 6356752.314245,
                "inverse_flattening": 298.257223563,
                "EPSG_code": "value is equal to EPSG code",
                "proj4_params": "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
                "projection_name": "unknown",
                "wkt": 'GEOGCS["WGS 84",\n    DATUM["WGS_1984",\n        SPHEROID["WGS 84",6378137,298.257223563,\n            AUTHORITY["EPSG","7030"]],\n        AUTHORITY["EPSG","6326"]],\n    PRIMEM["Greenwich",0,\n        AUTHORITY["EPSG","8901"]],\n    UNIT["degree",0.0174532925199433,\n        AUTHORITY["EPSG","9122"]],\n    AXIS["Latitude",NORTH],\n    AXIS["Longitude",EAST],\n    AUTHORITY["EPSG","4326"]]',
            }
        )

        mesh2d_node_x[:] = node_x
        mesh2d_node_y[:] = node_y
        mesh2d_node_z[:] = -node_z

        mesh2d_edge_x[:] = edge_x
        mesh2d_edge_y[:] = edge_y
        mesh2d_edge_nodes[:, :] = edge_node

        mesh2d_edge_faces[:] = edge_face
        mesh2d_face_nodes[:] = face_node + 1
        mesh2d_face_x[:] = face_x
        mesh2d_face_y[:] = face_y

        mesh2d_face_x_bnd[:] = face_x_bnd
        mesh2d_face_y_bnd[:] = face_y_bnd

        dataset.institution = "GeoOcean"
        dataset.references = "https://github.com/GeoOcean/BlueMath_tk"
        dataset.source = f"BlueMath tk {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        dataset.history = "Created with OCSmesh"
        dataset.Conventions = "CF-1.8 UGRID-1.0 Deltares-0.10"

        dataset.createDimension("str_dim", 1)
        mesh2d = dataset.createVariable("mesh2d", "i4", ("str_dim",))
        mesh2d.cf_role = "mesh_topology"
        mesh2d.long_name = "Topology data of 2D mesh"
        mesh2d.topology_dimension = 2
        mesh2d.node_coordinates = "mesh2d_node_x mesh2d_node_y"
        mesh2d.node_dimension = "mesh2d_nNodes"
        mesh2d.edge_node_connectivity = "mesh2d_edge_nodes"
        mesh2d.edge_dimension = "mesh2d_nEdges"
        mesh2d.edge_coordinates = "mesh2d_edge_x mesh2d_edge_y"
        mesh2d.face_node_connectivity = "mesh2d_face_nodes"
        mesh2d.face_dimension = "mesh2d_nFaces"
        mesh2d.face_coordinates = "mesh2d_face_x mesh2d_face_y"
        mesh2d.max_face_nodes_dimension = "mesh2d_nMax_face_nodes"
        mesh2d.edge_face_connectivity = "mesh2d_edge_faces"


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
    SWATH: bool = False,
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

    simple_quiver = 5
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

    ax1.set_title("Vortex wind")
    ax2.set_title("Wind partition (GreenSurge)")

    triangle_forcing_connectivity = ds_GFD_info.triangle_forcing_connectivity.values
    node_forcing_longitude = ds_GFD_info.node_forcing_longitude.values
    node_forcing_latitude = ds_GFD_info.node_forcing_latitude.values

    Lon = xds_vortex_GS.lon
    Lat = xds_vortex_GS.lat
    if not SWATH:
        W = xds_vortex_interp.W.isel(time=i_time)
        Dir = (270 - xds_vortex_interp.Dir.isel(time=i_time)) % 360
        W_reg = xds_vortex_GS.W.isel(time=i_time)
        Dir_reg = (270 - xds_vortex_GS.Dir.isel(time=i_time)) % 360
        fig.suptitle(
            f"Wind partition for {time.values.astype('datetime64[s]').astype(str)}",
        )
    else:
        W = xds_vortex_interp.W.max(dim="time")
        W_reg = xds_vortex_GS.W.max(dim="time")
        fig.suptitle(
            "Wind partition SWATH",
        )
    vmin = np.min((W.min(), W_reg.min()))
    vmax = np.max((W.max(), W_reg.max()))

    cmap = join_colormaps(
        cmap1="viridis",
        cmap2="plasma_r",
        name="wind_partition_cmap",
        range1=(0.2, 1.0),
        range2=(0.05, 0.8),
    )

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
    if not SWATH:
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
    info_ds: xr.Dataset, figsize: tuple = (10, 10), fig: Figure = None, ax: Axes = None
) -> Tuple[Figure, Axes]:
    """
    Plot the GreenSurge mesh setup from the provided dataset.

    Parameters
    ----------
    info_ds : xr.Dataset
        Dataset containing the mesh information.
    figsize : tuple, optional
        Figure size. Default is (10, 10).
    fig : Figure, optional
        Figure object to plot on. If None, a new figure will be created.
    ax : Axes, optional
        Axes object to plot on. If None, a new axes will be created.

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
    if fig is None or ax is None:
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
        _x = (
            node_forcing_longitude[int(node0)]
            + node_forcing_longitude[int(node1)]
            + node_forcing_longitude[int(node2)]
        ) / 3
        _y = (
            node_forcing_latitude[int(node0)]
            + node_forcing_latitude[int(node1)]
            + node_forcing_latitude[int(node2)]
        ) / 3
        plt.text(
            _x, _y, f"T{t}", fontsize=10, ha="center", va="center", fontweight="bold"
        )

    bnd = [
        min(node_computation_longitude.min(), node_forcing_longitude.min()),
        max(node_computation_longitude.max(), node_forcing_longitude.max()),
        min(node_computation_latitude.min(), node_forcing_latitude.min()),
        max(node_computation_latitude.max(), node_forcing_latitude.max()),
    ]
    ax.set_extent([*bnd], crs=ccrs.PlateCarree())
    plt.legend(loc="lower left", fontsize=10, markerscale=2.0)
    ax.set_title("GreenSurge Mesh Setup")
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    return fig, ax


def create_triangle_mask(
    lon_grid: np.ndarray, lat_grid: np.ndarray, triangle: np.ndarray
) -> np.ndarray:
    """
    Create a mask for a triangle defined by its vertices.

    Parameters
    ----------
    lon_grid : np.ndarray
        The longitude grid.
    lat_grid : np.ndarray
        The latitude grid.
    triangle : np.ndarray
        The triangle vertices.

    Returns
    -------
    np.ndarray
        The mask for the triangle.
    """

    triangle_path = Path(triangle)
    # if lon_grid.ndim == 1:
    #     lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
    points = np.vstack([lon_grid.flatten(), lat_grid.flatten()]).T
    inside_mask = triangle_path.contains_points(points)
    mask = inside_mask.reshape(lon_grid.shape)

    return mask


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
    Returns
    -------
    fig : Figure
        The figure object containing the plots.
    axs : list of Axes
        List of Axes objects for the two subplots.
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
    return fig, axs


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
    base_drag_coeff = GS_LinearWindDragCoef(
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
            drag_coeff_value = GS_LinearWindDragCoef(
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


def GS_LinearWindDragCoef(
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

    Wla = Wl_abc[0]
    Wlb = Wl_abc[1]
    Wlc = Wl_abc[2]
    CDa = CD_Wl_abc[0]
    CDb = CD_Wl_abc[1]
    CDc = CD_Wl_abc[2]

    # coefs lines y=ax+b
    if not Wla == Wlb:
        a_CDline_ab = (CDa - CDb) / (Wla - Wlb)
        b_CDline_ab = CDb - a_CDline_ab * Wlb
    else:
        a_CDline_ab = 0
        b_CDline_ab = CDa
    if not Wlb == Wlc:
        a_CDline_bc = (CDb - CDc) / (Wlb - Wlc)
        b_CDline_bc = CDc - a_CDline_bc * Wlc
    else:
        a_CDline_bc = 0
        b_CDline_bc = CDb
    a_CDline_cinf = 0
    b_CDline_cinf = CDc

    if Wspeed <= Wlb:
        CD = a_CDline_ab * Wspeed + b_CDline_ab
    elif Wspeed > Wlb and Wspeed <= Wlc:
        CD = a_CDline_bc * Wspeed + b_CDline_bc
    else:
        CD = a_CDline_cinf * Wspeed + b_CDline_cinf

    return CD


def plot_GS_vs_dynamic_windsetup(
    ds_WL_GS_WindSetUp: xr.Dataset,
    ds_WL_dynamic_WindSetUp: xr.Dataset,
    ds_gfd_metadata: xr.Dataset,
    time: datetime,
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
    figsize: tuple = (20, 7),
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

    nface_index = extract_pos_nearest_points_tri(ds_GFD_info, lon_obs, lat_obs)
    mesh2d_nFaces = extract_pos_nearest_points_tri(
        ds_WL_dynamic_WindSetUp, lon_obs, lat_obs
    )
    pos_lon_IB, pos_lat_IB = extract_pos_nearest_points(ds_WL_GS_IB, lon_obs, lat_obs)

    time = ds_WL_GS_WindSetUp.WL.time
    ds_WL_dynamic_WindSetUp = ds_WL_dynamic_WindSetUp.sel(time=time)
    ds_WL_GS_IB = ds_WL_GS_IB.interp(time=time)

    WL_GS = ds_WL_GS_WindSetUp.WL.sel(nface=nface_index).values.squeeze()
    WL_dyn = ds_WL_dynamic_WindSetUp.mesh2d_s1.sel(
        mesh2d_nFaces=mesh2d_nFaces
    ).values.squeeze()
    WL_IB = ds_WL_GS_IB.IB.values[pos_lat_IB, pos_lon_IB, :].squeeze()
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
    vmin = np.nanmin(Z)
    vmax = np.nanmax(Z)

    cmap, norm = join_colormaps(
        cmap1=hex_colors_water,
        cmap2=hex_colors_land,
        value_range1=(vmin, 0.0),
        value_range2=(0.0, vmax),
        name="raster_cmap",
    )
    ax_map.set_facecolor("#518134")

    ax_map.tripcolor(
        X,
        Y,
        triangles,
        facecolors=Z,
        cmap=cmap,
        norm=norm,
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
        WLmax = (
            max(
                np.nanmax(WL_SS_dyn),
                np.nanmax(WL_SS_GS),
                np.nanmax(WL_TG),
                np.nanmax(WL_GS),
            )
            * 1.05
        )
        WLmin = (
            min(
                np.nanmin(WL_SS_dyn),
                np.nanmin(WL_SS_GS),
                np.nanmin(WL_TG),
                np.nanmin(WL_GS),
            )
            * 1.05
        )

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
        # elements = ds_mesh_info.triangle_computation_connectivity.values
        # lon_mesh = np.mean(
        #     ds_mesh_info.node_computation_longitude.values[elements], axis=1
        # )
        # lat_mesh = np.mean(
        #     ds_mesh_info.node_computation_latitude.values[elements], axis=1
        # )

        lon_mesh = ds_mesh_info.node_computation_longitude.values
        lat_mesh = ds_mesh_info.node_computation_latitude.values
        type_ds = 0
    else:
        lon_mesh = ds_mesh_info.mesh2d_face_x.values
        lat_mesh = ds_mesh_info.mesh2d_face_y.values
        type_ds = 1

    nface_index = []  # np.zeros(len(lon_points))

    for i in range(len(lon_points)):
        lon = lon_points[i]
        lat = lat_points[i]

        distances = np.sqrt((lon_mesh - lon) ** 2 + (lat_mesh - lat) ** 2)
        min_idx = np.argmin(distances)

        if type_ds == 0:
            # nface_index[i] = ds_mesh_info.node_cumputation_index.values[min_idx].astype(int)
            nface_index.append(
                ds_mesh_info.node_cumputation_index.values[min_idx].astype(int)
            )
        elif type_ds == 1:
            # nface_index[i] = ds_mesh_info.mesh2d_nFaces.values[min_idx].astype(int)
            nface_index.append(ds_mesh_info.mesh2d_nFaces.values[min_idx].astype(int))

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

    pos_lon_points_mesh = []  # = np.zeros(len(lon_points))
    pos_lat_points_mesh = []  # = np.zeros(len(lat_points))

    for i in range(len(lon_points)):
        lon = lon_points[i]
        lat = lat_points[i]

        lat_index = np.nanargmin((lat - lat_mesh) ** 2)
        lon_index = np.nanargmin((lon - lon_mesh) ** 2)

        # pos_lon_points_mesh[i] = lon_index.astype(int)
        # pos_lat_points_mesh[i] = lat_index.astype(int)
        pos_lon_points_mesh.append(lon_index.astype(int))
        pos_lat_points_mesh.append(lat_index.astype(int))

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
    IB = (101325 - p) / 10000  # Convert pressure (Pa) to inverse barometer (m)

    xds_presure_modified = xds_presure.copy()
    xds_presure_modified["IB"] = (("lat", "lon", "time"), IB)

    return xds_presure_modified


def compute_water_level_for_time(
    time_index: int,
    direction_data: np.ndarray,
    wind_speed_data: np.ndarray,
    direction_bins: np.ndarray,
    forcing_cell_indices: np.ndarray,
    greensurge_dataset: xr.Dataset,
    wind_speed_reference: float,
    base_drag_coeff: float,
    drag_coefficients: np.ndarray,
    velocity_thresholds: np.ndarray,
    duration_in_steps: int,
    num_output_times: int,
) -> np.ndarray:
    """
    Compute the water level for a specific time index based on wind direction and speed.

    Parameters
    ----------
    time_index : int
        The index of the time step to compute the water level for.
    direction_data : np.ndarray
        2D array of wind direction data with shape (n_cells, n_times).
    wind_speed_data : np.ndarray
        2D array of wind speed data with shape (n_cells, n_times).
    direction_bins : np.ndarray
        1D array of wind direction bins.
    forcing_cell_indices : np.ndarray
        1D array of indices for the forcing cells.
    greensurge_dataset : xr.Dataset
        xarray Dataset containing the GreenSurge mesh and forcing data.
    wind_speed_reference : float
        Reference wind speed value for scaling.
    base_drag_coeff : float
        Base drag coefficient value for scaling.
    drag_coefficients : np.ndarray
        1D array of drag coefficients corresponding to the velocity thresholds.
    velocity_thresholds : np.ndarray
        1D array of velocity thresholds for drag coefficient calculation.
    duration_in_steps : int
        Total duration of the simulation in steps.
    num_output_times : int
        Total number of output time steps.

    Returns
    -------
    np.ndarray
        2D array of computed water levels for the specified time index.
    """

    adjusted_bins = np.where(direction_bins == 0, 360, direction_bins)
    n_faces = greensurge_dataset["mesh2d_s1"].isel(forcing_cell=0, direction=0).shape
    water_level_accumulator = np.zeros(n_faces)

    for cell_index in forcing_cell_indices.astype(int):
        current_dir = direction_data[cell_index, time_index] % 360
        closest_direction_index = np.abs(adjusted_bins - current_dir).argmin()

        water_level_case = (
            greensurge_dataset["mesh2d_s1"]
            .sel(forcing_cell=cell_index, direction=closest_direction_index)
            .values
        )
        water_level_case = np.nan_to_num(water_level_case, nan=0)

        wind_speed_value = wind_speed_data[cell_index, time_index]
        drag_coeff_value = GS_LinearWindDragCoef(
            wind_speed_value, drag_coefficients, velocity_thresholds
        )

        scaling_factor = (wind_speed_value**2 / wind_speed_reference**2) * (
            drag_coeff_value / base_drag_coeff
        )
        water_level_accumulator += water_level_case * scaling_factor

    step_window = min(duration_in_steps, num_output_times - time_index)
    result = np.zeros((num_output_times, n_faces[1]))
    if (num_output_times - time_index) > step_window:
        result[time_index : time_index + step_window] += water_level_accumulator
    else:
        shift_counter = step_window - (num_output_times - time_index)
        result[time_index : time_index + step_window - shift_counter] += (
            water_level_accumulator[: step_window - shift_counter]
        )
    return result


def GS_windsetup_reconstruction_with_postprocess_parallel(
    greensurge_dataset: xr.Dataset,
    ds_gfd_metadata: xr.Dataset,
    wind_direction_input: xr.Dataset,
    num_workers: int = None,
    velocity_thresholds: np.ndarray = np.array([0, 100, 100]),
    drag_coefficients: np.ndarray = np.array([0.00063, 0.00723, 0.00723]),
) -> xr.Dataset:
    """
    Reconstructs the GreenSurge wind setup using the provided wind direction input and metadata in parallel.

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

    if num_workers is None:
        num_workers = cpu_count()

    direction_bins = ds_gfd_metadata.wind_directions.values
    forcing_cell_indices = greensurge_dataset.forcing_cell.values
    wind_speed_reference = ds_gfd_metadata.wind_speed.values.item()
    base_drag_coeff = GS_LinearWindDragCoef(
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

    n_faces = greensurge_dataset["mesh2d_s1"].isel(forcing_cell=0, direction=0).shape[1]

    args = partial(
        compute_water_level_for_time,
        direction_data=direction_data,
        wind_speed_data=wind_speed_data,
        direction_bins=direction_bins,
        forcing_cell_indices=forcing_cell_indices,
        greensurge_dataset=greensurge_dataset,
        wind_speed_reference=wind_speed_reference,
        base_drag_coeff=base_drag_coeff,
        drag_coefficients=drag_coefficients,
        velocity_thresholds=velocity_thresholds,
        duration_in_steps=duration_in_steps,
        num_output_times=num_output_times,
    )

    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm(pool.imap(args, range(num_output_times)), total=num_output_times)
        )

    wind_setup_output = np.sum(results, axis=0)

    ds_wind_setup = xr.Dataset(
        {"WL": (["time", "nface"], wind_setup_output)},
        coords={
            "time": output_time_vector,
            "nface": np.arange(n_faces),
        },
    )
    ds_wind_setup.attrs["description"] = "Wind setup from GreenSurge methodology"

    return ds_wind_setup


def build_greensurge_infos_dataset(
    path_grd_calc: str,
    path_grd_forz: str,
    site: str,
    wind_speed: float,
    direction_step: float,
    simulation_duration_hours: float,
    simulation_time_step_hours: float,
    forcing_time_step: float,
    reference_date_dt: datetime,
    Eddy: float,
    Chezy: float,
) -> xr.Dataset:
    """
    Build a structured dataset containing simulation parameters for hybrid modeling.

    Parameters
    ----------
    path_grd_calc : str
        Path to the computational grid file.
    path_grd_forz : str
        Path to the forcing grid file.
    site : str
        Name of the case study location.
    wind_speed : float
        Wind speed for each discretized direction.
    direction_step : float
        Step size for wind direction discretization in degrees.
    simulation_duration_hours : float
        Total duration of the simulation in hours.
    simulation_time_step_hours : float
        Time step used in the simulation in hours.
    forcing_time_step : float
        Time step used for applying external forcing data in hours.
    reference_date_dt : datetime
        Reference start date of the simulation.
    Eddy : float
        Eddy viscosity used in the simulation in m²/s.
    Chezy : float
        Chezy coefficient used for bottom friction.
    Returns
    -------
    xr.Dataset
        A structured dataset containing simulation parameters for hybrid modeling.
    """

    Nodes_calc, Elmts_calc, lines_calc = read_adcirc_grd(path_grd_calc)
    Nodes_forz, Elmts_forz, lines_forz = read_adcirc_grd(path_grd_forz)

    num_elements = Elmts_forz.shape[0]

    triangle_node_indices = np.arange(3)

    num_directions = int(360 / direction_step)
    wind_directions = np.arange(0, 360, direction_step)
    wind_direction_indices = np.arange(0, num_directions)

    element_forcing_indices = np.arange(0, num_elements)
    element_computation_indices = np.arange(0, len(Elmts_calc[:, 1]))

    node_forcing_indices = np.arange(0, len(Nodes_forz[:, 1]))

    time_forcing_index = [
        0,
        forcing_time_step,
        forcing_time_step + 0.001,
        simulation_duration_hours - 1,
    ]

    node_cumputation_index = np.arange(0, len(Nodes_calc[:, 1]))

    reference_date_str = reference_date_dt.strftime("%Y-%m-%d %H:%M:%S")

    simulation_dataset = xr.Dataset(
        coords=dict(
            wind_direction_index=("wind_direction_index", wind_direction_indices),
            time_forcing_index=("time_forcing_index", time_forcing_index),
            node_computation_longitude=("node_cumputation_index", Nodes_calc[:, 1]),
            node_computation_latitude=("node_cumputation_index", Nodes_calc[:, 2]),
            triangle_nodes=("triangle_forcing_nodes", triangle_node_indices),
            node_forcing_index=("node_forcing_index", node_forcing_indices),
            element_forcing_index=("element_forcing_index", element_forcing_indices),
            node_cumputation_index=("node_cumputation_index", node_cumputation_index),
            element_computation_index=(
                "element_computation_index",
                element_computation_indices,
            ),
        ),
        data_vars=dict(
            triangle_computation_connectivity=(
                ("element_computation_index", "triangle_forcing_nodes"),
                Elmts_calc[:, 2:5].astype(int),
                {
                    "description": "Indices of nodes forming each triangular element of the computational grid (counter-clockwise order)"
                },
            ),
            node_forcing_longitude=(
                "node_forcing_index",
                Nodes_forz[:, 1],
                {
                    "units": "degrees_east",
                    "description": "Longitude of each mesh node of the forcing grid",
                },
            ),
            node_forcing_latitude=(
                "node_forcing_index",
                Nodes_forz[:, 2],
                {
                    "units": "degrees_north",
                    "description": "Latitude of each mesh node of the forcing grid",
                },
            ),
            triangle_forcing_connectivity=(
                ("element_forcing_index", "triangle_forcing_nodes"),
                Elmts_forz[:, 2:5].astype(int),
                {
                    "description": "Indices of nodes forming each triangular element of the forcing grid (counter-clockwise order)"
                },
            ),
            wind_directions=(
                "wind_direction_index",
                wind_directions,
                {
                    "units": "degrees",
                    "description": "Discretized wind directions (0 to 360°)",
                },
            ),
            total_elements=(
                (),
                num_elements,
                {"description": "Total number of triangular elements in the mesh"},
            ),
            simulation_duration_hours=(
                (),
                simulation_duration_hours,
                {"units": "hours", "description": "Total duration of the simulation"},
            ),
            time_step_hours=(
                (),
                simulation_time_step_hours,
                {"units": "hours", "description": "Time step used in the simulation"},
            ),
            wind_speed=(
                (),
                wind_speed,
                {
                    "units": "m/s",
                    "description": "Wind speed for each discretized direction",
                },
            ),
            location_name=((), site, {"description": "Name of case study location"}),
            eddy_viscosity=(
                (),
                Eddy,
                {
                    "units": "m²/s",
                    "description": "Eddy viscosity used in the simulation",
                },
            ),
            chezy_coefficient=(
                (),
                Chezy,
                {"description": "Chezy coefficient used for bottom friction"},
            ),
            reference_date=(
                (),
                reference_date_str,
                {"description": "Reference start date of the simulation"},
            ),
            forcing_time_step=(
                (),
                forcing_time_step,
                {
                    "units": "hour",
                    "description": "Time step used for applying external forcing data",
                },
            ),
        ),
    )

    simulation_dataset["time_forcing_index"].attrs = {
        "standard_name": "time",
        "long_name": f"Time - hours since {reference_date_str} +00:00",
        "time_origin": f"{reference_date_str}",
        "units": f"hours since {reference_date_str} +00:00",
        "calendar": "gregorian",
        "description": "Time definition for the forcing data",
    }

    simulation_dataset["node_computation_longitude"].attrs = {
        "description": "Longitude of each mesh node of the computational grid",
        "standard_name": "longitude",
        "long_name": "longitude",
        "units": "degrees_east",
    }
    simulation_dataset["node_computation_latitude"].attrs = {
        "description": "Latitude of each mesh node of the computational grid",
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
    }

    simulation_dataset.attrs = {
        "title": "Hybrid Simulation Input Dataset",
        "description": "Structured dataset containing simulation parameters for hybrid modeling.",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "institution": "GeoOcean",
        "model": "GreenSurge",
    }
    return simulation_dataset


def plot_greensurge_setup_with_raster(
    simulation_dataset,
    path_grd_calc,
    figsize=(7, 7),
) -> None:
    """
    Plot the GreenSurge setup with raster bathymetry.

    Parameters
    ----------
    simulation_dataset : xr.Dataset
        Dataset containing simulation information.
    path_grd_calc : str
        Path to the ADCIRC grid file for calculation.
    figsize : tuple, optional
        Size of the figure, by default (7, 7)
    Returns
    -------
    None
    -----------
    This function plots the GreenSurge setup using raster bathymetry
    and the ADCIRC grid for calculation. It uses Cartopy for geographic
    projections and matplotlib for plotting.
    """

    Nodes_calc, Elmts_calc, lines_calc = read_adcirc_grd(path_grd_calc)

    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=figsize,
        constrained_layout=True,
    )

    # ax.set_facecolor("#518134")
    Longitude_nodes_calc = Nodes_calc[:, 1]
    Latitude_nodes_calc = Nodes_calc[:, 2]
    Elements_calc = Elmts_calc[:, 2:5].astype(int)
    depth = -np.mean(Nodes_calc[Elements_calc, 3], axis=1)

    vmin = np.nanmin(depth)
    vmax = np.nanmax(depth)

    cmap, norm = join_colormaps(
        cmap1=hex_colors_water,
        cmap2=hex_colors_land,
        value_range1=(vmin, 0.0),
        value_range2=(0.0, vmax),
        name="raster_cmap",
    )

    tpc = ax.tripcolor(
        Longitude_nodes_calc,
        Latitude_nodes_calc,
        Elements_calc,
        facecolors=depth,
        cmap=cmap,
        norm=norm,
        shading="flat",
        transform=ccrs.PlateCarree(),
    )
    cbar = plt.colorbar(tpc, ax=ax)
    cbar.set_label("Depth (m)")

    plot_greensurge_setup(simulation_dataset, figsize=(7, 7), ax=ax, fig=fig)


def plot_triangle_points(
    lon_all: np.ndarray,
    lat_all: np.ndarray,
    i: int,
    ds_GFD_info: xr.Dataset,
    figsize: tuple = (7, 7),
) -> None:
    """
    Plot a triangle and points selection for GreenSurge.
    Parameters
    ----------
    lon_all : array-like
        Longitudes of the points.
    lat_all : array-like
        Latitudes of the points.
    i : int
        Index of the triangle to plot.
    ds_GFD_info : xarray.Dataset
        Dataset containing GreenSurge information.
    figsize : tuple, optional
        Size of the figure, by default (7, 7).
    """

    lon_points = lon_all[i]
    lat_points = lat_all[i]
    triangle = np.array(
        [
            [lon_points[0], lat_points[0]],
            [lon_points[1], lat_points[1]],
            [lon_points[2], lat_points[2]],
            [lon_points[0], lat_points[0]],
        ]
    )

    fig, ax = plot_greensurge_setup(ds_GFD_info, figsize=figsize)
    ax.fill(
        triangle[:, 0],
        triangle[:, 1],
        color="green",
        alpha=0.5,
        transform=ccrs.PlateCarree(),
    )
    ax.scatter(
        lon_points,
        lat_points,
        color="red",
        marker="o",
        transform=ccrs.PlateCarree(),
        label="Points selection",
    )
    ax.set_title("Exemple of point selection for GreenSurge")
    ax.legend()
    fig.show()


def interp_vortex_to_triangles(
    xds_vortex_GS: xr.Dataset,
    lon_all: np.ndarray,
    lat_all: np.ndarray,
    type: str = "tri_mean",
) -> xr.Dataset:
    """
    Interpolates the vortex model data to the triangle points.
    Parameters
    ----------
    xds_vortex_GS : xr.Dataset
        Dataset containing the vortex model data.
    lon_all : np.ndarray
        Longitudes of the triangle points.
    lat_all : np.ndarray
        Latitudes of the triangle points.
    Returns
    -------
    xds_vortex_interp : xr.Dataset
        Dataset containing the interpolated vortex model data at the triangle points.
    -----------
    This function interpolates the vortex model data (wind speed, direction, and pressure)
    to the triangle points defined by `lon_all` and `lat_all`. It reshapes the data
    to match the number of triangles and points, and computes the mean values for each triangle.
    """

    if type == "tri_mean":
        n_tri, n_pts = lat_all.shape
        lat_interp = lat_all.reshape(-1)
        lon_interp = lon_all.reshape(-1)
    elif type == "tri_points":
        n_tri = lat_all.shape
        lat_interp = lat_all
        lon_interp = lon_all

    lat_interp = xr.DataArray(lat_interp, dims="point")
    lon_interp = xr.DataArray(lon_interp, dims="point")

    if type == "tri_mean":
        W_interp = xds_vortex_GS.W.interp(lat=lat_interp, lon=lon_interp)
        Dir_interp = xds_vortex_GS.Dir.interp(lat=lat_interp, lon=lon_interp)
        p_interp = xds_vortex_GS.p.interp(lat=lat_interp, lon=lon_interp)

        W_interp = W_interp.values.reshape(n_tri, n_pts, -1)
        Dir_interp = Dir_interp.values.reshape(n_tri, n_pts, -1)
        p_interp = p_interp.values.reshape(n_tri, n_pts, -1)

        theta_rad = np.deg2rad(Dir_interp)
        u = np.cos(theta_rad)
        v = np.sin(theta_rad)
        u_mean = u.mean(axis=1)
        v_mean = v.mean(axis=1)
        Dir_out = (np.rad2deg(np.arctan2(v_mean, u_mean))) % 360
        W_out = W_interp.mean(axis=1)
        p_out = p_interp.mean(axis=1)
    elif type == "tri_points":
        xds_vortex_interp = xds_vortex_GS.interp(lat=lat_interp, lon=lon_interp)
        return xds_vortex_interp

    xds_vortex_interp = xr.Dataset(
        data_vars={
            "W": (("element", "time"), W_out),
            "Dir": (("element", "time"), Dir_out),
            "p": (("element", "time"), p_out),
        },
        coords={"time": xds_vortex_GS.time.values, "element": np.arange(n_tri)},
    )

    return xds_vortex_interp


def load_GS_database(
    xds_vortex_interp: xr.Dataset, ds_GFD_info: xr.Dataset, p_GFD_libdir: str
) -> xr.Dataset:
    """
    Load the Green Surge database based on the interpolated vortex data.
    Parameters
    ----------
    xds_vortex_interp : xarray.Dataset
        Interpolated vortex data on the structured grid.
    ds_GFD_info : xarray.Dataset
        Dataset containing information about the Green Surge database.
    p_GFD_libdir : str
        Path to the Green Surge database directory.
    Returns
    -------
    xarray.Dataset
        Dataset containing the Green Surge data for the specified wind directions.
    """

    wind_direction_interp = xds_vortex_interp.Dir

    wind_direction_database = ds_GFD_info.wind_directions.values
    wind_direction_step = np.mean(np.diff(wind_direction_database))
    wind_direction_indices = (
        (np.round((wind_direction_interp.values % 360) / wind_direction_step))
        % len(wind_direction_database)
    ).astype(int)
    unique_direction_indices = np.unique(wind_direction_indices).astype(str)

    green_surge_file_paths = np.char.add(
        np.char.add(p_GFD_libdir + "/GreenSurge_DB_", unique_direction_indices), ".nc"
    )

    def preprocess(dataset):
        file_name = dataset.encoding.get("source", "Unknown")
        direction_string = file_name.split("_DB_")[-1].split(".")[0]
        direction_index = int(direction_string)
        return (
            dataset[["mesh2d_s1"]]
            .expand_dims("direction")
            .assign_coords(direction=[direction_index])
        )

    greensurge_dataset = xr.open_mfdataset(
        green_surge_file_paths,
        parallel=False,
        combine="by_coords",
        preprocess=preprocess,
        engine="netcdf4",
    )

    return greensurge_dataset


def plot_GS_validation_timeseries(
    ds_WL_GS_WindSetUp: xr.Dataset,
    ds_WL_GS_IB: xr.Dataset,
    ds_WL_dynamic_WindSetUp: xr.Dataset,
    ds_GFD_info: xr.Dataset,
    lon_obs: float = [184.8],
    lat_obs: float = [-21.14],
    figsize: tuple = (20, 7),
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
    ds_GFD_info : xr.Dataset
        Dataset containing grid information with longitude and latitude coordinates.
    figsize : tuple, optional
        Size of the figure for the plot. Default is (15, 7).
    WLmin : float, optional
        Minimum water level for the plot. Default is None.
    WLmax : float, optional
        Maximum water level for the plot. Default is None.
    """

    lon_obs = [lon - 360 if lon > 180 else lon for lon in lon_obs]

    nface_index = extract_pos_nearest_points_tri(ds_GFD_info, lon_obs, lat_obs)
    mesh2d_nFaces = extract_pos_nearest_points_tri(
        ds_WL_dynamic_WindSetUp, lon_obs, lat_obs
    )
    pos_lon_IB, pos_lat_IB = extract_pos_nearest_points(ds_WL_GS_IB, lon_obs, lat_obs)

    time = ds_WL_GS_WindSetUp.WL.time
    ds_WL_dynamic_WindSetUp = ds_WL_dynamic_WindSetUp.sel(time=time)
    ds_WL_GS_IB = ds_WL_GS_IB.interp(time=time)

    WL_GS = ds_WL_GS_WindSetUp.WL.sel(nface=nface_index).values
    WL_dyn = ds_WL_dynamic_WindSetUp.mesh2d_s1.sel(mesh2d_nFaces=mesh2d_nFaces).values
    WL_IB = ds_WL_GS_IB.IB.values[pos_lat_IB, pos_lon_IB, :].T

    WL_SS_dyn = WL_dyn + WL_IB
    WL_SS_GS = WL_GS + WL_IB

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3], figure=fig)

    ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    X = ds_WL_dynamic_WindSetUp.mesh2d_node_x.values
    Y = ds_WL_dynamic_WindSetUp.mesh2d_node_y.values
    triangles = ds_WL_dynamic_WindSetUp.mesh2d_face_nodes.values.astype(int) - 1
    Z = np.mean(ds_WL_dynamic_WindSetUp.mesh2d_node_z.values[triangles], axis=1)
    vmin = np.nanmin(Z)
    vmax = np.nanmax(Z)

    cmap, norm = join_colormaps(
        cmap1=hex_colors_water,
        cmap2=hex_colors_land,
        value_range1=(vmin, 0.0),
        value_range2=(0.0, vmax),
        name="raster_cmap",
    )
    ax_map.set_facecolor("#518134")

    ax_map.tripcolor(
        X,
        Y,
        triangles,
        facecolors=Z,
        cmap=cmap,
        norm=norm,
        shading="flat",
        transform=ccrs.PlateCarree(),
    )

    ax_map.scatter(
        lon_obs,
        lat_obs,
        color="red",
        marker="x",
        transform=ccrs.PlateCarree(),
        label="Observation Point",
    )

    ax_map.set_extent([X.min(), X.max(), Y.min(), Y.max()], crs=ccrs.PlateCarree())
    ax_map.set_title("Bathymetry Map")
    ax_map.legend(loc="upper left", fontsize="small")
    time_vals = time.values
    n_series = len(lon_obs)
    ax_ts = gridspec.GridSpecFromSubplotSpec(
        n_series, 1, subplot_spec=gs[0, 1], hspace=0.3
    )
    if WLmin is None or WLmax is None:
        typee = 1
    else:
        typee = 0

    axes_right = []
    for i in range(n_series):
        ax_map.text(
            lon_obs[i],
            lat_obs[i],
            f"Point {i + 1}",
            color="k",
            fontsize=10,
            transform=ccrs.PlateCarree(),
            ha="left",
            va="bottom",
        )
        ax = fig.add_subplot(ax_ts[i, 0])
        ax.plot(
            time_vals,
            WL_SS_dyn[:, i],
            c="blue",
            label=f"Dynamic simulation Point {i + 1}",
        )
        ax.plot(
            time_vals, WL_SS_GS[:, i], c="tomato", label=f"GreenSurge Point {i + 1}"
        )
        ax.set_ylabel("Water level (m)")
        ax.legend()
        if i != n_series - 1:
            ax.set_xticklabels([])
        if typee == 1:
            WLmax = (
                max(
                    np.nanmax(WL_SS_dyn[:, i]),
                    np.nanmax(WL_SS_GS[:, i]),
                    np.nanmax(WL_GS[:, i]),
                )
                * 1.05
            )
            WLmin = (
                min(
                    np.nanmin(WL_SS_dyn[:, i]),
                    np.nanmin(WL_SS_GS[:, i]),
                    np.nanmin(WL_GS[:, i]),
                )
                * 1.05
            )
        ax.set_ylim(WLmin, WLmax)
        ax.set_xlim(time_vals[0], time_vals[-1])
        axes_right.append(ax)
    axes_right[0].set_title("Tide Gauge Validation")

    plt.tight_layout()
    plt.show()


def get_regular_grid(
    node_computation_longitude: np.ndarray,
    node_computation_latitude: np.ndarray,
    node_computation_elements: np.ndarray,
    factor: float = 10,
) -> tuple:
    """
    Generate a regular grid based on the node computation longitude and latitude.
    The grid is defined by the minimum and maximum longitude and latitude values,
    and the minimum distance between nodes in both dimensions.
    The grid is generated with a specified factor to adjust the resolution.
    Parameters:
    - node_computation_longitude: 1D array of longitudes for the nodes.
    - node_computation_latitude: 1D array of latitudes for the nodes.
    - node_computation_elements: 2D array of indices defining the elements (triangles).
    - factor: A scaling factor to adjust the resolution of the grid.
    Returns:
    - lon_grid: 1D array of longitudes defining the grid.
    - lat_grid: 1D array of latitudes defining the grid.
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


def GS_wind_partition_tri(ds_GFD_info, xds_vortex):
    element_forcing_index = ds_GFD_info.element_forcing_index.values
    num_element = len(element_forcing_index)
    triangle_forcing_connectivity = ds_GFD_info.triangle_forcing_connectivity.values
    node_forcing_longitude = ds_GFD_info.node_forcing_longitude.values
    node_forcing_latitude = ds_GFD_info.node_forcing_latitude.values
    longitude_forcing_cells = node_forcing_longitude[triangle_forcing_connectivity]
    latitude_forcing_cells = node_forcing_latitude[triangle_forcing_connectivity]

    # if np.abs(np.mean(lon_grid)-np.mean(lon_teselas))>180:
    #     lon_teselas = lon_teselas+360

    # TC_info
    time = xds_vortex.time.values
    lon_grid = xds_vortex.lon.values
    lat_grid = xds_vortex.lat.values
    Ntime = len(time)

    # storage
    U_tes = np.zeros((num_element, Ntime))
    V_tes = np.zeros((num_element, Ntime))
    p_tes = np.zeros((num_element, Ntime))
    Dir_tes = np.zeros((num_element, Ntime))
    Wspeed_tes = np.zeros((num_element, Ntime))

    for i in range(Ntime):
        W_grid = xds_vortex.W.values[:, :, i]
        p_grid = xds_vortex.p.values[:, :, i]
        Dir_grid = (270 - xds_vortex.Dir.values[:, :, i]) * np.pi / 180

        u_sel_t = W_grid * np.cos(Dir_grid)
        v_sel_t = W_grid * np.sin(Dir_grid)

        for element in element_forcing_index:
            X0, X1, X2 = longitude_forcing_cells[element, :]
            Y0, Y1, Y2 = latitude_forcing_cells[element, :]

            triangle = [(X0, Y0), (X1, Y1), (X2, Y2)]

            mask = create_triangle_mask(lon_grid, lat_grid, triangle)

            u_sel = u_sel_t[mask]
            v_sel = v_sel_t[mask]
            p_sel = p_grid[mask]

            p_mean = np.nanmean(p_sel)
            u_mean = np.nanmean(u_sel)
            v_mean = np.nanmean(v_sel)

            U_tes[element, i] = u_mean
            V_tes[element, i] = v_mean
            p_tes[element, i] = p_mean

            Dir_tes[element, i] = get_degrees_from_uv(-u_mean, -v_mean)
            Wspeed_tes[element, i] = np.sqrt(u_mean**2 + v_mean**2)

    xds_vortex_interp = xr.Dataset(
        data_vars={
            "Dir": (("element", "time"), Dir_tes),
            "W": (("element", "time"), Wspeed_tes),
            "p": (("element", "time"), p_tes),
        },
        coords={
            "element": element_forcing_index,
            "time": time,
        },
    )
    return xds_vortex_interp
