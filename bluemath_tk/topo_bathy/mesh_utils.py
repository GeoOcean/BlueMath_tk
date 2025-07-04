import re
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Tuple

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import ocsmesh
import pandas as pd
import rasterio
from jigsawpy.msh_t import jigsaw_msh_t
from matplotlib.axes import Axes
from netCDF4 import Dataset
from pyproj.enums import TransformDirection
from rasterio.mask import mask
from shapely.geometry import LineString, MultiLineString, Polygon, mapping
from shapely.ops import transform

from ..core.geo import buffer_area_for_polygon
from ..core.plotting.colors import hex_colors_land, hex_colors_water
from ..core.plotting.utils import join_colormaps


def plot_mesh_edge(
    msh_t: jigsaw_msh_t, ax: Axes = None, to_geo: callable = None, **kwargs
) -> None:
    """
    Plots the edges of a triangular mesh on a given set of axes.

    Parameters
    ----------
    msh_t : jigsaw_msh_t
        An object containing mesh data. It must have:
        - 'vert2['coord']' containing the coordinates of the mesh vertices
        - 'tria3['index']' containing the indices of the triangles
    ax : Axes, optional
        The axes to plot on. If None, a new plot is created. Default is None.
    to_geo : callable, optional
        A function to transform (x, y) coordinates from projected to geographic CRS.
    **kwargs : keyword arguments, optional
        Additional keyword arguments passed to the `triplot` function.
        These can be used to customize the plot (e.g., color, line style).
    """

    crd = np.array(msh_t.vert2["coord"], copy=True)
    cnn = msh_t.tria3["index"]

    if to_geo is not None:
        crd[:, 0], crd[:, 1] = to_geo(crd[:, 0], crd[:, 1])
        bnd = [crd[:, 0].min(), crd[:, 0].max(), crd[:, 1].min(), crd[:, 1].max()]

    ax.triplot(crd[:, 0], crd[:, 1], cnn, **kwargs)
    ax.set_extent([*bnd], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False


def plot_mesh_vals(
    msh_t: jigsaw_msh_t,
    ax: Axes = None,
    colorbar: bool = True,
    clim: Tuple[float, float] = None,
    to_geo: callable = None,
    **kwargs,
) -> Axes:
    """
    Plots the mesh values on a triangular mesh, with optional transformation
    from UTM to geographic coordinates.

    Parameters
    ----------
    msh_t : jigsaw_msh_t
        An object containing the mesh data. Must include:
        - vert2['coord']: coordinates of mesh vertices (N, 2)
        - tria3['index']: triangle connectivity (M, 3)
        - value: values to plot (length M or Mx1)
    ax : matplotlib Axes, optional
        Axes to draw on. If None, a new one is created.
    colorbar : bool, optional
        If True, show colorbar. Default is True.
    clim : tuple, optional
        Color limits (vmin, vmax). If None, autoscale.
    to_geo : callable, optional
        Function to transform (x, y) in projected coordinates to (lon, lat),
    **kwargs : additional keyword args for tricontourf

    Returns
    -------
    ax : matplotlib Axes
        The axes with the plot.
    """

    # Copy coordinates to avoid modifying original mesh
    crd = np.array(msh_t.vert2["coord"], copy=True)
    cnn = msh_t.tria3["index"]
    val = msh_t.value.flatten()

    # Transform to geographic coordinates if needed
    if to_geo is not None:
        crd[:, 0], crd[:, 1] = to_geo(crd[:, 0], crd[:, 1])

    if ax is None:
        _, ax = plt.subplots()

    mappable = ax.tricontourf(crd[:, 0], crd[:, 1], cnn, val, **kwargs)

    if colorbar:
        if clim is not None:
            mappable.set_clim(*clim)
        cbar = plt.colorbar(mappable, ax=ax)
        cbar.set_label("Mesh spacing conditioning (m)")

    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    return ax


def plot_bathymetry(rasters_path: List[str], polygon: Polygon, ax: Axes) -> Axes:
    """
    Plots bathymetric raster data and overlays a polygon on top of it.

    Parameters
    ----------
    rasters_path : List[str]
        A list of file paths to the raster files.
    polygon : Polygon
        A polygon to overlay on the raster data.
    ax : Axes
        The axes on which to plot the data.

    Returns
    -------
    ax : Axes
        The axes object with the plotted raster data and polygon.
    """

    data = []
    for path in rasters_path:
        with rasterio.open(path) as src:
            raster_data = src.read(1)
            no_data_value = src.nodata
            if no_data_value is not None:
                raster_data = np.ma.masked_equal(raster_data, no_data_value)
            data.append(raster_data)
            transform = src.transform

    x_polygon, y_polygon = polygon.exterior.xy

    height, width = data[0].shape
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

    im = ax.imshow(
        data[0],
        cmap=cmap,
        norm=norm,
        extent=(np.min(xs), np.max(xs), np.min(ys), np.max(ys)),
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Depth (m)")
    ax.set_title("Raster")

    ax.plot(x_polygon, y_polygon, color="red", linewidth=1)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    return ax


def clip_bathymetry(
    input_raster_paths: List[str],
    output_path: str,
    domain: Polygon,
    margin: float,
) -> float:
    """
    Clips bathymetric raster data using a specified domain polygon,
    applies a margin buffer, and saves the clipped raster.

    Parameters
    ----------
    input_raster_paths : List[str]
        List of file paths to the input raster files.
    output_path : str
        Destination file path to save the clipped raster.
    domain : Polygon
        Polygon geometry used to clip the raster.
    margin : float
        Buffer factor applied to the domain before clipping.

    Returns
    -------
    float
        The mean resolution of the raster.
    """

    buffered_polygon = buffer_area_for_polygon(domain, margin)

    for path in input_raster_paths:
        with rasterio.open(path) as src:
            domain_gdf = gpd.GeoDataFrame(
                index=[0], geometry=[buffered_polygon], crs="EPSG:4326"
            ).to_crs(src.crs)

            out_image, out_transform = mask(
                src, [mapping(domain_gdf.geometry[0])], crop=True
            )

            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                }
            )
        mean_raster_resolution = get_raster_resolution(path)
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)
    return mean_raster_resolution


def get_raster_resolution(raster_path: str) -> float:
    """
    Get the mean resolution of a raster in meters.

    Parameters
    ----------
    raster_path : str
        Path to the raster file.
    Returns
    -------
    float
        Mean resolution of the raster in meters.
    ----------
    Notes
    This function uses rasterio to open the raster file and extract its resolution.
    The mean resolution is calculated as the average of the absolute values of the x and y resolutions.
    """

    with rasterio.open(raster_path) as src:
        res_x, res_y = src.res
        mean_resolution = (abs(res_x) + abs(res_y)) / 2
    return mean_resolution


def clip_bati_manning(
    rasters_path: List[str],
    output_path: str,
    domain: Polygon,
    mas: float,
    UTM: bool,
    manning: float,
) -> None:
    """
    Clips bathymetric raster data using a specified domain polygon and applies
    Manning's coefficient.

    Parameters
    ----------
    rasters_path : List[str]
        A list of file paths to the raster files to be clipped.
    output_path : str
        The file path to save the clipped raster data.
    domain : Polygon
        The domain polygon used to clip the rasters.
    mas : float
        A buffer factor applied to the domain polygon based on its area and length.
    UTM : bool
        If True, assumes the coordinate reference system is EPSG:4326;
        otherwise, assumes EPSG:32630 (UTM projection).
    manning : float
        The Manning's coefficient to apply to the raster data.
    """

    original_polygon = buffer_area_for_polygon(domain, mas)

    if UTM:
        crrs = "EPSG:4326"
    else:
        crrs = "EPSG:32630"
    for path in rasters_path:
        with rasterio.open(path) as src:
            gdf_polygon = gpd.GeoDataFrame(
                index=[0], geometry=[original_polygon], crs=crrs
            )
            gdf_polygon = gdf_polygon.to_crs(src.crs)

            out_image, out_transform = mask(
                src, [mapping(gdf_polygon.geometry[0])], crop=True
            )

            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                }
            )

    out_image = np.ones(out_image.shape) * (-manning)
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)


def plot_boundaries(mesh: jigsaw_msh_t, ax: Axes, to_geo: callable = None) -> None:
    """
    Plots the boundaries of a mesh, including ocean, interior (islands), and land areas.

    Parameters
    ----------
    mesh : jigsaw_msh_t
        The mesh object containing the mesh data and boundaries.
    ax : Axes
        The axes on which to plot the boundaries.
    to_geo : callable, optional
        A function to transform coordinates from projected to geographic CRS.
    """

    plot_mesh_edge(mesh.msh_t, to_geo=to_geo, ax=ax, color="gray", lw=0.5)

    def plot_boundary(gdf, color, label):
        try:
            if to_geo:
                gdf = gdf.copy()
                gdf["geometry"] = gdf["geometry"].apply(
                    lambda geom: transform(to_geo, geom)
                )
            gdf.plot(ax=ax, color=color, label=label)
        except Exception as _e:
            print(f"No {label} boundaries available")

    plot_boundary(mesh.boundaries.ocean(), color="b", label="Ocean")
    plot_boundary(mesh.boundaries.interior(), color="g", label="Islands")
    plot_boundary(mesh.boundaries.land(), color="r", label="Land")

    ax.set_title("Mesh Boundaries")
    ax.legend()


def plot_bathymetry_interp(mesh: jigsaw_msh_t, to_geo, ax: Axes) -> None:
    """
    Plots the interpolated bathymetry data on a mesh.

    Parameters
    ----------
    mesh : jigsaw_msh_t
        The mesh object containing the bathymetry values and mesh structure.
    ax : Axes
        The axes on which to plot the interpolated bathymetry.
    to_geo : callable
        A function to transform coordinates from projected to geographic CRS.
    """

    crd = np.array(mesh.msh_t.vert2["coord"], copy=True)

    if to_geo is not None:
        crd[:, 0], crd[:, 1] = to_geo(crd[:, 0], crd[:, 1])
        bnd = [crd[:, 0].min(), crd[:, 0].max(), crd[:, 1].min(), crd[:, 1].max()]

    triangle = mesh.msh_t.tria3["index"]
    Z = np.mean(mesh.msh_t.value.flatten()[triangle], axis=1)
    vmin = np.nanmin(Z)
    vmax = np.nanmax(Z)

    cmap, norm = join_colormaps(
        cmap1=hex_colors_water,
        cmap2=hex_colors_land,
        value_range1=(vmin, 0.0),
        value_range2=(0.0, vmax),
        name="raster_cmap",
    )

    im = ax.tripcolor(
        crd[:, 0],
        crd[:, 1],
        triangle,
        facecolors=Z,
        cmap=cmap,
        norm=norm,
    )
    ax.set_title("Interpolated Bathymetry")
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Depth (m)")
    ax.set_extent([*bnd], crs=ccrs.PlateCarree())


def simply_polygon(base_shape: Polygon, simpl_UTM: float, project) -> Polygon:
    """
    Simplifies a polygon by transforming it to a projected coordinate system (e.g., UTM),
    applying geometric simplification, and transforming it back to geographic coordinates.

    Parameters
    ----------
    base_shape : Polygon
        The input polygon in geographic coordinates (e.g., WGS84).
    simpl_UTM : float
        Tolerance for simplification (in projected units, typically meters). Higher values result in simpler shapes.
    project : pyproj.Transformer.transform
        A projection function that converts coordinates from geographic to projected (e.g., WGS84 → UTM).

    Returns
    -------
    Polygon
        The simplified polygon in geographic coordinates.

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> from pyproj import Transformer
    >>> from shapely.ops import transform
    >>> base_shape = Polygon([(0, 0), (1, 1), (1, 0), (0, 0)])
    >>> project = Transformer.from_crs("EPSG:4326", "EPSG:32630").transform
    >>> simpl_UTM = 100.0  # Simplification tolerance in meters
    >>> simplified_shape = simply_polygon(base_shape, simpl_UTM, project)
    >>> print(simplified_shape)
    """

    base_shape_utm = transform(project, base_shape)

    simple_shape_utm = base_shape_utm.simplify(simpl_UTM)

    simple_shape = transform(
        lambda x, y: project(x, y, direction=TransformDirection.INVERSE),
        simple_shape_utm,
    )

    return simple_shape


def remove_islands(base_shape: Polygon, threshold_area: float, project) -> Polygon:
    """
    Transforms a polygon to a projected coordinate system (e.g., UTM), filters out small interior rings
    (holes) based on a minimum area threshold, and transforms the simplified polygon back to geographic coordinates.

    Parameters
    ----------
    base_shape : Polygon
        The input polygon in geographic coordinates (e.g., WGS84).
    threshold_area : float
        Minimum area (in projected units, e.g., square meters) for interior rings (holes) to be preserved.
        Interior rings smaller than this threshold will be removed.
    project : callable
        A projection function, typically `pyproj.Transformer.transform`, that converts coordinates
        from geographic to projected CRS.

    Returns
    -------
    Polygon
        The polygon with small interior rings removed, transformed back to geographic coordinates.

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> from pyproj import Transformer
    >>> from shapely.ops import transform
    >>> base_shape = Polygon([(0, 0), (1, 1), (1, 0), (0, 0)])
    >>> project = Transformer.from_crs("EPSG:4326", "EPSG:32630").transform
    >>> threshold_area = 100.0  # Minimum area for interior rings in square meters
    >>> simplified_shape = remove_islands(base_shape, threshold_area, project)
    >>> print(simplified_shape)
    """

    base_shape_utm = transform(project, base_shape)
    interior_shape_utm = [
        interior
        for interior in base_shape_utm.interiors
        if Polygon(interior).area >= threshold_area
    ]
    simple_shape_utm = Polygon(base_shape_utm.exterior, interior_shape_utm)
    simple_shape = transform(
        lambda x, y: project(x, y, direction=TransformDirection.INVERSE),
        simple_shape_utm,
    )

    return simple_shape


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

    Examples
    --------
    >>> Elmts = np.array([[0, 1, 2], [1, 2, 3], [2, 0, 3]])
    >>> edges = calculate_edges(Elmts)
    >>> print(edges)
    [[0 1]
     [0 2]
     [1 2]
     [1 3]
     [2 3]]
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


def decode_open_boundary_data(data: List[str]) -> dict:
    """
    Decodes open boundary data from a given list of strings and
    returns a dictionary containing boundary information.

    Parameters
    ----------
    data : List[str]
        List of strings containing boundary data.

    Returns
    -------
    dict
        A dictionary with keys corresponding to open boundary identifiers (e.g., 'open_boundary_1')
        and values as lists of integers representing boundary node indices.

    Examples
    --------
    >>> data = [
    ...     "100! 200! 300!",
    ...     "open_boundary_1",
    ...     "open_boundary_2",
    ...     "open_boundary_3",
    ...     "land boundaries",
    ...     "open_boundary_1! 10",
    ...     "open_boundary_2! 20",
    ...     "open_boundary_3! 30",
    ... ]
    >>> boundaries = decode_open_boundary_data(data)
    >>> print(boundaries)
    {'open_boundary_1': [10], 'open_boundary_2': [20], 'open_boundary_3': [30]}
    """

    N_obd = int(data[0].split("!")[0])
    boundary_info = {}
    key = data[2][-16:-1]
    boundary_info[key] = []

    for line in data[3:-1]:
        line = line.strip()
        if "!" not in line:
            N = int(line)
            boundary_info[key].append(N)
        else:
            if "land boundaries" in line:
                if len(boundary_info) != N_obd:
                    print("reading error")
                return boundary_info
            match = re.search(r"open_boundary_\d+", line)
            key = match.group(0)
            boundary_info[key] = []

    return boundary_info


def compute_circumcenter(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Technical Reference Manual D-Flow Flexible Mesh 13 May 2025 Revision: 80268
    Compute the circumcenter of a triangle from its 3 vertices.
    Ref: Figure 3.12, Equation (3.6), D-Flow Flexible Mesh Technical Reference Manual.

    Parameters
    ----------
    p0, p1, p2 : np.ndarray
        2D coordinates of the triangle vertices.

    Returns
    -------
    np.ndarray
        2D coordinates of the circumcenter.

    Examples
    --------
    >>> p0 = np.array([0, 0])
    >>> p1 = np.array([1, 0])
    >>> p2 = np.array([0, 1])
    >>> center = compute_circumcenter(p0, p1, p2)
    >>> print(center)
    [0.5 0.5]
    """

    A = p1 - p0
    B = p2 - p0
    AB_perp = np.array([A[1], -A[0]])
    AC_perp = np.array([B[1], -B[0]])
    mid_AB = (p0 + p1) / 2
    mid_AC = (p0 + p2) / 2
    M = np.array([AB_perp, -AC_perp]).T
    b = mid_AC - mid_AB
    try:
        t = np.linalg.solve(M, b)
        center = mid_AB + t[0] * AB_perp
    except np.linalg.LinAlgError:
        center = (p0 + p1 + p2) / 3
    return center


def build_edge_to_cells(elements: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    """
    Technical Reference Manual D-Flow Flexible Mesh 13 May 2025 Revision: 80268
    Build edge -> list of adjacent element indices (max 2).
    Ref: Connectivity structure implied in Section 3.5.1 and Fig 3.2a.

    Parameters
    ----------
    elements : np.ndarray
        Array of triangle elements (indices of vertices).

    Returns
    -------
    edge_to_cells : Dict[Tuple[int, int], List[int]]
        Dictionary mapping edges to the list of adjacent element indices.

    Examples
    --------
    >>> elements = np.array([[0, 1, 2], [1, 2, 3], [2, 0, 3]])
    >>> edge_to_cells = build_edge_to_cells(elements)
    >>> print(edge_to_cells)
    {(0, 1): [0], (0, 2): [0, 2], (1, 2): [0, 1], (1, 3): [1], (2, 3): [1]}
    """

    edge_to_cells = defaultdict(list)
    for idx, elem in enumerate(elements):
        for i in range(3):
            a = elem[i]
            b = elem[(i + 1) % 3]
            edge = tuple(sorted((a, b)))
            edge_to_cells[edge].append(idx)

    return edge_to_cells


def detect_circumcenter_too_close(
    X: np.ndarray, Y: np.ndarray, elements: np.ndarray, aj_threshold: float = 0.1
) -> np.ndarray:
    """
    Technical Reference Manual D-Flow Flexible Mesh 13 May 2025 Revision: 80268
    Detect elements where the distance between adjacent circumcenters is small compared
    to the shared edge length: aj = ||xRj - xLj|| / ||x1j - x0||
    Ref: Equation (3.6) in Section 3.5.1 (Grid Orthogonalization), D-Flow Flexible Mesh Manual.

    Parameters
    ----------
    X, Y : np.ndarray
        1D arrays of x and y coordinates of the nodes.
    elements : np.ndarray
        2D array of shape (nelmts, 3) containing the node indices for each triangle element.
    aj_threshold : float, optional
        Threshold for the ratio of circumcenter distance to edge length. Default is 0.1.

    Returns
    -------
    bad_elements_mask : np.ndarray
        Boolean mask indicating which elements are problematic (True if bad).

    Examples
    --------
    >>> X = np.array([0, 1, 0, 1])
    >>> Y = np.array([0, 0, 1, 1])
    >>> elements = np.array([[0, 1, 2], [1, 3, 2]])
    >>> bad_elements = detect_circumcenter_too_close(X, Y, elements, aj_threshold=0.1)
    >>> print(bad_elements)
    [False False]
    """

    nodes = np.column_stack((X, Y))
    centers = np.array(
        [
            compute_circumcenter(nodes[i0], nodes[i1], nodes[i2])
            for i0, i1, i2 in elements
        ]
    )  # Ref: Fig 3.12

    edge_to_cells = build_edge_to_cells(elements)
    bad_elements_mask = np.zeros(len(elements), dtype=bool)

    for edge, cells in edge_to_cells.items():
        if len(cells) != 2:
            continue  # Internal edges only (Ref: Ji in Eq. 3.7)

        idx0, idx1 = cells
        c0 = centers[idx0]  # x_Lj
        c1 = centers[idx1]  # x_Rj
        node0 = nodes[edge[0]]  # x0
        node1 = nodes[edge[1]]  # x1j

        edge_length = np.linalg.norm(node1 - node0)  # Denominator of aj (||x1j - x0||)
        center_dist = np.linalg.norm(c1 - c0)  # Numerator of aj (||xRj - xLj||)

        aj = center_dist / edge_length if edge_length > 0 else 0  # Equation (3.6)

        if aj < aj_threshold:
            # If the ratio is too low, mark both elements as problematic
            bad_elements_mask[idx0] = True
            bad_elements_mask[idx1] = True

    return bad_elements_mask


def define_mesh_target_size(
    rasters: List[rasterio.io.DatasetReader],
    raster_resolution_meters: float,
    depth_ranges: dict,
    nprocs: int = 1,
) -> ocsmesh.Hfun:
    """
    Define the mesh target size based on depth ranges and their corresponding values.

    Parameters
    ----------
    rasters : List[rasterio.io.DatasetReader]
        List of raster objects.
    raster_resolution_meters : float
        Resolution of the raster in meters.
    depth_ranges : dict, optional
        Dictionary containing depth ranges and their corresponding mesh sizes and rates.
        Format: {(lower_bound, upper_bound): {'value': mesh_size, 'rate': expansion_rate}}
    nprocs : int, optional
        Number of processors to use for the mesh generation. Default is 1.

    Returns
    -------
    mesh_spacing : ocsmesh.Hfun
        Hfun object with the defined mesh target size.
    """

    if depth_ranges is None:
        # Default depth-to-mesh size mapping
        depth_ranges = {
            (-200_000, -250): {"value": 26_000, "rate": 0.0001},  # Very deep ocean
            (-250, -100): {"value": 13_000, "rate": 0.0001},  # Continental slope
            (-100, -75): {"value": 6_500, "rate": 0.0001},  # Outer shelf
            (-75, -25): {"value": 3_250, "rate": 0.0001},  # Mid shelf
            (-25, 2.5): {"value": 1_700, "rate": 0.0001},  # Coastal zone
        }

    all_values = [zone["value"] for zone in depth_ranges.values()]

    points_by_cell = 1  # Number of depth points per minimum size cell for the final cell size definition

    rasters_copy = deepcopy(rasters)
    for raster in rasters_copy:
        raster.resample(
            scaling_factor=raster_resolution_meters * points_by_cell / min(all_values)
        )

    mesh_spacing = ocsmesh.Hfun(
        rasters_copy,
        hmin=min(all_values),
        hmax=max(all_values),
        nprocs=nprocs,
    )

    for (lower, upper), params in depth_ranges.items():
        mesh_spacing.add_topo_bound_constraint(
            value=params["value"],
            lower_bound=lower,
            upper_bound=upper,
            value_type="max",
            rate=params["rate"],
        )

    return mesh_spacing


def read_lines(poly_line: str) -> MultiLineString:
    """
    Reads a CSV file containing coordinates of a polyline and returns a MultiLineString.
    The CSV file should have two columns for x and y coordinates, with NaN values indicating breaks in the line.
    Parameters
    ----------
    poly_line : str
        Path to the CSV file containing the polyline coordinates
    Returns
    -------
    MultiLineString
        A MultiLineString object representing the polyline segments
    """

    coords_line = pd.read_csv(poly_line, sep=",", header=None)
    segments = []
    current_segment = []
    for index, row in coords_line.iterrows():
        if row.isna().any():
            if current_segment:
                segments.append(LineString(current_segment))
                current_segment = []
        else:
            current_segment.append(tuple(row))

    if current_segment:
        segments.append(LineString(current_segment))
    return MultiLineString(segments)


def get_raster_resolution_meters(lon_center, lat_center, raster_resolution, project):
    """
    Calculate the raster resolution in meters based on the center coordinates and the raster resolution in degrees.

    Parameters
    ----------
    lon_center : float
        Longitude of the center point.
    lat_center : float
        Latitude of the center point.
    raster_resolution : float
        Raster resolution in degrees.
    Returns
    -------
    float
        Raster resolution in meters.
    """
    x_center, y_center = project(lon_center, lat_center)
    x_center_raster_resolution, y_center_raster_resolution = project(
        lon_center + raster_resolution / np.sqrt(2),
        lat_center + raster_resolution / np.sqrt(2),
    )
    raster_resolution_meters = np.mean(
        [
            abs(x_center - x_center_raster_resolution),
            abs(y_center - y_center_raster_resolution),
        ]
    )
    return raster_resolution_meters


if __name__ == "__main__":
    # Example usage
    from pyproj import Transformer
    from shapely.geometry import Polygon

    base_shape = Polygon([(0, 0), (1, 1), (1, 0), (0, 0)])
    project = Transformer.from_crs("EPSG:4326", "EPSG:32630").transform
    simpl_UTM = 100.0  # Simplification tolerance in meters
    simplified_shape = simply_polygon(base_shape, simpl_UTM, project)
    print(simplified_shape)
