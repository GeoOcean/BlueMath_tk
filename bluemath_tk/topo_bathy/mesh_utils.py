import re
from typing import Any, List, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import xarray as xr
from jigsawpy.msh_t import jigsaw_msh_t
from matplotlib.axes import Axes
from matplotlib.tri import Triangulation
from netCDF4 import Dataset
from rasterio.mask import mask
from shapely.geometry import MultiPolygon, Point, Polygon, mapping
from shapely.ops import transform


def plot_mesh_edge(msh_t: jigsaw_msh_t, ax: Axes = None, **kwargs) -> Axes:
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
    **kwargs : keyword arguments, optional
        Additional keyword arguments passed to the `triplot` function.
        These can be used to customize the plot (e.g., color, line style).

    Returns
    -------
    ax : Axes
        The axes object with the plotted mesh edges.
    """

    crd = msh_t.vert2["coord"]
    cnn = msh_t.tria3["index"]

    if ax is None:
        _fig, ax = plt.subplots()
    ax.triplot(crd[:, 0], crd[:, 1], cnn, **kwargs)
    ax.set_title("Mesh Design Criteria")
    ax.set_xlabel("X UTM")
    ax.set_ylabel("Y UTM")

    return ax


def plot_mesh_vals(
    msh_t: jigsaw_msh_t,
    ax: Axes = None,
    colorbar: bool = True,
    clim: tuple = None,
    **kwargs,
) -> Axes:
    """
    Plots the mesh values on a triangular mesh.

    Parameters
    ----------
    msh_t : jigsaw_msh_t
        An object containing the mesh data. It must have:
        - 'vert2['coord']' containing the coordinates of the mesh vertices.
        - 'tria3['index']' containing the indices of the triangles.
        - 'value' containing the mesh values to be plotted.
    ax : Axes, optional
        The axes to plot on. If None, a new plot is created. Default is None.
    colorbar : bool, optional
        Whether to display the colorbar. Default is True.
    clim : tuple, optional
        The limits for the color scale. If None, the limits are automatically
        determined from the data. Default is None.
    **kwargs : keyword arguments, optional
        Additional keyword arguments passed to the `tricontourf` function.
        These can be used to customize the plot (e.g., color, line style).

    Returns
    -------
    ax : Axes
        The axes object with the plotted mesh values.
    """

    crd = msh_t.vert2["coord"]
    cnn = msh_t.tria3["index"]
    val = msh_t.value.flatten()

    if ax is None:
        _fig, ax = plt.subplots()
    mappable = ax.tricontourf(crd[:, 0], crd[:, 1], cnn, val, **kwargs)
    if colorbar:
        if clim is not None:
            mappable.set_clim(*clim)
        _cb = plt.colorbar(mappable, ax=ax)

    return ax


def plot_bati(rasters_path: List[str], polygon: Polygon, ax: Axes) -> Axes:
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

    im = ax.imshow(
        data[0], cmap="terrain", extent=(np.min(xs), np.max(xs), np.min(ys), np.max(ys))
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Depth (m)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Raster")

    ax.plot(x_polygon, y_polygon, color="red", linewidth=1)
    ax.axis("equal")

    return ax


def clip_bati(
    rasters_path: List[str], output_path: str, domain: Polygon, mas: float, UTM: bool
) -> None:
    """
    Clips bathymetric raster data using a specified domain polygon
    and saves the clipped raster to the specified output path.

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
    """

    original_polygon = domain.buffer(mas * domain.area / domain.length)

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

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)


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

    original_polygon = domain.buffer(mas * domain.area / domain.length)

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


def plot_poly(largest_polygon: Polygon, final_polygon: Polygon) -> Axes:
    """
    Plots two polygons on a map: the largest polygon and the final polygon.

    Parameters
    ----------
    largest_polygon : Polygon
        The largest polygon to plot.
    final_polygon : Polygon
        The final polygon to plot.

    Returns
    -------
    ax : Axes
        The axes object with the plotted polygons.
    """

    exterior_points = list(largest_polygon.exterior.coords)
    interior_points = [list(interior.coords) for interior in largest_polygon.interiors]
    all_points = exterior_points + [
        point for island in interior_points for point in island
    ]

    exterior_points_1 = list(final_polygon.exterior.coords)
    interior_points_1 = [list(interior.coords) for interior in final_polygon.interiors]
    all_points_1 = exterior_points_1 + [
        point for island in interior_points_1 for point in island
    ]

    x1, y1 = zip(*all_points_1)
    x, y = zip(*all_points)
    xx, yy = largest_polygon.exterior.xy

    _fig, ax = plt.subplots()
    ax.fill(xx, yy, alpha=0.5, fc="lightgrey", ec="black")
    ax.scatter(x, y, color="red", s=1, label="Initial Polygon Points")
    ax.scatter(x1, y1, color="black", s=1, label="Final Polygon Points")
    ax.axis("equal")
    ax.set_title("Polygon Domain")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()

    return ax


def plot_boundaries(mesh: jigsaw_msh_t, ax: Axes) -> Axes:
    """
    Plots the boundaries of a mesh, including ocean, interior (islands), and land areas.

    Parameters
    ----------
    mesh : jigsaw_msh_t
        The mesh object containing the mesh data and boundaries.
    ax : Axes
        The axes on which to plot the boundaries.

    Returns
    -------
    ax : Axes
        The axes object with the plotted boundaries.
    """

    plot_mesh_edge(mesh.msh_t, ax=ax, lw=0.2, color="c")

    try:
        mesh.boundaries.ocean().plot(ax=ax, color="b", label="Ocean")
    except Exception as e:
        print(f"No Ocean boundaries available. Error: {e}")
    try:
        mesh.boundaries.interior().plot(ax=ax, color="g", label="Islands")
    except Exception as e:
        print(f"No Islands boundaries available. Error: {e}")
    try:
        mesh.boundaries.land().plot(ax=ax, color="r", label="Land")
    except Exception as e:
        print(f"No Land boundaries available. Error: {e}")

    ax.legend()
    ax.axis("equal")
    ax.set_title("Mesh Boundaries")
    ax.set_xlabel("X UTM")
    ax.set_ylabel("Y UTM")

    return ax


def plot_bati_interp(mesh: jigsaw_msh_t, ax: Axes) -> Axes:
    """
    Plots the interpolated bathymetry data on a mesh.

    Parameters
    ----------
    mesh : jigsaw_msh_t
        The mesh object containing the bathymetry values and mesh structure.
    ax : Axes
        The axes on which to plot the interpolated bathymetry.

    Returns
    -------
    ax : Axes
        The axes object with the plotted interpolated bathymetry.
    """

    im = ax.tricontourf(
        Triangulation(
            mesh.msh_t.vert2["coord"][:, 0],
            mesh.msh_t.vert2["coord"][:, 1],
            triangles=mesh.msh_t.tria3["index"],
        ),
        mesh.msh_t.value.flatten(),
    )
    ax.set_title("Interpolated Bathymetry")
    ax.axis("equal")
    ax.set_xlabel("X UTM")
    ax.set_ylabel("Y UTM")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Depth (m)")

    return ax


def simply_poly(base_shape: Polygon, simpl_UTM: float, UTM_zone: Any) -> Polygon:
    """
    Simplifies the input polygon, optionally transforming it to UTM coordinates for simplification.

    Parameters
    ----------
    base_shape : Polygon
        The polygon to be simplified.
    simpl_UTM : float
        The tolerance for simplification. A higher value results in greater simplification.
    UTM_zone : Any
        The UTM zone for transformation. If None, no transformation is done, and the input shape
        is simplified in its current coordinate system.

    Returns
    -------
    Polygon
        The simplified polygon.
    """

    if UTM_zone:
        transformer_to_utm = pyproj.Transformer.from_proj(
            pyproj.Proj(proj="latlong", datum="WGS84"),
            pyproj.Proj(proj="utm", zone=UTM_zone, ellps="WGS84"),
        )
        base_shape_utm = transform(
            lambda x, y: transformer_to_utm.transform(x, y), base_shape
        )
        simple_shape_UTM = base_shape_utm.simplify(simpl_UTM)

        transformer_to_latlon = pyproj.Transformer.from_proj(
            pyproj.Proj(proj="utm", zone=UTM_zone, ellps="WGS84"),
            pyproj.Proj(proj="latlong", datum="WGS84"),
        )

        simple_shape = transform(
            lambda x, y: transformer_to_latlon.transform(x, y), simple_shape_UTM
        )
    else:
        simple_shape = base_shape.simplify(simpl_UTM)

    return simple_shape


def remove_islands(
    main_polygon: Union[Polygon, MultiPolygon], threshold_area: float, UTM_zone: Any
) -> Union[Polygon, MultiPolygon]:
    """
    Removes small islands (interior polygons) from a given polygon based on a threshold area.

    Parameters
    ----------
    main_polygon : Union[Polygon, MultiPolygon]
        The main polygon which may contain smaller islands (interior polygons).
    threshold_area : float
        The minimum area required for an interior polygon to be retained.
        Islands with smaller areas are removed.
    UTM_zone : int
        The UTM zone to which the coordinates will be transformed for accurate area calculation.

    Returns
    -------
    Union[Polygon, MultiPolygon]
        The resulting polygon with small islands removed.
    """

    transformer_to_utm = pyproj.Transformer.from_proj(
        pyproj.Proj(proj="latlong", datum="WGS84"),
        pyproj.Proj(proj="utm", zone=UTM_zone, ellps="WGS84"),
    )

    main_polygon_utm = transform(
        lambda x, y: transformer_to_utm.transform(x, y), main_polygon
    )

    if isinstance(main_polygon_utm, MultiPolygon):
        new_polygons = []
        for poly in main_polygon_utm.geoms:
            new_poly = remove_islands(poly, threshold_area, UTM_zone)
            new_polygons.append(new_poly)
        result_polygon = MultiPolygon(new_polygons)

    elif isinstance(main_polygon_utm, Polygon):
        new_interior = [
            interior
            for interior in main_polygon_utm.interiors
            if Polygon(interior).area >= threshold_area
        ]
        result_polygon = Polygon(main_polygon_utm.exterior, new_interior)

    transformer_to_latlon = pyproj.Transformer.from_proj(
        pyproj.Proj(proj="utm", zone=UTM_zone, ellps="WGS84"),
        pyproj.Proj(proj="latlong", datum="WGS84"),
    )

    result_polygon_latlon = transform(
        lambda x, y: transformer_to_latlon.transform(x, y), result_polygon
    )

    return result_polygon_latlon


def is_any_point_outside(triangle_coords: List[tuple], poly: Polygon) -> bool:
    """
    Checks if any of the points (vertices) of a triangle are outside a given polygon.

    Parameters
    ----------
    triangle_coords : List[tuple]
        Coordinates of the triangle vertices [(x1, y1), (x2, y2), (x3, y3)].
    poly : Polygon
        The polygon within which the triangle points should be checked.

    Returns
    -------
    bool
        True if any point is outside the polygon, False if all points are inside.
    """

    return any(not Point(coord).within(poly) for coord in triangle_coords)


def circumcenter(triangles_coords: np.ndarray) -> np.ndarray:
    """
    Calculates the circumcenter of a triangle given its vertex coordinates.

    Parameters
    ----------
    triangles_coords : np.ndarray
        A 2D array of shape (n, 3, 2), where each row represents the coordinates
        of a triangle's three vertices.

    Returns
    -------
    np.ndarray
        A 2D array of shape (n, 2), where each row represents the (x, y) c
        oordinates of the circumcenter for each triangle.
    """

    triangles_coords = np.array(triangles_coords)

    x1, y1 = triangles_coords[:, 0, 0], triangles_coords[:, 0, 1]
    x2, y2 = triangles_coords[:, 1, 0], triangles_coords[:, 1, 1]
    x3, y3 = triangles_coords[:, 2, 0], triangles_coords[:, 2, 1]

    A = 0.5 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    D_x = (
        (x1**2 + y1**2) * (y2 - y3)
        + (x2**2 + y2**2) * (y3 - y1)
        + (x3**2 + y3**2) * (y1 - y2)
    )
    D_y = (
        (x1**2 + y1**2) * (x3 - x2)
        + (x2**2 + y2**2) * (x1 - x3)
        + (x3**2 + y3**2) * (x2 - x1)
    )

    x_circumcenter = D_x / (2 * A)
    y_circumcenter = D_y / (2 * A)

    return np.vstack((x_circumcenter, y_circumcenter)).T


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


def adcirc2netcdf(Path_grd: str, netcdf_path: str) -> None:
    """
    Converts ADCIRC grid data to a NetCDF Delft3DFM format.

    Parameters
    ----------
    Path_grd : str
        Path to the ADCIRC grid file.
    netcdf_path : str
        Path where the resulting NetCDF file will be saved.

    TODO: Check the whole function for correctness and completeness.
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

        dataset.institution = "Deltares"
        dataset.references = "http://www.deltares.nl"
        dataset.source = "RGFGRID 7.03.00.77422. Model: ---"
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


def extract_pos_nearest_points_regular(
    xds_grid: xr.Dataset, lon_points: np.ndarray, lat_points: np.ndarray
) -> Tuple[int, int]:
    """
    Find the closest grid point indices for given longitude and latitude points.

    Parameters
    ----------
    xds_grid : xr.Dataset
        Dataset containing grid data (lon_z, lat_z, hz).
    lon_points : np.ndarray
        Longitudes of interest.
    lat_points : np.ndarray
        Latitudes of interest.

    Returns
    -------
    pos_lon_points_mesh : int
        Index of closest longitude grid point.
    pos_lat_points_mesh : int
        Index of closest latitude grid point.

    Notes
    -----
    Converts negative longitudes by adding 360 and uses squared differences to find
    the nearest grid point. Assumes grid is regularly spaced.
    """

    if lon_points < 0:
        lon_points = lon_points + 360

    z = xds_grid.hz.values
    Lon_grid = xds_grid.lon_z.values
    Lat_grid = xds_grid.lat_z.values
    LLat, LLon = np.meshgrid(Lat_grid, Lon_grid)

    Lon_val = LLon[z != 0]
    Lat_val = LLat[z != 0]

    index = np.nanargmin((lat_points - Lat_val) ** 2 + (lon_points - Lon_val) ** 2)

    pos_lon_points_mesh = Lon_val[index]
    pos_lat_points_mesh = Lat_val[index]

    return pos_lon_points_mesh, pos_lat_points_mesh
