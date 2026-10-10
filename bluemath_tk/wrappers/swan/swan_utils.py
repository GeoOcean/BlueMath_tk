import os.path as op

import numpy as np
import pandas as pd
import xarray as xr

from scipy.io import loadmat, whosmat
import warnings

from ...core.operations import nautical_to_mathematical
from ...additive.additive import (
    create_triangle_mask_from_points,
    read_adcirc_grd,
)

sbatch_file_greenwaves = """#!/bin/bash
#SBATCH --ntasks=1              # Number of tasks (MPI processes)
#SBATCH --partition=geocean     # Standard output and error log
#SBATCH --nodes=1               # Number of nodes to use
#SBATCH --mem=4gb               # Memory per node in GB (see also --mem-per-cpu)
#SBATCH --time=24:00:00

source /nfs/home/geocean/faugeree/miniforge3/etc/profile.d/conda.sh
conda activate work

case_dir=$(ls | awk "NR == $SLURM_ARRAY_TASK_ID")
launchSwan.sh --case-dir $case_dir

output_file="${case_dir}/output.mat"
output_file_raw="${case_dir}/output.raw"

python3 - <<EOF
import os, xarray as xr, numpy as np, struct
from bluemath_tk.wrappers.swan.swan_utils import mat_to_xr_dataset

output_file = r"${output_file}"
output_file_raw = r"${output_file_raw}"

ds = mat_to_xr_dataset(output_file, variables=["Hsig"])
data = ds["Hsig"].values.astype(np.float32)
shape = list(data.shape)
shape += [0] * (4 - len(shape))
header = struct.pack("4i", *shape) + bytes(256 - 16)

with open(output_file_raw, "wb") as f:
    f.write(header)
    f.write(data.tobytes())

# if ${SLURM_ARRAY_TASK_ID} != 1:
#     os.remove(output_file)
EOF
"""

def generate_forcing_file_GreenWaves(
    case_context: dict,
    case_dir: str,
    ds_GFD_info: xr.Dataset,
):
    """
    Generate the wind forcing files for a case in netCDF format (optimized version).
    """
    triangle_index = case_context.get("tesela")
    direction_index = case_context.get("direction")
    wind_direction = ds_GFD_info.wind_directions.values[direction_index]
    wind_speed = case_context.get("wind_magnitude")

    ref_date = pd.to_datetime(ds_GFD_info.reference_date.values.item())
    new_date = ref_date + pd.Timedelta(hours=int(case_context.get("simul_time")))
    d_date = pd.Timedelta(seconds=int(case_context.get("time_step_comp")))

    time = np.arange(ref_date, new_date + d_date, step = d_date, dtype = "datetime64[s]")

    connectivity = ds_GFD_info.triangle_forcing_connectivity
    triangle_longitude = ds_GFD_info.node_forcing_longitude.isel(
        node_forcing_index=connectivity
    ).values
    triangle_latitude = ds_GFD_info.node_forcing_latitude.isel(
        node_forcing_index=connectivity
    ).values

    connectivity_compo = ds_GFD_info.triangle_computation_connectivity.values
    node_lon = ds_GFD_info.node_computation_longitude.values.ravel()
    node_lat = ds_GFD_info.node_computation_latitude.values.ravel()

    # Selection triangle vertices
    x0, x1, x2 = triangle_longitude[triangle_index, :]
    y0, y1, y2 = triangle_latitude[triangle_index, :]
    triangle_vertices = np.array([(x0, y0), (x1, y1), (x2, y2)], dtype=float)

    # Compute centroid with NumPy (avoids shapely dependency)
    centroid = triangle_vertices.mean(axis=0)
    scale_factor = 1.001
    verts_buffered = centroid + (triangle_vertices - centroid) * scale_factor

    # Triangle mask (uses matplotlib Path)
    triangle_mask = create_triangle_mask_from_points(
        node_lon,
        node_lat,
        verts_buffered,
    )

    # Wind computation
    angle_rad = nautical_to_mathematical(wind_direction) * np.pi / 180
    wind_u = -np.cos(angle_rad) * wind_speed
    wind_v = -np.sin(angle_rad) * wind_speed

    # Initialize and assign wind arrays
    n_points = len(node_lon)
    n_time = len(time)
    windx = np.zeros((n_time, n_points))
    windy = np.zeros((n_time, n_points))



    windx[:, triangle_mask] = wind_u
    windy[:, triangle_mask] = wind_v

    with open(op.join(case_dir, case_context.get("forcing_file", "wind_forcing.wind")), "w") as f:
        for t in range(n_time):
            time_str = np.datetime_as_string(time[t], unit="s")
            time_str = time_str.replace("T", ".").replace("-", "").replace(":", "")
            f.write(f"{time_str}\n")
            f.write("wind x component\n")
            np.savetxt(f, windx[t, :].reshape(1, -1), fmt="%.3f")
            f.write("wind y component\n")
            np.savetxt(f, windy[t, :].reshape(1, -1), fmt="%.3f")


def mat_to_xr_dataset(path, variables=None, time_range=None):
    """
    Convert a MATLAB file to xarray.Dataset, loading only requested data.
    
    Parameters
    ----------
    path : str
        Path to .mat file
    variables : list of str, optional
        Variable names to load (e.g., ['Hsig', 'Windv_x']). 
        If None, loads all variables.
    time_range : tuple of (start, end), optional
        Only load timesteps within this range.
        e.g., ('2000-01-01', '2000-01-02')
    
    Returns
    -------
    xr.Dataset
    """
    mat_info = whosmat(path)
    all_keys = [name for name, shape, dtype in mat_info]

    vars_with_keys = {}
    
    for key in all_keys:
        if key.startswith(("Xp", "Yp", "__")):
            continue
        parts = key.rsplit("_", 2)
        if len(parts) >= 3:
            varname = "_".join(parts[:-2])
            timestamp_str = f"{parts[-2]}_{parts[-1]}"
            try:
                timestamp = pd.to_datetime(timestamp_str, format="%Y%m%d_%H%M%S")
                vars_with_keys.setdefault(varname, []).append((key, timestamp))
            except ValueError:
                continue

    if variables is not None:
        vars_with_keys = {k: v for k, v in vars_with_keys.items() if k in variables}
    
    if not vars_with_keys:
        raise ValueError(f"No matching variables found. Available: {list(vars_with_keys.keys())}")

    if time_range is not None:
        t_start, t_end = pd.to_datetime(time_range[0]), pd.to_datetime(time_range[1])
        for varname in vars_with_keys:
            vars_with_keys[varname] = [
                (k, t) for k, t in vars_with_keys[varname] 
                if t_start <= t <= t_end
            ]

    keys_to_load = {"Xp", "Yp"}
    for varname, key_times in vars_with_keys.items():
        keys_to_load.update(k for k, t in key_times)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Duplicate variable name")
        mat = loadmat(path, variable_names=list(keys_to_load))
    
    lon = mat["Xp"].ravel().astype(np.float32)
    lat = mat["Yp"].ravel().astype(np.float32)

    ds_vars = {}

    for varname, key_times in vars_with_keys.items():
        key_times.sort(key=lambda x: x[1])  # Sort by timestamp
        
        n_times = len(key_times)
        n_points = lon.size
        data_array = np.empty((n_times, n_points), dtype=np.float32)
        
        for i, (key, _) in enumerate(key_times):
            data_array[i] = mat[key].ravel()
        
        time_index = pd.DatetimeIndex([t for _, t in key_times])
        
        unique_times, time_indices = np.unique(time_index, return_index=True)
        data_array = data_array[time_indices]
        time_index = pd.DatetimeIndex(unique_times)
        
        ds_vars[varname] = (("time", "points"), data_array)
    
    return xr.Dataset(
        data_vars=ds_vars,
        coords={
            "time": time_index,
            "lon": ("points", lon),
            "lat": ("points", lat),
        },
    )


def vortex2SWAN(
    mesh: str,
    ds_vortex: xr.Dataset,
    output_path: str,
    fill_value: float = 0,
) -> xr.Dataset:
    """
    Convert the vortex dataset to a Delft3D FM compatible netCDF forcing file.

    Parameters
    ----------
    mesh : str
        The mesh path.
    ds_vortex : xarray.Dataset
        The vortex dataset containing wind speed and pressure data.
    path_output : str
        The output path where the netCDF file will be saved.
    ds_name : str
        The name of the output netCDF file, default is "forcing_Tonga_vortex.nc".
    forcing_ext : str
        The extension for the forcing file, default is "GreenSurge_GFDcase_wind.ext".
    fill_value : float
        The fill value to use for missing data, default is 0.
        
    Returns
    -------
    xarray.Dataset
        A dataset containing the interpolated wind speed and pressure data,
        ready for use in Delft3D FM.
    """

    Nodes_calc, _, _ = read_adcirc_grd(mesh)

    longitude = Nodes_calc[:, 1]
    latitude = Nodes_calc[:, 2]

    dt = (ds_vortex.time[1] - ds_vortex.time[0]).values
    time = np.arange(ds_vortex.time.size) * dt
    n_time = len(time)

    lat_interp = xr.DataArray(latitude, dims="node")
    lon_interp = xr.DataArray(longitude, dims="node")
    angle = np.deg2rad((270 - ds_vortex.Dir.values) % 360)
    W = ds_vortex.W.values

    windx_data = (W * np.cos(angle)).astype(np.float32)
    windy_data = (W * np.sin(angle)).astype(np.float32)
    pressure_data = ds_vortex.p.values.astype(np.float32)

    if windx_data.ndim == 3:
        windx_data = np.transpose(windx_data, (2, 0, 1))
        windy_data = np.transpose(windy_data, (2, 0, 1))
        pressure_data = np.transpose(pressure_data, (2, 0, 1))

    ds_vortex_interp = xr.Dataset(
        {
            "windx": (
                ("time", "latitude", "longitude"),
                np.nan_to_num(windx_data, nan=fill_value),
            ),
            "windy": (
                ("time", "latitude", "longitude"),
                np.nan_to_num(windy_data, nan=fill_value),
            ),
            "airpressure": (
                ("time", "latitude", "longitude"),
                np.nan_to_num(pressure_data, nan=fill_value),
            ),
        },
        coords={
            "time": time,
            "latitude": ds_vortex.lat.values,
            "longitude": ds_vortex.lon.values,
        },
    )
    forcing_dataset = ds_vortex_interp.interp(latitude=lat_interp, longitude=lon_interp)

    windx = forcing_dataset.windx.values
    windy = forcing_dataset.windy.values

    with open(output_path, "w") as f:
        for t in range(n_time):
            time_str = np.datetime_as_string(time[t], unit="s")
            time_str = time_str.replace("T", ".").replace("-", "").replace(":", "")
            f.write(f"{time_str}\n")
            f.write("wind x component\n")
            np.savetxt(f, windx[t, :].reshape(1, -1), fmt="%.3f")
            f.write("wind y component\n")
            np.savetxt(f, windy[t, :].reshape(1, -1), fmt="%.3f")

# --------------------------------------------------------------------------------------
# Unstructured SWAN cases forced along the open boundary (HyWaves-style), see
# SwanMetaModelWrapper / SwanDynamicModelWrapper.
# --------------------------------------------------------------------------------------

#: SWAN ``TABLE`` quantity -> short output name (``hs``, ``tp``, ``dir``, ``spr``).
TABLE_VARS: dict[str, str] = {
    "HSIGN": "hs",
    "TPS": "tp",
    "DIR": "dir",
    "DSPR": "spr",
}


def read_adcirc_grid(grd_file: str) -> dict:
    """
    Vertices, triangles and boundary node strings of an ADCIRC grid (``fort.14``).

    Parameters
    ----------
    grd_file : str
        ADCIRC grid file.

    Returns
    -------
    dict
        ``vertices``: ``(n, 3)`` ``x, y, depth`` (depth positive down);
        ``triangles``: ``(m, 3)`` 0-based vertex indices; ``open_boundaries``
        and ``land_boundaries``: lists of 0-based vertex-index arrays, in
        file order.
    """

    def read_strings(f, n_strings: int) -> list[np.ndarray]:
        strings = []
        for _ in range(n_strings):
            n_nodes = int(f.readline().split()[0])
            nodes = [int(f.readline().split()[0]) - 1 for _ in range(n_nodes)]
            strings.append(np.asarray(nodes, dtype=int))
        return strings

    with open(grd_file) as f:
        f.readline()
        n_elements, n_vertices = (int(v) for v in f.readline().split()[:2])
        vertices = np.loadtxt(f, max_rows=n_vertices, usecols=(1, 2, 3))
        triangles = np.loadtxt(f, max_rows=n_elements, usecols=(2, 3, 4), dtype=int)
        triangles -= 1
        n_open = int(f.readline().split()[0])
        f.readline()  # total number of open-boundary nodes
        open_boundaries = read_strings(f, n_open)
        line = f.readline()
        n_land = int(line.split()[0]) if line.strip() else 0
        f.readline()  # total number of land-boundary nodes
        land_boundaries = read_strings(f, n_land)

    return {
        "vertices": vertices,
        "triangles": triangles,
        "open_boundaries": open_boundaries,
        "land_boundaries": land_boundaries,
    }


def clockwise_triangles(grid: dict) -> np.ndarray:
    """
    Mask of the triangles whose vertices are in clockwise order.

    SWAN stops on unstructured grids with any clockwise cell ("not all cells
    have counterclockwise order of vertices").

    Parameters
    ----------
    grid : dict
        Output of :func:`read_adcirc_grid`.

    Returns
    -------
    np.ndarray
        Boolean mask, one entry per triangle.
    """

    v = grid["vertices"][:, :2]
    a, b, c = (v[grid["triangles"][:, k]] for k in range(3))
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
        c[:, 0] - a[:, 0]
    )

    return cross < 0


def write_ccw_adcirc_grid(grd_file: str, out_file: str) -> int:
    """
    Copy an ADCIRC grid with every triangle in counterclockwise order.

    Only the element lines change (two vertices swapped where needed); the
    header, vertices and boundary lists are copied as they are.

    Parameters
    ----------
    grd_file : str
        Source ADCIRC grid.
    out_file : str
        Grid to write.

    Returns
    -------
    int
        Number of triangles that were reoriented.
    """

    grid = read_adcirc_grid(grd_file)
    flip = clockwise_triangles(grid)
    triangles = grid["triangles"].copy()
    triangles[flip] = triangles[flip][:, [0, 2, 1]]
    n_vertices, n_elements = len(grid["vertices"]), len(triangles)

    with open(grd_file) as src, open(out_file, "w") as dst:
        for _ in range(2 + n_vertices):
            dst.write(src.readline())
        for k in range(n_elements):
            src.readline()
            t = triangles[k] + 1
            dst.write(f"{k + 1} 3 {t[0]} {t[1]} {t[2]}\n")
        for line in src:
            dst.write(line)

    return int(flip.sum())


def swan_boundary_side(grid: dict, open_boundary: int = 0) -> tuple[np.ndarray, str]:
    """
    Vertices and ``BOUNDSPEC SIDE`` arguments SWAN uses for one open boundary.

    SWAN marks ADCIRC boundary vertices with the 1-based number of their node
    string (open boundaries first, then land boundaries, which overwrite
    shared vertices), and ``BOUNDSPEC SIDE k CCW|CLOCKWISE`` walks the
    vertices with marker *k* in that direction around the domain, measuring
    ``[len]`` from the first. This returns those vertices in the file order of
    the open boundary, and the direction keyword that makes SWAN walk them in
    the same order.

    Parameters
    ----------
    grid : dict
        Output of :func:`read_adcirc_grid`.
    open_boundary : int, optional
        0-based open boundary. Default is 0.

    Returns
    -------
    tuple[np.ndarray, str]
        ``(n, 2)`` side vertices, and e.g. ``"1 CCW"`` for ``BOUNDSPEC SIDE``.
    """

    from matplotlib.tri import Triangulation

    nodes = grid["open_boundaries"][open_boundary]
    land = {n for b in grid["land_boundaries"] for n in b.tolist()}
    later_open = [
        n for b in grid["open_boundaries"][open_boundary + 1 :] for n in b.tolist()
    ]
    nodes = np.array([n for n in nodes if n not in land and n not in later_open])
    xy = grid["vertices"][nodes, :2]

    # The domain lies to the left of a counterclockwise walk: probe just left
    # of every boundary edge and see whether it falls inside the mesh.
    x, y = grid["vertices"][:, 0], grid["vertices"][:, 1]
    tri = Triangulation(x, y, grid["triangles"])
    edges = np.diff(xy, axis=0)
    length = np.hypot(*edges.T)
    left = np.c_[-edges[:, 1], edges[:, 0]] / length[:, None]
    probes = (xy[:-1] + xy[1:]) / 2 + 0.25 * length[:, None] * left
    inside = tri.get_trifinder()(probes[:, 0], probes[:, 1]) >= 0
    direction = "CCW" if inside.mean() > 0.5 else "CLOCKWISE"

    return xy, f"{open_boundary + 1} {direction}"


def boundary_distances(xy: np.ndarray) -> np.ndarray:
    """
    Cumulative distance along a boundary polyline, as SWAN measures ``[len]``.

    SWAN accumulates ``sqrt(dx**2 + dy**2)`` in grid units between consecutive
    boundary vertices (degrees for spherical coordinates, no cos(lat) factor),
    so ``VARIABLE PAR`` lengths must use the same metric.

    Parameters
    ----------
    xy : np.ndarray
        ``(n, 2)`` boundary vertices in order.

    Returns
    -------
    np.ndarray
        ``(n,)`` distance of each vertex from the first one.
    """

    steps = np.hypot(*np.diff(np.asarray(xy, dtype=float), axis=0).T)

    return np.concatenate(([0.0], np.cumsum(steps)))


def boundary_par_table(
    open_xy: np.ndarray, node_xy: np.ndarray, node_values: dict[str, list]
) -> dict[str, list[float]]:
    """
    ``BOUNDSPEC SIDE ... VARIABLE PAR`` rows for values given at boundary nodes.

    Each node (e.g. a HyWaves goal) is snapped to its nearest boundary
    vertex; SWAN interpolates linearly between them along the boundary. The
    first and last vertices repeat the values of the nearest node, so the
    whole boundary is forced (beyond the last ``[len]`` SWAN would leave it
    unforced).

    Parameters
    ----------
    open_xy : np.ndarray
        ``(n, 2)`` boundary vertices in the order SWAN walks them (see
        :func:`swan_boundary_side`).
    node_xy : np.ndarray
        ``(n_nodes, 2)`` boundary nodes, in order along the open boundary.
    node_values : dict
        ``{"hs": [...], "tp": [...], "dir": [...], "spr": [...]}``, one value
        per node.

    Returns
    -------
    dict[str, list[float]]
        ``len`` plus the *node_values* keys, one entry per ``PAR`` row.

    Raises
    ------
    ValueError
        If the nodes are not in increasing order along the open boundary.
    """

    open_xy = np.asarray(open_xy, dtype=float)
    distances = boundary_distances(open_xy)
    snapped = [
        int(np.argmin(np.hypot(*(open_xy - p).T)))
        for p in np.asarray(node_xy, dtype=float).reshape(-1, 2)
    ]
    if np.any(np.diff(snapped) <= 0):
        raise ValueError(
            f"boundary nodes snap to open-boundary vertices {snapped}: they must "
            "be distinct and ordered along the open boundary"
        )
    lengths = [float(distances[i]) for i in snapped]
    rows = {
        "len": lengths,
        **{k: [float(v) for v in vals] for k, vals in node_values.items()},
    }
    if snapped[0] > 0:
        rows = {k: [0.0 if k == "len" else v[0]] + v for k, v in rows.items()}
    if snapped[-1] < len(open_xy) - 1:
        rows = {
            k: v + [float(distances[-1]) if k == "len" else v[-1]]
            for k, v in rows.items()
        }

    return rows


def write_swan_points(points: np.ndarray, out_path: str) -> str:
    """
    Write ``x y`` output points for ``POINTS 'name' FILE``.

    Parameters
    ----------
    points : np.ndarray
        ``(n, 2)`` points.
    out_path : str
        File to write.

    Returns
    -------
    str
        *out_path*.
    """

    np.savetxt(out_path, np.asarray(points, dtype=float).reshape(-1, 2), fmt="%.8f")

    return out_path


def write_swan_wind(path: str, wind: xr.Dataset) -> str:
    """
    Write a regular wind field for ``INPGRID WIND REGULAR`` / ``READINP WIND``.

    The file holds the ``u10`` block then the ``v10`` block, rows from south
    to north (``READINP WIND 1 'file' 3 0 FREE``). The returned grid string
    goes after ``INPGRID WIND REGULAR``.

    Parameters
    ----------
    path : str
        File to write.
    wind : xr.Dataset
        ``u10`` and ``v10`` (m/s) on ``(latitude, longitude)``, evenly spaced
        (see :func:`bluemath_tk.waves.wind.wind_field_at`).

    Returns
    -------
    str
        ``"xpinp ypinp alpinp mxinp myinp dxinp dyinp"``.
    """

    wind = wind.sortby("latitude").sortby("longitude")
    lon, lat = wind["longitude"].values, wind["latitude"].values
    with open(path, "w") as f:
        for name in ("u10", "v10"):
            values = wind[name].transpose("latitude", "longitude").values
            np.savetxt(f, np.nan_to_num(values.astype(float)), fmt="%.3f")
    dx = float(np.diff(lon).mean()) if lon.size > 1 else 1.0
    dy = float(np.diff(lat).mean()) if lat.size > 1 else 1.0

    return f"{lon[0]:.6f} {lat[0]:.6f} 0 {lon.size - 1} {lat.size - 1} {dx:.6f} {dy:.6f}"


def read_swan_table(path: str, quantities: list[str]) -> pd.DataFrame:
    """
    Read a ``TABLE ... NOHEAD`` output, one row per output point.

    SWAN writes exception values (e.g. -9 or -999) at dry or out-of-grid
    points; every quantity read here is non-negative, so negative values
    become NaN.

    Parameters
    ----------
    path : str
        Table file.
    quantities : list of str
        SWAN quantities in the order of the ``TABLE`` command (e.g.
        ``["HSIGN", "TPS", "DIR", "DSPR"]``).

    Returns
    -------
    pd.DataFrame
        One column per quantity, renamed with :data:`TABLE_VARS` when known.
    """

    table = pd.read_csv(path, sep=r"\s+", header=None, comment="%")
    table.columns = [TABLE_VARS.get(q, q) for q in quantities]

    return table.mask(table < 0)
