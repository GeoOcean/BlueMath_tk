"""
SnapWave input/output helpers: enclosure, boundary and forcing files.

SnapWave case folders contain plain-text inputs next to ``snapwave.inp``:

- ``boundary.txt``: boundary node coordinates (one ``x y`` row per node).
- ``enclosure.txt``: vertices of the computational enclosure polygon.
- ``hs.txt``, ``tp.txt``, ``dir.txt``, ``spr.txt``, ``wl.txt``: forcing tables,
  first column time in seconds, then one column per boundary node.
- the observation-points file (``obsfile``), one ``x y`` row per point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon

#: SnapWave history (``his_file``) variables -> short names.
POINT_VARS: dict[str, str] = {
    "point_hm0": "hs",
    "point_tp": "tp",
    "point_wavdir": "dir",
    "point_dirspr": "spr",
}

#: Forcing variable -> file name inside a case folder.
FORCING_FILES: dict[str, str] = {
    "hs": "hs.txt",
    "tp": "tp.txt",
    "dir": "dir.txt",
    "spr": "spr.txt",
    "wl": "wl.txt",
}

#: Bulk variables expected by :func:`boundary_time_series`.
BOUNDARY_VARS: tuple[str, ...] = ("hs", "tp", "dir", "spr")


# ---------------------------------------------------------------------------
# Enclosure polygon
# ---------------------------------------------------------------------------


def _largest_polygon(geometry: Polygon | MultiPolygon) -> Polygon:
    """Return *geometry* itself, or the largest part of a MultiPolygon."""

    if isinstance(geometry, MultiPolygon):
        geometry = max(geometry.geoms, key=lambda g: g.area)
    if not isinstance(geometry, Polygon):
        raise TypeError(f"Expected Polygon/MultiPolygon, got {geometry.geom_type}")

    return geometry


def buffer_polygon(
    polygon: Polygon | MultiPolygon,
    buffer_m: float,
    geo_crs: str = "EPSG:4326",
) -> Polygon | MultiPolygon:
    """
    Buffer a lon/lat polygon by *buffer_m* metres in a local UTM projection.

    Parameters
    ----------
    polygon : Polygon or MultiPolygon
        Input geometry in *geo_crs*.
    buffer_m : float
        Outward buffer distance in metres. ``0`` returns *polygon* unchanged.
    geo_crs : str, optional
        CRS of input and output coordinates. Default is "EPSG:4326".

    Returns
    -------
    Polygon or MultiPolygon
        Buffered geometry in *geo_crs*.
    """

    if buffer_m <= 0:
        return polygon

    import geopandas as gpd

    gdf = gpd.GeoDataFrame(geometry=[polygon], crs=geo_crs)
    work_crs = gdf.estimate_utm_crs()
    if work_crs is None:
        raise ValueError("Could not estimate a UTM CRS for the polygon.")

    return gdf.to_crs(work_crs).buffer(buffer_m).to_crs(geo_crs).iloc[0]


def build_enclosure_polygon(
    polygon: Polygon | MultiPolygon,
    buffer_m: float = 0.0,
    geo_crs: str = "EPSG:4326",
) -> Polygon:
    """
    Build a simple SnapWave enclosure around an area of interest.

    Exterior ring only (holes dropped), buffered outwards, then reduced to its
    minimum rotated rectangle (4 vertices).

    Parameters
    ----------
    polygon : Polygon or MultiPolygon
        Area of interest; for a MultiPolygon the largest part is used.
    buffer_m : float, optional
        Outward buffer in metres. Default is 0.
    geo_crs : str, optional
        CRS of input and output coordinates. Default is "EPSG:4326".

    Returns
    -------
    Polygon
        The enclosure quadrilateral.
    """

    exterior = Polygon(_largest_polygon(polygon).exterior.coords)
    buffered = _largest_polygon(buffer_polygon(exterior, buffer_m, geo_crs=geo_crs))

    return buffered.minimum_rotated_rectangle


def write_polygon_vertices_to_txt(
    polygon: Polygon | MultiPolygon,
    out_path: str | Path,
    drop_closing_vertex: bool = True,
    fmt: str = "{x}, {y}",
) -> Path:
    """
    Write polygon exterior vertices to a SnapWave enclosure text file.

    Parameters
    ----------
    polygon : Polygon or MultiPolygon
        Enclosure geometry (every part's exterior is written).
    out_path : str or Path
        Output file path.
    drop_closing_vertex : bool, optional
        Omit the repeated closing vertex. Default is True.
    fmt : str, optional
        Per-vertex format with ``x`` and ``y`` placeholders.

    Returns
    -------
    Path
        The written file.
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts = polygon.geoms if isinstance(polygon, MultiPolygon) else [polygon]

    lines: list[str] = []
    for part in parts:
        ring = list(part.exterior.coords)
        if drop_closing_vertex and len(ring) > 1 and ring[0] == ring[-1]:
            ring = ring[:-1]
        lines.extend(fmt.format(x=x, y=y) for x, y in ring)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return out_path


def write_points_to_txt(points: np.ndarray, out_path: str | Path) -> Path:
    """
    Write ``(x, y)`` points (boundary nodes or observation points) to a file.

    Parameters
    ----------
    points : np.ndarray
        ``(n, 2)`` coordinates.
    out_path : str or Path
        Output file path.

    Returns
    -------
    Path
        The written file.
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, np.asarray(points, dtype=float).reshape(-1, 2), fmt="%.8f")

    return out_path


# ---------------------------------------------------------------------------
# Reading case folders
# ---------------------------------------------------------------------------


def parse_snapwave_inp(path: str | Path) -> dict[str, str]:
    """
    Parse a ``snapwave.inp`` file into a flat ``{key: value}`` dict of strings.

    Parameters
    ----------
    path : str or Path
        The ``snapwave.inp`` file.

    Returns
    -------
    dict[str, str]
        Keyword values (comments after ``#`` ignored).
    """

    entries: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            entries[key.strip()] = value.strip()

    return entries


def read_xy_table(path: str | Path) -> np.ndarray:
    """
    Read a whitespace- or comma-separated ``x y`` coordinate file.

    Parameters
    ----------
    path : str or Path
        Text file with at least two numeric columns.

    Returns
    -------
    np.ndarray
        ``(n, 2)`` coordinates.

    Raises
    ------
    ValueError
        If the file has no coordinate rows.
    """

    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            rows.append([float(parts[0]), float(parts[1])])
    if not rows:
        raise ValueError(f"No coordinates found in {path}")

    return np.asarray(rows, dtype=float)


def read_enclosure_polygon(case_dir: str | Path) -> Polygon:
    """
    Read ``enclosure.txt`` from a case folder.

    Parameters
    ----------
    case_dir : str or Path
        SnapWave case folder.

    Returns
    -------
    Polygon
        The enclosure.
    """

    return Polygon(read_xy_table(Path(case_dir) / "enclosure.txt"))


def read_boundary_nodes(case_dir: str | Path) -> np.ndarray:
    """
    Read ``boundary.txt`` from a case folder.

    Parameters
    ----------
    case_dir : str or Path
        SnapWave case folder.

    Returns
    -------
    np.ndarray
        ``(n_nodes, 2)`` boundary node coordinates.
    """

    return read_xy_table(Path(case_dir) / "boundary.txt")


def read_forcing_table(path: str | Path) -> pd.DataFrame:
    """
    Read a SnapWave forcing file (e.g. ``hs.txt``) as a time-indexed table.

    Parameters
    ----------
    path : str or Path
        Forcing file: time (s) in the first column, one column per node.

    Returns
    -------
    pd.DataFrame
        Index time in seconds, columns ``b0``, ``b1``, ... per boundary node.
    """

    data = np.atleast_2d(np.loadtxt(path, dtype=float))
    columns = [f"b{i}" for i in range(data.shape[1] - 1)]

    return pd.DataFrame(data[:, 1:], index=data[:, 0], columns=columns)


def read_his_file(
    case_dir: str | Path, his_file: str = "output_sites.nc"
) -> xr.Dataset:
    """
    Load a SnapWave history (observation points) NetCDF, renamed to short names.

    ``point_hm0``/``point_tp``/``point_wavdir``/``point_dirspr`` become
    ``hs``/``tp``/``dir``/``spr``; the ``runtime`` dimension and bookkeeping
    variables (``crs``, ``station_id``) are dropped.

    Parameters
    ----------
    case_dir : str or Path
        SnapWave case folder.
    his_file : str, optional
        History file name (``his_file`` in ``snapwave.inp``).

    Returns
    -------
    xr.Dataset
        Loaded dataset on ``(time, stations)``.
    """

    with xr.open_dataset(Path(case_dir) / his_file) as ds:
        ds = ds.drop_dims(["runtime"], errors="ignore")
        ds = ds.drop_vars(["crs", "station_id"], errors="ignore").load()

    return ds.rename({k: v for k, v in POINT_VARS.items() if k in ds.data_vars})


# ---------------------------------------------------------------------------
# Forcing
# ---------------------------------------------------------------------------


def boundary_time_series(
    forcing: xr.Dataset,
    node_dim: str = "node",
    key_suffix: str = "nodes",
    variables: tuple[str, ...] = BOUNDARY_VARS,
) -> dict[str, list]:
    """
    Turn a ``(time, node)`` boundary forcing dataset into per-case parameters.

    One SnapWave case is run per time step; this returns, for each variable,
    one list of per-node values per time step, plus ``tref`` (the SnapWave
    reference time ``YYYYmmdd HHMMSS``), ready for ``metamodel_parameters``
    of :class:`~bluemath_tk.wrappers.snapwave.SnapWaveModelWrapper`.

    Parameters
    ----------
    forcing : xr.Dataset
        Dataset with dims ``(time, node_dim)`` and the bulk *variables*
        (directions nautical, coming from). Nodes are sorted by their
        coordinate, which must follow the order of ``boundary.txt``.
    node_dim : str, optional
        Boundary-node dimension name. Default is "node".
    key_suffix : str, optional
        Output keys are ``f"{var}_{key_suffix}"``. Default is "nodes"
        (``hs_nodes``, ...).
    variables : tuple of str, optional
        Variables to export. Default is ``("hs", "tp", "dir", "spr")``.

    Returns
    -------
    dict[str, list]
        ``{f"{var}_{key_suffix}": [[v_node0, v_node1, ...], ...], "tref": [...]}``.

    Raises
    ------
    ValueError
        If *node_dim* or a variable is missing.
    """

    if node_dim not in forcing.dims:
        raise ValueError(
            f"forcing must have a {node_dim!r} dimension; got {dict(forcing.sizes)}"
        )
    missing = [v for v in variables if v not in forcing]
    if missing:
        raise ValueError(f"forcing is missing variables: {missing}")

    nodes = forcing[node_dim].values
    if nodes.dtype.kind in "OUS" and all(str(n).isdigit() for n in nodes):
        # numeric ids stored as strings: sort 1, 2, ..., 10, not "1", "10", "2"
        forcing = forcing.isel({node_dim: np.argsort([int(n) for n in nodes])})
    else:
        forcing = forcing.sortby(node_dim)
    forcing = forcing.transpose("time", node_dim, ...)
    out: dict[str, list] = {
        f"{var}_{key_suffix}": forcing[var].values.tolist() for var in variables
    }
    out["tref"] = pd.to_datetime(forcing.time.values).strftime("%Y%m%d %H%M%S").tolist()

    return out
