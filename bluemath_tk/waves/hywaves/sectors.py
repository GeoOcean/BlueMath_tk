"""
HyWaves goals: where the offshore forcing enters the nearshore domain.

Goals are points in a buffer ring just inside the area of interest (AoI),
each with an open-sea propagation sector and its nearest node in every
offshore hindcast catalog:

1. :func:`prepare_case_hindcasts` / :func:`load_catalogs` / :func:`load_land`
2. :func:`build_ring` -> :func:`random_targets_in_ring` ->
   :func:`sort_targets_along_aoi` (``goal_id`` 1..N along the AoI contour)
3. :class:`AoIRingSectors` -> :func:`build_goals_dict` (the goals mapping)
4. :func:`boundary_snaps` / :func:`goals_layers` / :func:`save_goals_gpkg`
   for maps and GIS.

Sectors are **nautical** wave-from directions, clockwise from North, stored
as ``[left, right]`` with ``left < right`` (``left`` negative when crossing
North); see :func:`bluemath_tk.core.operations.in_nautical_sector`.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.prepared import prep

from ...core.operations import (
    math_sector_to_nautical_clockwise,
    nautical_sector_wedge_coords,
)

GEO_CRS = "EPSG:4326"


def _read_catalog(path: Path) -> gpd.GeoDataFrame:
    """Read a coords GPKG and ensure a ``fid`` attribute column exists."""

    try:
        import pyogrio

        gdf = pyogrio.read_dataframe(path, fid_as_index=True)
        gdf = gdf.reset_index()  # OGR fid → column
    except Exception:
        gdf = gpd.read_file(path)
        if "fid" not in gdf.columns:
            if gdf.index.name in ("fid", "id"):
                gdf = gdf.reset_index()
            else:
                gdf = gdf.copy()
                gdf["fid"] = range(len(gdf))

    return gpd.GeoDataFrame(gdf, geometry="geometry", crs=GEO_CRS)


def load_catalogs(hindcasts_dir: Path | str) -> dict[str, gpd.GeoDataFrame]:
    """
    Load every ``*_coords.gpkg`` in ``hindcasts_dir`` (stem → catalog name).

    Parameters
    ----------
    hindcasts_dir
        Directory containing one ``<name>_coords.gpkg`` per hindcast catalog.

    Returns
    -------
    dict[str, geopandas.GeoDataFrame]
        Catalog name → node coordinates GeoDataFrame.

    Raises
    ------
    FileNotFoundError
        If no ``*_coords.gpkg`` files are found in *hindcasts_dir*.
    """

    root = Path(hindcasts_dir)
    catalogs: dict[str, gpd.GeoDataFrame] = {}
    for path in sorted(root.glob("*_coords.gpkg")):
        name = path.stem.removesuffix("_coords")
        catalogs[name] = _read_catalog(path)
    if not catalogs:
        raise FileNotFoundError(f"No *_coords.gpkg files in {root}")
    return catalogs


def load_land(land_path: Path | str) -> gpd.GeoDataFrame:
    """
    Load a land polygon file in geographic CRS.

    Parameters
    ----------
    land_path
        Path to a land vector dataset.

    Returns
    -------
    geopandas.GeoDataFrame
        Land geometries in ``EPSG:4326``.
    """
    return gpd.read_file(land_path).to_crs(GEO_CRS)


def prepare_case_hindcasts(
    aoi: gpd.GeoDataFrame,
    out_dir: Path | str,
    world_hindcasts_dir: Path | str,
    world_land_path: Path | str,
    buffer_km: float = 150.0,
    overwrite: bool = False,
) -> Path:
    """
    Crop the world-scale hindcast catalogs and land polygons down to the AoI.

    Reads only the features inside the AoI's bounding box (padded by
    ``buffer_km``) from the world master files via a bbox-filtered
    ``geopandas.read_file`` -- backed by each format's spatial index -- so
    global files (millions of points, a full-resolution world coastline)
    never get loaded in full. The cropped subset is written to *out_dir* in
    the layout :func:`load_catalogs` and :func:`load_land` expect
    (``<name>_coords.gpkg`` per hindcast, ``land.gpkg``).

    Parameters
    ----------
    aoi
        Area-of-interest polygon(s), any CRS.
    out_dir
        Per-case output directory. If it already exists, this is a no-op
        (delete it, or pass ``overwrite=True``, to rebuild -- e.g. after
        moving the AoI).
    world_hindcasts_dir
        Directory of world-scale ``<name>_coords.gpkg`` hindcast catalogs.
    world_land_path
        Path to the world-scale land polygon dataset.
    buffer_km
        Padding around the AoI bounding box, in kilometres. Should cover the
        largest ``max_hindcast_distance_m`` you plan to snap goals within.
    overwrite
        Rebuild *out_dir* even if it already exists.

    Returns
    -------
    pathlib.Path
        *out_dir*.
    """

    out_dir = Path(out_dir)
    if out_dir.exists() and not overwrite:
        return out_dir

    world_hindcasts_dir = Path(world_hindcasts_dir)
    world_land_path = Path(world_land_path)

    minx, miny, maxx, maxy = aoi.to_crs(GEO_CRS).total_bounds
    mean_lat = (miny + maxy) / 2.0
    lat_pad = buffer_km / 111.0
    lon_pad = buffer_km / (111.0 * max(0.15, math.cos(math.radians(mean_lat))))
    bbox = (minx - lon_pad, miny - lat_pad, maxx + lon_pad, maxy + lat_pad)

    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    n_catalogs = 0
    for path in sorted(world_hindcasts_dir.glob("*_coords.gpkg")):
        gdf = gpd.read_file(path, bbox=bbox)
        if gdf.empty:
            shutil.rmtree(tmp_dir)
            raise ValueError(
                f"No {path.stem} nodes within {buffer_km} km of the AoI -- "
                "check the AoI or increase buffer_km"
            )
        gdf.to_file(tmp_dir / path.name, driver="GPKG")
        n_catalogs += 1
    if n_catalogs == 0:
        shutil.rmtree(tmp_dir)
        raise FileNotFoundError(f"No *_coords.gpkg files in {world_hindcasts_dir}")

    land = gpd.read_file(world_land_path, bbox=bbox)
    land.to_file(tmp_dir / "land.gpkg", driver="GPKG")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)
    return out_dir


def build_ring(
    aoi: gpd.GeoDataFrame,
    land: gpd.GeoDataFrame,
    buffer_distance_m: float,
    work_crs: str,
) -> tuple[Polygon, gpd.GeoDataFrame]:
    """
    Inland buffer ring: ``(AoI − AoI.buffer(-d)) − land`` in projected metres.

    Parameters
    ----------
    aoi
        Area-of-interest polygon(s), geographic CRS.
    land
        Land polygon(s), geographic CRS.
    buffer_distance_m
        Inward buffer distance in metres (ring width).
    work_crs
        Projected CRS used for the metric buffer operation.

    Returns
    -------
    tuple[shapely.Polygon, geopandas.GeoDataFrame]
        Ring geometry in geographic CRS, and the same geometry wrapped in a
        one-row GeoDataFrame.
    """

    aoi_xy = aoi.to_crs(work_crs)
    land_xy = land.to_crs(work_crs)
    aoi_union = aoi_xy.union_all()
    land_union = land_xy.union_all()
    ring = aoi_union.difference(aoi_union.buffer(-buffer_distance_m)).difference(
        land_union
    )
    ring_gdf = gpd.GeoDataFrame(geometry=[ring], crs=work_crs).to_crs(GEO_CRS)
    return ring_gdf.geometry.iloc[0], ring_gdf


def _aoi_exterior_ring(aoi: gpd.GeoDataFrame, work_crs: str) -> Any:
    """Exterior ring of the (largest, if multi-) AoI polygon, in *work_crs*."""
    aoi_xy = aoi.to_crs(work_crs)
    aoi_poly = aoi_xy.geometry.iloc[0]
    if aoi_poly.geom_type == "MultiPolygon":
        aoi_poly = max(aoi_poly.geoms, key=lambda g: g.area)
    return aoi_poly.exterior


def sort_targets_along_aoi(
    targets: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
    *,
    work_crs: str,
    clockwise: bool | None = True,
    start_at: str = "west",
) -> gpd.GeoDataFrame:
    """Renumber ``goal_id`` by position along the AoI exterior contour.

    Each target is projected onto the AoI boundary; ordering follows increasing
    arc-length from ``start_at`` along the ring.

    Parameters
    ----------
    targets
        Target points to renumber (geographic CRS).
    aoi
        Area-of-interest polygon whose exterior contour defines the ordering.
    work_crs
        Projected CRS used for arc-length projection.
    clockwise
        ``True`` (default) — clockwise on the map; ``False`` — counter-clockwise;
        ``None`` — follow the shapefile vertex order.
    start_at
        Where goal ``1`` begins on the contour: ``"west"``, ``"north"``, or
        ``"ring_origin"`` (first shapefile vertex).

    Returns
    -------
    geopandas.GeoDataFrame
        *targets* with ``goal_id`` renumbered ``1..N`` in contour order.

    Raises
    ------
    ValueError
        If *start_at* is not one of ``"west"``, ``"north"``, ``"ring_origin"``.
    """

    if start_at not in ("west", "north", "ring_origin"):
        raise ValueError(
            f"start_at must be west, north, or ring_origin — got {start_at!r}"
        )

    ring = _aoi_exterior_ring(aoi, work_crs)
    targets_xy = targets.to_crs(work_crs)
    arclen = targets_xy.geometry.apply(ring.project)

    if start_at == "ring_origin":
        d0 = 0.0
    else:
        coords = np.array(ring.coords)
        idx = int(
            np.argmin(coords[:, 0]) if start_at == "west" else np.argmax(coords[:, 1])
        )
        d0 = float(ring.project(Point(coords[idx])))

    sort_key = (arclen - d0) % ring.length
    ascending = True if clockwise is None else clockwise != ring.is_ccw

    gdf = targets.copy()
    gdf["_sort_key"] = sort_key.values
    gdf = gdf.sort_values("_sort_key", ascending=ascending, kind="stable")
    gdf["goal_id"] = range(1, len(gdf) + 1)
    return gdf.drop(columns="_sort_key").reset_index(drop=True)


def random_targets_in_ring(
    ring: Polygon,
    land: gpd.GeoDataFrame,
    *,
    min_spacing_m: float,
    work_crs: str,
    random_seed: int = 42,
    batch_size: int = 500,
    max_batches: int = 200,
) -> gpd.GeoDataFrame:
    """Greedy random targets inside ``ring``, not on land, spaced by ``min_spacing_m``.

    Call :func:`sort_targets_along_aoi` afterwards to assign ``goal_id`` along the
    AoI contour.

    Parameters
    ----------
    ring
        Candidate area, geographic CRS (see :func:`build_ring`).
    land
        Land polygon(s) to exclude, geographic CRS.
    min_spacing_m
        Minimum distance between accepted targets, in metres.
    work_crs
        Projected CRS used for sampling and spacing checks.
    random_seed
        Seed for the random number generator.
    batch_size
        Candidate points drawn per batch.
    max_batches
        Maximum number of batches before giving up.

    Returns
    -------
    geopandas.GeoDataFrame
        Accepted target points, geographic CRS.

    Raises
    ------
    RuntimeError
        If no target could be placed after *max_batches* batches.
    """

    ring_xy = gpd.GeoDataFrame(geometry=[ring], crs=GEO_CRS).to_crs(work_crs)
    ring_geom = ring_xy.geometry.iloc[0]
    land_union = land.to_crs(work_crs).union_all()
    # Prepared geometries index their boundary once, turning each .contains()
    # call below into an O(log n) query instead of an O(n) edge scan -- with
    # up to batch_size * max_batches candidates tested against a full-
    # resolution coastline (easily 1M+ vertices), unprepared contains() calls
    # make this loop take hours instead of seconds.
    ring_geom_p = prep(ring_geom)
    land_union_p = prep(land_union)

    minx, miny, maxx, maxy = ring_geom.bounds
    rng = np.random.default_rng(random_seed)
    min_spacing_m = float(min_spacing_m)

    chosen_xy: list[tuple[float, float]] = []
    for _ in range(max_batches):
        xs = rng.uniform(minx, maxx, batch_size)
        ys = rng.uniform(miny, maxy, batch_size)
        for x, y in zip(xs, ys):
            p = Point(x, y)
            if not ring_geom_p.contains(p) or land_union_p.contains(p):
                continue
            if not chosen_xy:
                chosen_xy.append((x, y))
                continue
            dx = np.array([c[0] for c in chosen_xy]) - x
            dy = np.array([c[1] for c in chosen_xy]) - y
            if float((dx * dx + dy * dy).min()) >= min_spacing_m**2:
                chosen_xy.append((x, y))

    if not chosen_xy:
        raise RuntimeError("No targets placed in ring — try smaller spacing or buffer.")

    gdf = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in chosen_xy],
        crs=work_crs,
    ).to_crs(GEO_CRS)
    return gdf


def _row_fid(row) -> int:
    """Extract the integer feature id from a nearest-join result row."""
    for key in ("fid", "id", "fid_right", "id_right"):
        if key in row.index and row[key] is not None and row[key] == row[key]:
            return int(row[key])
    if "index_right" in row.index:
        return int(row["index_right"])
    raise KeyError("No fid/id column in nearest-neighbour join result")


def _nearest_per_catalog(
    points: gpd.GeoDataFrame,
    catalogs: dict[str, gpd.GeoDataFrame],
    *,
    work_crs: str,
) -> dict[Any, dict[str, dict[str, Any]]]:
    """Nearest hindcast node per catalog, for every point in one join per catalog.

    Each catalog is reprojected once regardless of how many points are
    queried — :func:`snap_to_catalogs` wraps this for a single point, but
    calling it once per point (as looping callers used to) reprojects the
    same catalog once per point too.
    """
    pts_xy = points.to_crs(work_crs)
    out: dict[Any, dict[str, dict[str, Any]]] = {idx: {} for idx in points.index}
    for name, catalog in catalogs.items():
        cat_xy = catalog.to_crs(work_crs)
        joined = gpd.sjoin_nearest(pts_xy, cat_xy, how="left", distance_col="dist")
        joined = joined[~joined.index.duplicated(keep="first")]
        for idx, row in joined.iterrows():
            out[idx][name] = {
                "fid": _row_fid(row),
                "longitude": float(row["longitude"]),
                "latitude": float(row["latitude"]),
                "distance_m": round(float(row["dist"]), 1),
            }
    return out


def snap_to_catalogs(
    target: Point | gpd.GeoSeries,
    catalogs: dict[str, gpd.GeoDataFrame],
    *,
    work_crs: str,
) -> dict[str, dict[str, Any]]:
    """
    Nearest hindcast node per catalog for one target point.

    Parameters
    ----------
    target
        Target point, geographic CRS.
    catalogs
        Catalog name → node coordinates GeoDataFrame (see :func:`load_catalogs`).
    work_crs
        Projected CRS used for the nearest-neighbour join.

    Returns
    -------
    dict[str, dict]
        Catalog name → ``{fid, longitude, latitude, distance_m}``.
    """

    pt = gpd.GeoDataFrame(geometry=[target], crs=GEO_CRS)
    return _nearest_per_catalog(pt, catalogs, work_crs=work_crs)[0]


@dataclass
class AoIRingSectors:
    """Pre-computed AoI exterior ring geometry for adaptive open-sea sectors.

    Attributes
    ----------
    aoi_ring
        Exterior ring of the AoI polygon, in *work_crs* (see
        :func:`_aoi_exterior_ring`).
    n_segs
        Number of ring segments (vertices, excluding the closing duplicate).
    seg_cumlen
        Cumulative arc length at each segment start, length ``n_segs + 1``.
    seg_tau_deg
        Segment tangent direction, math degrees (CCW from East), one per
        segment.
    turn_deg
        Exterior turn angle at each vertex, degrees, sign-corrected for ring
        winding direction.
    outward_side_deg
        Offset (``+90`` or ``-90``) from segment tangent to the outward
        normal, depending on ring winding direction.
    """

    aoi_ring: Any
    n_segs: int
    seg_cumlen: np.ndarray
    seg_tau_deg: np.ndarray
    turn_deg: np.ndarray
    outward_side_deg: float

    @classmethod
    def from_aoi(cls, aoi: gpd.GeoDataFrame, work_crs: str) -> AoIRingSectors:
        """
        Build ring segment geometry (tangents, turn angles) from an AoI polygon.

        Parameters
        ----------
        aoi
            Area-of-interest polygon, geographic CRS.
        work_crs
            Projected CRS used for the ring geometry computation.

        Returns
        -------
        AoIRingSectors
            Pre-computed ring geometry for :meth:`sector_at_boundary`.
        """
        aoi_ring = _aoi_exterior_ring(aoi, work_crs)
        is_ccw = aoi_ring.is_ccw

        ring_coords = list(aoi_ring.coords)[:-1]
        n_segs = len(ring_coords)

        seg_lens = np.array(
            [
                np.hypot(
                    ring_coords[(i + 1) % n_segs][0] - ring_coords[i][0],
                    ring_coords[(i + 1) % n_segs][1] - ring_coords[i][1],
                )
                for i in range(n_segs)
            ]
        )
        seg_cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
        seg_tau_deg = np.array(
            [
                float(
                    np.degrees(
                        np.arctan2(
                            ring_coords[(i + 1) % n_segs][1] - ring_coords[i][1],
                            ring_coords[(i + 1) % n_segs][0] - ring_coords[i][0],
                        )
                    )
                )
                for i in range(n_segs)
            ]
        )
        turn_deg = np.array(
            [
                ((seg_tau_deg[i] - seg_tau_deg[(i - 1) % n_segs] + 180.0) % 360.0)
                - 180.0
                for i in range(n_segs)
            ]
        )
        if not is_ccw:
            turn_deg = -turn_deg
        outward_side_deg = -90.0 if is_ccw else 90.0

        return cls(
            aoi_ring=aoi_ring,
            n_segs=n_segs,
            seg_cumlen=seg_cumlen,
            seg_tau_deg=seg_tau_deg,
            turn_deg=turn_deg,
            outward_side_deg=outward_side_deg,
        )

    def _segment_index(self, d: float) -> int:
        """Ring segment index containing arc-length position *d*."""
        i = int(np.searchsorted(self.seg_cumlen, d, side="right") - 1)
        return max(0, min(self.n_segs - 1, i))

    def sector_at_boundary(
        self, point_xy: Point, margin_deg: float = 5.0
    ) -> dict[str, Any]:
        """Adaptive open-sea sector aligned with the local AoI boundary segment.

        Projects ``point_xy`` onto the exterior ring, snaps to that boundary point,
        and builds the sector from the segment tangent / outward normal there
        (NC ``00_Compute_Goal_Directions`` geometry).

        Parameters
        ----------
        point_xy
            Query point in the same CRS as :attr:`aoi_ring` (typically
            *work_crs*).
        margin_deg
            Extra half-width added on each side of the boundary-normal sector,
            in degrees.

        Returns
        -------
        dict
            ``angles`` — nautical clockwise ``[left, right]`` (see
            module docstring); ``left < right``, negative ``left`` when
            the sector crosses North.
            ``boundary_snap_xy`` — snapped point in the same CRS as *point_xy*.
        """

        d = self.aoi_ring.project(point_xy)
        boundary_snap_xy = self.aoi_ring.interpolate(d)
        seg_idx = self._segment_index(d)
        v_a = seg_idx
        v_b = (seg_idx + 1) % self.n_segs

        center_deg = self.seg_tau_deg[seg_idx] + self.outward_side_deg
        opening_deg = 180.0 + 0.5 * (self.turn_deg[v_a] + self.turn_deg[v_b])
        half_w = opening_deg / 2.0
        low_math = center_deg - half_w - margin_deg
        high_math = center_deg + half_w + margin_deg
        return {
            "angles": math_sector_to_nautical_clockwise(low_math, high_math),
            "boundary_snap_xy": boundary_snap_xy,
        }


def boundary_snaps(
    targets: gpd.GeoDataFrame,
    sectors: AoIRingSectors,
    *,
    work_crs: str,
) -> dict[str, dict[str, float]]:
    """AoI boundary snap per goal (for plotting / GPKG only, not stored in goals JSON).

    Parameters
    ----------
    targets
        Goal target points with a ``goal_id`` column, geographic CRS.
    sectors
        Pre-computed ring geometry (see :meth:`AoIRingSectors.from_aoi`).
    work_crs
        Projected CRS used for the boundary projection.

    Returns
    -------
    dict[str, dict[str, float]]
        ``goal_id`` (as string) → ``{longitude, latitude}`` of the snapped
        boundary point.
    """

    targets_xy = targets.to_crs(work_crs)
    out: dict[str, dict[str, float]] = {}
    for _, row in targets.iterrows():
        key = str(int(row["goal_id"]))
        snap_xy = sectors.sector_at_boundary(targets_xy.loc[row.name].geometry)[
            "boundary_snap_xy"
        ]
        snap_geo = gpd.GeoSeries([snap_xy], crs=work_crs).to_crs(GEO_CRS).iloc[0]
        out[key] = {
            "longitude": round(float(snap_geo.x), 6),
            "latitude": round(float(snap_geo.y), 6),
        }
    return out


def build_goals_dict(
    targets: gpd.GeoDataFrame,
    catalogs: dict[str, gpd.GeoDataFrame],
    sectors: AoIRingSectors,
    *,
    work_crs: str,
    margin_deg: float = 5.0,
) -> dict[str, dict[str, Any]]:
    """Build the ``goals`` object for the case goals JSON.

    Each goal includes ``angles`` as nautical clockwise ``[left, right]``
    (nautical, see module docstring). Boundary snap coordinates are not stored
    in the JSON (use :func:`boundary_snaps` for maps).

    Parameters
    ----------
    targets
        Goal target points with a ``goal_id`` column, geographic CRS.
    catalogs
        Catalog name → node coordinates GeoDataFrame (see :func:`load_catalogs`).
    sectors
        Pre-computed ring geometry (see :meth:`AoIRingSectors.from_aoi`).
    work_crs
        Projected CRS used for sector and nearest-neighbour computations.
    margin_deg
        Extra half-width added on each side of the boundary-normal sector,
        in degrees.

    Returns
    -------
    dict[str, dict]
        ``goal_id`` (as string) → ``{target, angles, hindcast}``.
    """

    targets_xy = targets.to_crs(work_crs)
    hindcasts = _nearest_per_catalog(targets, catalogs, work_crs=work_crs)
    goals: dict[str, dict[str, Any]] = {}

    for _, row in targets.iterrows():
        goal_key = str(int(row["goal_id"]))
        pt_xy = targets_xy.loc[row.name].geometry
        pt_geo = row.geometry

        goals[goal_key] = {
            "target": {
                "longitude": round(float(pt_geo.x), 6),
                "latitude": round(float(pt_geo.y), 6),
            },
            "angles": sectors.sector_at_boundary(pt_xy, margin_deg=margin_deg)[
                "angles"
            ],
            "hindcast": hindcasts[row.name],
        }
    return goals


def _sector_wedge(
    lon: float,
    lat: float,
    left: float,
    right: float,
    radius_deg: float,
) -> Polygon:
    """Build a sector polygon from nautical ``[left, right]`` (GPKG / QGIS)."""
    return Polygon(nautical_sector_wedge_coords(lon, lat, left, right, radius_deg))


def hindcast_snaps_gdf(goals: dict[str, dict[str, Any]]) -> gpd.GeoDataFrame:
    """
    Flat GeoDataFrame of all hindcast snap points from a goals dict.

    Parameters
    ----------
    goals
        Goals mapping from :func:`build_goals_dict`.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per ``(goal_id, catalog)`` pair, with ``fid``, ``longitude``,
        ``latitude``, ``distance_m`` columns; geographic CRS.
    """

    rows: list[dict[str, Any]] = []
    for goal_key in sorted(goals, key=int):
        for catalog, h in goals[goal_key]["hindcast"].items():
            rows.append(
                {
                    "goal_id": int(goal_key),
                    "catalog": catalog,
                    "fid": h["fid"],
                    "longitude": h["longitude"],
                    "latitude": h["latitude"],
                    "distance_m": h["distance_m"],
                }
            )
    return gpd.GeoDataFrame(
        rows,
        geometry=gpd.points_from_xy(
            [r["longitude"] for r in rows],
            [r["latitude"] for r in rows],
        ),
        crs=GEO_CRS,
    )


def goals_layers(
    goals: dict[str, dict[str, Any]],
    targets: gpd.GeoDataFrame,
    boundary_snaps: dict[str, dict[str, float]],
    *,
    sector_radius_deg: float = 0.35,
) -> dict[str, gpd.GeoDataFrame]:
    """Build GeoDataFrames for QGIS export (targets, sectors, links).

    ``angle_low`` / ``angle_high`` match the nautical ``angles`` stored in the
    goals JSON (left / right clockwise boundaries).

    Parameters
    ----------
    goals
        Goals mapping from :func:`build_goals_dict`.
    targets
        Goal target points with a ``goal_id`` column, geographic CRS.
    boundary_snaps
        ``goal_id`` (as string) → ``{longitude, latitude}``, from
        :func:`boundary_snaps`.
    sector_radius_deg
        Sector wedge radius, in degrees, for the ``sectors`` layer.

    Returns
    -------
    dict[str, geopandas.GeoDataFrame]
        ``{"targets", "boundary_snaps", "sectors", "links"}`` layers, all in
        geographic CRS.
    """

    target_rows: list[dict[str, Any]] = []
    snap_rows: list[dict[str, Any]] = []
    sector_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []

    for _, row in targets.sort_values("goal_id").iterrows():
        key = str(int(row["goal_id"]))
        goal = goals[key]
        t = goal["target"]
        snap = boundary_snaps[key]
        low, high = goal["angles"]

        target_rows.append(
            {
                "goal_id": int(key),
                "angle_low": low,
                "angle_high": high,
                "geometry": Point(t["longitude"], t["latitude"]),
            }
        )
        snap_rows.append(
            {
                "goal_id": int(key),
                "angle_low": low,
                "angle_high": high,
                "geometry": Point(snap["longitude"], snap["latitude"]),
            }
        )
        sector_rows.append(
            {
                "goal_id": int(key),
                "angle_low": low,
                "angle_high": high,
                "geometry": _sector_wedge(
                    snap["longitude"], snap["latitude"], low, high, sector_radius_deg
                ),
            }
        )
        link_rows.append(
            {
                "goal_id": int(key),
                "geometry": LineString(
                    [
                        (t["longitude"], t["latitude"]),
                        (snap["longitude"], snap["latitude"]),
                    ]
                ),
            }
        )

    return {
        "targets": gpd.GeoDataFrame(target_rows, crs=GEO_CRS),
        "boundary_snaps": gpd.GeoDataFrame(snap_rows, crs=GEO_CRS),
        "sectors": gpd.GeoDataFrame(sector_rows, crs=GEO_CRS),
        "links": gpd.GeoDataFrame(link_rows, crs=GEO_CRS),
    }


def save_goals_gpkg(
    goals: dict[str, dict[str, Any]],
    targets: gpd.GeoDataFrame,
    boundary_snaps: dict[str, dict[str, float]],
    path: Path | str,
    *,
    ring: gpd.GeoDataFrame | None = None,
    sector_radius_deg: float = 0.35,
) -> Path:
    """
    Write all goal layers to one GeoPackage (one layer per hindcast catalog).

    Parameters
    ----------
    goals
        Goals mapping from :func:`build_goals_dict`.
    targets
        Goal target points with a ``goal_id`` column, geographic CRS.
    boundary_snaps
        ``goal_id`` (as string) → ``{longitude, latitude}``, from
        :func:`boundary_snaps`.
    path
        Output ``.gpkg`` path (overwritten if it exists).
    ring
        Optional buffer-ring GeoDataFrame, added as a ``buffer_ring`` layer.
    sector_radius_deg
        Sector wedge radius, in degrees, for the ``sectors`` layer.

    Returns
    -------
    Path
        Written GeoPackage path.
    """

    path = Path(path)
    layers = goals_layers(
        goals, targets, boundary_snaps, sector_radius_deg=sector_radius_deg
    )
    if ring is not None:
        layers["buffer_ring"] = ring.to_crs(GEO_CRS)

    snaps = hindcast_snaps_gdf(goals)
    for catalog in sorted(snaps["catalog"].unique()):
        layer = snaps[snaps["catalog"] == catalog].drop(columns="catalog")
        # GPKG reserves "fid" as the feature primary key. Catalog snap IDs can
        # legitimately repeat across goals, so keep them under a non-reserved
        # name to avoid UNIQUE constraint errors on export.
        if "fid" in layer.columns:
            layer = layer.rename(columns={"fid": "node_fid"})
        layers[catalog] = layer

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    for i, (name, gdf) in enumerate(layers.items()):
        gdf.to_file(path, layer=name, driver="GPKG", mode="w" if i == 0 else "a")
    return path


__all__ = [
    "AoIRingSectors",
    "GEO_CRS",
    "boundary_snaps",
    "build_goals_dict",
    "build_ring",
    "goals_layers",
    "hindcast_snaps_gdf",
    "load_catalogs",
    "load_land",
    "random_targets_in_ring",
    "save_goals_gpkg",
    "snap_to_catalogs",
    "sort_targets_along_aoi",
]
