import math
from io import BytesIO
from typing import Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import requests
from PIL import Image

from ..constants import EARTH_RADIUS

EARTH_RADIUS_M = EARTH_RADIUS * 1000  # Convert km to m


def deg2num(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """
    Converts geographic coordinates to tile numbers for a given zoom level.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    zoom : int
        Zoom level.

    Returns
    -------
    xtile : int
        Tile number in x-direction.
    ytile : int
        Tile number in y-direction.
    """

    lat_rad = math.radians(lat)
    n = 2.0**zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int(
        (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    )

    return xtile, ytile


def num2deg(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
    """
    Converts tile numbers back to geographic coordinates.

    Parameters
    ----------
    xtile : int
        Tile number in x-direction.
    ytile : int
        Tile number in y-direction.
    zoom : int
        Zoom level.

    Returns
    -------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    """

    n = 2.0**zoom
    lon = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat = math.degrees(lat_rad)

    return lat, lon


def lonlat_to_webmercator(lon: float, lat: float) -> Tuple[float, float]:
    """
    Converts lon/lat to Web Mercator projection coordinates in meters.

    Parameters
    ----------
    lon : float
        Longitude in degrees.
    lat : float
        Latitude in degrees.

    Returns
    -------
    x : float
        X coordinate in meters.
    y : float
        Y coordinate in meters.
    """

    x = EARTH_RADIUS_M * math.radians(lon)
    y = EARTH_RADIUS_M * math.log(math.tan((math.pi / 4) + math.radians(lat) / 2))

    return x, y


def tile_bounds_meters(
    x_start: int, y_start: int, x_end: int, y_end: int, zoom: int
) -> Tuple[float, float, float, float]:
    """
    Computes the bounding box of the tile region in Web Mercator meters.

    Returns
    -------
    xmin, ymin, xmax, ymax : float
        Bounding box in meters (Web Mercator projection).
    """

    lat1, lon1 = num2deg(x_start, y_start, zoom)
    lat2, lon2 = num2deg(x_end + 1, y_end + 1, zoom)
    x1, y1 = lonlat_to_webmercator(lon1, lat2)
    x2, y2 = lonlat_to_webmercator(lon2, lat1)

    return x1, y1, x2, y2


def calculate_zoom(
    lon_min: float, lon_max: float, display_width_px: int = 1024, tile_size: int = 256
) -> int:
    """
    Automatically estimates an appropriate zoom level for the bounding box.

    Returns
    -------
    zoom : int
        Estimated zoom level.
    """

    WORLD_MAP_WIDTH = 2 * math.pi * EARTH_RADIUS_M
    x1, _ = lonlat_to_webmercator(lon_min, 0)
    x2, _ = lonlat_to_webmercator(lon_max, 0)
    region_width_m = abs(x2 - x1)
    meters_per_pixel_desired = region_width_m / display_width_px
    zoom = math.log2(WORLD_MAP_WIDTH / (tile_size * meters_per_pixel_desired))

    return int(round(zoom))


def get_cartopy_scale(zoom: int) -> str:
    """
    Select appropriate cartopy feature scale based on zoom level.

    Parameters
    ----------
    zoom : int
        Web Mercator zoom level.

    Returns
    -------
    scale : str
        One of '110m', '50m', or '10m'.
    """

    if zoom >= 9:
        return "10m"
    elif zoom >= 6:
        return "50m"
    else:
        return "110m"


def plot_usgs_raster_map(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    zoom: int = None,
    source: str = "arcgis",
    verbose: bool = True,
    mask_ocean: bool = False,
    add_features: bool = True,
    display_width_px: int = 1024,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Downloads and displays a USGS raster map for the given bounding box.

    Parameters
    ----------
    lat_min : float
        Minimum latitude of the bounding box.
    lat_max : float
        Maximum latitude of the bounding box.
    lon_min : float
        Minimum longitude of the bounding box.
    lon_max : float
        Maximum longitude of the bounding box.
    zoom : int, optional
        Zoom level (default is None, which auto-calculates based on bounding box).
    source : str, optional
        Source of the map tiles (default is "arcgis"). Options include:
        - "arcgis" (Esri World Imagery)
        - "google" (Google Maps Satellite)
        - "eox" (EOX Sentinel-2 Cloudless)
        - "osm" (OpenStreetMap Standard Tiles)
        - "amazon" (Amazon Terrarium Elevation Tiles)
        - "esri_world" (Esri World Imagery)
        - "geoportail_fr" (Geoportail France Orthophotos)
        - "ign_spain_pnoa" (IGN Spain PNOA Orthophotos)
    verbose : bool, optional
        If True, prints additional information about the source and zoom level.
    mask_ocean : bool, optional
        If True, masks ocean areas in the map.
    add_features : bool, optional
        If True, adds borders, coastlines, and states to the map.
    display_width_px : int, optional
        Width of the display in pixels (default is 1024).

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure containing the map.
    ax : plt.Axes
        The matplotlib axes with the map projection.
    """

    tile_size = 256
    if zoom is None:
        zoom = calculate_zoom(lon_min, lon_max, display_width_px, tile_size)
    if verbose:
        print(f"Auto-selected zoom level: {zoom}")

    x_start, y_start = deg2num(lat_max, lon_min, zoom)
    x_end, y_end = deg2num(lat_min, lon_max, zoom)
    width = x_end - x_start + 1
    height = y_end - y_start + 1

    map_img = Image.new("RGB", (width * tile_size, height * tile_size))
    if source == "arcgis":
        tile_url = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        z_max = 19
        if verbose:
            print(
                "Using Esri World Imagery (ArcGIS):\n"
                "- Free for public and non-commercial use\n"
                "- Commercial use allowed under Esri terms of service\n"
                "- Attribution required: 'Tiles © Esri — Sources: Esri, Earthstar Geographics, CNES/Airbus DS, "
                "USDA, USGS, AeroGRID, IGN, and the GIS User Community'\n"
                "- Max zoom ~19"
            )

    elif source == "google":
        tile_url = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
        z_max = 21
        if verbose:
            print(
                "Using Google Maps Satellite:\n"
                "- NOT license-free\n"
                "- Usage outside official Google Maps SDKs/APIs is prohibited\n"
                "- Commercial use requires a paid license from Google\n"
                "- May be blocked without an API key\n"
                "- Max zoom ~21"
            )

    elif source == "eox":
        tile_url = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/{z}/{y}/{x}.jpg"
        z_max = 16
        if verbose:
            print(
                "Using EOX Sentinel-2 Cloudless:\n"
                "- Based on Copernicus Sentinel-2 data processed by EOX\n"
                "- Licensed under Creative Commons BY-NC-SA 4.0 (non-commercial use only)\n"
                "- Attribution required: 'Sentinel-2 cloudless – © EOX, based on modified Copernicus Sentinel data 2016–2024'\n"
                "- Versions from 2016–2017 are under CC BY 4.0 (commercial use allowed)\n"
                "- Max zoom ~16"
            )

    elif source == "osm":
        tile_url = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        z_max = 19
        if verbose:
            print(
                "Using OpenStreetMap Standard Tiles:\n"
                "- Free and open-source (ODbL license)\n"
                "- Commercial use allowed with attribution\n"
                "- Attribution required: '© OpenStreetMap contributors'\n"
                "- Max zoom ~19"
            )

    elif source == "amazon":
        tile_url = (
            "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
        )
        z_max = 15
        if verbose:
            print(
                "Using Amazon Terrarium Elevation Tiles:\n"
                "- Free for public use\n"
                "- Attribution recommended: 'Amazon Terrarium'\n"
                "- Max zoom ~15"
            )

    elif source == "esri_world":
        tile_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        z_max = 19
        if verbose:
            print(
                "Using Esri World Imagery:\n"
                "- High-resolution global imagery\n"
                "- Free with attribution under Esri terms\n"
                "- Max zoom ~19"
            )

    elif source == "geoportail_fr":
        tile_url = (
            "https://data.geopf.fr/wmts?"
            "REQUEST=GetTile&SERVICE=WMTS&VERSION=1.0.0"
            "&STYLE=normal&TILEMATRIXSET=PM&FORMAT=image/jpeg"
            "&LAYER=ORTHOIMAGERY.ORTHOPHOTOS&TILEMATRIX={z}"
            "&TILEROW={y}&TILECOL={x}"
        )
        z_max = 19
        if verbose:
            print(
                "Using Geoportail France (Orthophotos):\n"
                "- Aerial orthophotos from the French National Institute (IGN)\n"
                "- Free for public use under Etalab license\n"
                "- Attribution: Geoportail France / IGN\n"
                "- Max zoom ~19"
            )
    elif source == "ign_spain_pnoa":
        tile_url = "https://tms-pnoa-ma.idee.es/1.0.0/pnoa-ma/{z}/{x}/{y_inv}.jpeg"
        z_max = 19
        if verbose:
            print(
                "Using IGN Spain PNOA Orthophotos:\n"
                "- High-resolution aerial imagery (PNOA program)\n"
                "- Provided by IGN/CNIG (Government of Spain)\n"
                "- Free to use under Creative Commons BY 4.0 license\n"
                "- Attribution required: 'Ortofotografía PNOA – © IGN / CNIG (Gobierno de España) – CC BY 4.0'\n"
                "- Max zoom ~19"
            )

    if zoom > z_max:
        zoom = z_max

    for x in range(x_start, x_end + 1):
        for y in range(y_start, y_end + 1):
            if source == "ign_spain_pnoa":
                y_inv = (2**zoom - 1) - y
                url = tile_url.format(z=zoom, x=x, y_inv=y_inv)
            else:
                url = tile_url.format(z=zoom, x=x, y=y)
            try:
                response = requests.get(url, timeout=10)
                tile = Image.open(BytesIO(response.content))
                map_img.paste(
                    tile, ((x - x_start) * tile_size, (y - y_start) * tile_size)
                )
            except Exception as e:
                print(f"Error fetching tile {x},{y}: {e}")

    xmin, ymin, xmax, ymax = tile_bounds_meters(x_start, y_start, x_end, y_end, zoom)

    crs_tiles = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=crs_tiles)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=crs_tiles)
    ax.imshow(
        map_img,
        origin="upper",
        extent=[xmin, xmax, ymin, ymax],
        transform=ccrs.Mercator.GOOGLE,
    )
    scale = get_cartopy_scale(zoom)
    if verbose:
        print(f"Using Cartopy scale: {scale}")

    if add_features:
        ax.add_feature(cfeature.BORDERS.with_scale(scale), linewidth=0.8)
        ax.add_feature(cfeature.COASTLINE.with_scale(scale))
        ax.add_feature(cfeature.STATES.with_scale(scale), linewidth=0.5)

    if mask_ocean:
        ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor="w", zorder=1)

    return fig, ax


if __name__ == "__main__":
    lat_min, lat_max = 35.406, 54.372
    lon_min, lon_max = -13.148, 12.325
    fig, ax = plot_usgs_raster_map(
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        zoom=None,
        verbose=True,
        add_features=False,
        mask_ocean=True,
    )
    fig.show()
