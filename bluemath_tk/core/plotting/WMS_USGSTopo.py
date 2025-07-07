import math
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def deg2num(lat: float, lon: float, zoom: int) -> tuple[int, int]:
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


def num2deg(xtile: int, ytile: int, zoom: int) -> tuple[float, float]:
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


def lonlat_to_webmercator(lon: float, lat: float) -> tuple[float, float]:
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
    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan((math.pi / 4) + math.radians(lat) / 2))
    return x, y


def tile_bounds_meters(
    x_start: int, y_start: int, x_end: int, y_end: int, zoom: int
) -> tuple[float, float, float, float]:
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
    WORLD_MAP_WIDTH = 2 * math.pi * 6378137
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
    verbose: bool = True,
    mask_ocean: bool = False,
    add_features: bool = True,
    display_width_px: int = 1024,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Downloads and displays a USGS raster map for the given bounding box.

    Parameters
    ----------
    lat_min : float
        Minimum latitude of the region.
    lat_max : float
        Maximum latitude of the region.
    lon_min : float
        Minimum longitude of the region.
    lon_max : float
        Maximum longitude of the region.
    display_width_px : int, optional
        Approximate pixel width for display (default is 1024).
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
    tile_url = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"

    for x in range(x_start, x_end + 1):
        for y in range(y_start, y_end + 1):
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

    crs_tiles = ccrs.Mercator.GOOGLE
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=crs_tiles)
    ax.set_extent([xmin, xmax, ymin, ymax], crs=crs_tiles)

    ax.imshow(
        map_img, origin="upper", extent=[xmin, xmax, ymin, ymax], transform=crs_tiles
    )
    scale = get_cartopy_scale(zoom)
    if verbose:
        print(f"Using Cartopy scale: {scale}")

    if add_features:
        ax.add_feature(cfeature.BORDERS.with_scale(scale), linewidth=0.8)
        ax.add_feature(cfeature.COASTLINE.with_scale(scale))
        ax.add_feature(cfeature.STATES.with_scale(scale), linewidth=0.5)

    if mask_ocean:
        ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor="w", zorder=3)
    return fig, ax