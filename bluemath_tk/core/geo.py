from math import pi
from typing import List, Tuple, Union

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.vectorized import contains

from .constants import EARTH_RADIUS_NM

# Constants
FLATTENING = 1 / 298.257223563
EPS = 0.00000000005
DEG2RAD = pi / 180.0
RAD2DEG = 180.0 / pi


# TODO: Check which functions are implemented in Pyproj library!


def convert_to_radians(*args: Union[float, np.ndarray]) -> tuple:
    """
    Convert degree inputs to radians.

    Parameters
    ----------
    *args : Union[float, np.ndarray]
        Variable number of inputs in degrees to convert to radians.
        Can be either scalar floats or numpy arrays.

    Returns
    -------
    tuple
        Tuple of input values converted to radians, preserving input types.

    Examples
    --------
    >>> convert_to_radians(90.0)
    (1.5707963267948966,)
    >>> convert_to_radians(90.0, 180.0)
    (1.5707963267948966, 3.141592653589793)
    """

    return tuple(np.radians(arg) for arg in args)


def geodesic_distance(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate great circle distance between two points on Earth.

    Parameters
    ----------
    lat1 : Union[float, np.ndarray]
        Latitude of first point(s) in degrees
    lon1 : Union[float, np.ndarray]
        Longitude of first point(s) in degrees
    lat2 : Union[float, np.ndarray]
        Latitude of second point(s) in degrees
    lon2 : Union[float, np.ndarray]
        Longitude of second point(s) in degrees

    Returns
    -------
    Union[float, np.ndarray]
        Great circle distance(s) in degrees

    Notes
    -----
    Uses the haversine formula to calculate great circle distance.
    The result is in degrees of arc on a sphere.

    Examples
    --------
    >>> geodesic_distance(0, 0, 0, 90)
    90.0
    >>> geodesic_distance([0, 45], [0, -90], [0, -45], [90, 90])
    array([90., 180.])
    """

    lon1, lat1, lon2, lat2 = convert_to_radians(lon1, lat1, lon2, lat2)

    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    a = np.clip(a, 0, 1)

    rng = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return np.degrees(rng)


def geo_distance_cartesian(
    y_matrix: Union[float, np.ndarray],
    x_matrix: Union[float, np.ndarray],
    y_point: Union[float, np.ndarray],
    x_point: Union[float, np.ndarray],
) -> np.ndarray:
    """
    Returns cartesian distance between y,x matrix and y,x point.
    Optimized using vectorized operations.

    Parameters
    ----------
    y_matrix : Union[float, np.ndarray]
        2D array of y-coordinates (latitude or y in Cartesian).
    x_matrix : Union[float, np.ndarray]
        2D array of x-coordinates (longitude or x in Cartesian).
    y_point : Union[float, np.ndarray]
        y-coordinate of the point (latitude or y in Cartesian).
    x_point : Union[float, np.ndarray]
        x-coordinate of the point (longitude or x in Cartesian).

    Returns
    -------
    np.ndarray
        Array of distances in the same units as x_matrix and y_matrix.
    """

    dist = np.sqrt((y_point - y_matrix) ** 2 + (x_point - x_matrix) ** 2)

    return dist


def geodesic_azimuth(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate azimuth between two points on Earth.

    Parameters
    ----------
    lat1 : Union[float, np.ndarray]
        Latitude of first point(s) in degrees
    lon1 : Union[float, np.ndarray]
        Longitude of first point(s) in degrees
    lat2 : Union[float, np.ndarray]
        Latitude of second point(s) in degrees
    lon2 : Union[float, np.ndarray]
        Longitude of second point(s) in degrees

    Returns
    -------
    Union[float, np.ndarray]
        Azimuth(s) in degrees from North

    Notes
    -----
    The azimuth is the angle between true north and the direction to the second point,
    measured clockwise from north. Special cases are handled for points at the poles.

    Examples
    --------
    >>> geodesic_azimuth(0, 0, 0, 90)
    90.0
    >>> geodesic_azimuth([0, 45], [0, -90], [0, -45], [90, 90])
    array([90., 90.])
    """

    lon1, lat1, lon2, lat2 = convert_to_radians(lon1, lat1, lon2, lat2)

    az = np.arctan2(
        np.cos(lat2) * np.sin(lon2 - lon1),
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1),
    )

    # Handle special cases at poles
    az = np.where(lat1 <= -pi / 2, 0, az)
    az = np.where(lat2 >= pi / 2, 0, az)
    az = np.where(lat2 <= -pi / 2, pi, az)
    az = np.where(lat1 >= pi / 2, pi, az)

    return np.degrees(az % (2 * pi))


def geodesic_distance_azimuth(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Calculate both great circle distance and azimuth between two points.

    Parameters
    ----------
    lat1 : Union[float, np.ndarray]
        Latitude of first point(s) in degrees
    lon1 : Union[float, np.ndarray]
        Longitude of first point(s) in degrees
    lat2 : Union[float, np.ndarray]
        Latitude of second point(s) in degrees
    lon2 : Union[float, np.ndarray]
        Longitude of second point(s) in degrees

    Returns
    -------
    Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]
        Tuple containing:
        - distance(s) : Great circle distance(s) in degrees
        - azimuth(s) : Azimuth(s) in degrees from North

    See Also
    --------
    geodesic_distance : Calculate only the great circle distance
    geodesic_azimuth : Calculate only the azimuth

    Examples
    --------
    >>> dist, az = geodesic_distance_azimuth(0, 0, 0, 90)
    >>> dist
    90.0
    >>> az
    90.0
    """

    return geodesic_distance(lat1, lon1, lat2, lon2), geodesic_azimuth(
        lat1, lon1, lat2, lon2
    )


def shoot(
    lon: Union[float, np.ndarray],
    lat: Union[float, np.ndarray],
    azimuth: Union[float, np.ndarray],
    maxdist: Union[float, np.ndarray],
) -> Tuple[
    Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]
]:
    """
    Calculate endpoint given starting point, azimuth and distance.

    Parameters
    ----------
    lon : Union[float, np.ndarray]
        Starting longitude(s) in degrees
    lat : Union[float, np.ndarray]
        Starting latitude(s) in degrees
    azimuth : Union[float, np.ndarray]
        Initial azimuth(s) in degrees
    maxdist : Union[float, np.ndarray]
        Distance(s) to travel in kilometers

    Returns
    -------
    Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]
        Tuple containing:
        - final_lon : Final longitude(s) in degrees
        - final_lat : Final latitude(s) in degrees
        - back_azimuth : Back azimuth(s) in degrees

    Notes
    -----
    This function implements a geodesic shooting algorithm based on
    T. Vincenty's method. It accounts for the Earth's ellipsoidal shape.

    Raises
    ------
    ValueError
        If attempting to shoot from a pole in a direction not along a meridian.

    Examples
    --------
    >>> lon_f, lat_f, baz = shoot(0, 0, 90, 111.195)  # ~1 degree at equator
    >>> round(lon_f, 6)
    1.0
    >>> round(lat_f, 6)
    0.0
    >>> round(baz, 6)
    270.0
    """

    # Convert inputs to arrays
    lon, lat, azimuth, maxdist = map(np.asarray, (lon, lat, azimuth, maxdist))

    glat1 = lat * DEG2RAD
    glon1 = lon * DEG2RAD
    s = maxdist / 1.852  # Convert km to nautical miles
    faz = azimuth * DEG2RAD

    # Check for pole condition
    pole_condition = (np.abs(np.cos(glat1)) < EPS) & ~(np.abs(np.sin(faz)) < EPS)
    if np.any(pole_condition):
        raise ValueError("Only N-S courses are meaningful, starting at a pole!")

    r = 1 - FLATTENING
    tu = r * np.tan(glat1)
    sf = np.sin(faz)
    cf = np.cos(faz)

    # Handle cf == 0 case
    b = np.zeros_like(cf)
    nonzero_cf = cf != 0
    b[nonzero_cf] = 2.0 * np.arctan2(tu[nonzero_cf], cf[nonzero_cf])

    cu = 1.0 / np.sqrt(1 + tu * tu)
    su = tu * cu
    sa = cu * sf
    c2a = 1 - sa * sa
    x = 1.0 + np.sqrt(1.0 + c2a * (1.0 / (r * r) - 1.0))
    x = (x - 2.0) / x
    c = 1.0 - x
    c = (x * x / 4.0 + 1.0) / c
    d = (0.375 * x * x - 1.0) * x
    tu = s / (r * EARTH_RADIUS_NM * c)
    y = tu.copy()

    # Iterative solution
    while True:
        sy = np.sin(y)
        cy = np.cos(y)
        cz = np.cos(b + y)
        e = 2.0 * cz * cz - 1.0
        c = y.copy()
        x = e * cy
        y = e + e - 1.0
        y = (
            ((sy * sy * 4.0 - 3.0) * y * cz * d / 6.0 + x) * d / 4.0 - cz
        ) * sy * d + tu

        if np.all(np.abs(y - c) <= EPS):
            break

    b = cu * cy * cf - su * sy
    c = r * np.sqrt(sa * sa + b * b)
    d = su * cy + cu * sy * cf
    glat2 = (np.arctan2(d, c) + pi) % (2 * pi) - pi
    c = cu * cy - su * sy * cf
    x = np.arctan2(sy * sf, c)
    c = ((-3.0 * c2a + 4.0) * FLATTENING + 4.0) * c2a * FLATTENING / 16.0
    d = ((e * cy * c + cz) * sy * c + y) * sa
    glon2 = ((glon1 + x - (1.0 - c) * d * FLATTENING + pi) % (2 * pi)) - pi
    baz = (np.arctan2(sa, b) + pi) % (2 * pi)

    return (glon2 * RAD2DEG, glat2 * RAD2DEG, baz * RAD2DEG)


def create_polygon(coordinates: List[Tuple[float, float]]) -> Polygon:
    """
    Create a polygon from a list of (longitude, latitude) coordinates.

    Parameters
    ----------
    coordinates : List[Tuple[float, float]]
        List of (longitude, latitude) coordinate pairs that define the polygon vertices.
        The first and last points should be the same to close the polygon.

    Returns
    -------
    Polygon
        A shapely Polygon object.

    Examples
    --------
    >>> coords = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]  # Square
    >>> poly = create_polygon(coords)
    """

    return Polygon(coordinates)


def points_in_polygon(
    lon: Union[List[float], np.ndarray],
    lat: Union[List[float], np.ndarray],
    polygon: Polygon,
) -> np.ndarray:
    """
    Check which points are inside a polygon.

    Parameters
    ----------
    lon : Union[List[float], np.ndarray]
        Array or list of longitude values.
    lat : Union[List[float], np.ndarray]
        Array or list of latitude values.
        Must have the same shape as lon.
    polygon : Polygon
        A shapely Polygon object.

    Returns
    -------
    np.ndarray
        Boolean array indicating which points are inside the polygon.

    Raises
    ------
    ValueError
        If lon and lat arrays have different shapes.

    Examples
    --------
    >>> coords = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]  # Square
    >>> poly = create_polygon(coords)
    >>> lon = [0.5, 2.0]
    >>> lat = [0.5, 2.0]
    >>> mask = points_in_polygon(lon, lat, poly)
    >>> print(mask)  # [True, False]
    """

    if isinstance(lon, list):
        lon = np.array(lon)
    if isinstance(lat, list):
        lat = np.array(lat)

    if lon.shape != lat.shape:
        raise ValueError("lon and lat arrays must have the same shape")

    # Create Point objects for each coordinate pair
    points_geom = [Point(x, y) for x, y in zip(lon, lat)]

    # Check which points are inside the polygon
    return np.array([polygon.contains(p) for p in points_geom])


def filter_points_in_polygon(
    lon: Union[List[float], np.ndarray],
    lat: Union[List[float], np.ndarray],
    polygon: Polygon,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter points to keep only those inside a polygon.

    Parameters
    ----------
    lon : Union[List[float], np.ndarray]
        Array or list of longitude values.
    lat : Union[List[float], np.ndarray]
        Array or list of latitude values.
        Must have the same shape as lon.
    polygon : Polygon
        A shapely Polygon object.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing:
        - filtered_lon : Array of longitudes inside the polygon
        - filtered_lat : Array of latitudes inside the polygon

    Raises
    ------
    ValueError
        If lon and lat arrays have different shapes.

    Examples
    --------
    >>> coords = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]  # Square
    >>> poly = create_polygon(coords)
    >>> lon = [0.5, 2.0]
    >>> lat = [0.5, 2.0]
    >>> filtered_lon, filtered_lat = filter_points_in_polygon(lon, lat, poly)
    >>> print(filtered_lon)  # [0.5]
    >>> print(filtered_lat)  # [0.5]
    """

    if isinstance(lon, list):
        lon = np.array(lon)
    if isinstance(lat, list):
        lat = np.array(lat)

    if lon.shape != lat.shape:
        raise ValueError("lon and lat arrays must have the same shape")

    mask = points_in_polygon(lon, lat, polygon)
    return lon[mask], lat[mask]


def buffer_area_for_polygon(polygon: Polygon, area_factor: float) -> Polygon:
    """
    Buffer the polygon by a factor of its area divided by its length.
    This is a heuristic to ensure that the buffer is proportional to the size of the polygon.

    Parameters
    ----------
    polygon : Polygon
        The polygon to be buffered.
    mas : float
        The buffer factor.

    Returns
    -------
    Polygon
        The buffered polygon.

    Example
    -------
    >>> from shapely.geometry import Polygon
    >>> polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    >>> mas = 0.1
    >>> buffered_polygon = buffer_area_for_polygon(polygon, mas)
    >>> print(buffered_polygon)
    POLYGON ((-0.1 -0.1, 1.1 -0.1, 1.1 1.1, -0.1 1.1, -0.1 -0.1))
    """

    return polygon.buffer(area_factor * polygon.area / polygon.length)


def mask_points_outside_polygon(
    elements: np.ndarray, node_coords: np.ndarray, poly: Polygon
) -> np.ndarray:
    """
    Returns a boolean mask indicating which triangle elements have at least two vertices outside the polygon.

    This version uses matplotlib.path.Path for high-performance point-in-polygon testing.

    Parameters
    ----------
    elements : (n_elements, 3) np.ndarray
        Array containing indices of triangle vertices.
    node_coords : (n_nodes, 2) np.ndarray
        Array of node coordinates as (x, y) pairs.
    poly : shapely.geometry.Polygon
        Polygon used for containment checks.

    Returns
    -------
    mask : (n_elements,) np.ndarray
        Boolean array where True means at least two vertices of the triangle lie outside the polygon.

    Example
    -------
    >>> import numpy as np
    >>> from shapely.geometry import Polygon
    >>> elements = np.array([[0, 1, 2], [1, 2, 3]])
    >>> node_coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    >>> poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    >>> mask = mask_points_outside_polygon(elements, node_coords, poly)
    >>> print(mask)
    [False False]
    """

    tri_coords = node_coords[elements]

    x = tri_coords[..., 0]
    y = tri_coords[..., 1]

    inside = contains(poly, x, y)

    num_inside = np.sum(inside, axis=1)

    return num_inside < 3
