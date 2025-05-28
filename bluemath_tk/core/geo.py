from math import pi
from typing import Tuple, Union

import numpy as np

# from .constants import EARTH_RADIUS_NM
EARTH_RADIUS_NM = 6378.135 / 1.852

# Constants
FLATTENING = 1 / 298.257223563
EPS = 0.00000000005
DEG2RAD = pi / 180.0
RAD2DEG = 180.0 / pi


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
    array([90., 90.])
    """

    lon1, lat1, lon2, lat2 = convert_to_radians(lon1, lat1, lon2, lat2)

    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    a = np.clip(a, 0, 1)

    rng = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return np.degrees(rng)


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
    array([90., 45.])
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


if __name__ == "__main__":
    # Examples with float inputs
    print(geodesic_distance(0, 0, 0, 90))
    print(geodesic_azimuth(0, 0, 0, 90))
    print(geodesic_distance_azimuth(0, 0, 0, 90))
    print(shoot(0, 0, 90, 111.195))

    # Examples with numpy arrays
    lat1 = np.array([0, 45])
    lon1 = np.array([0, -90])
    lat2 = np.array([0, -45])
    lon2 = np.array([90, 90])
    print(geodesic_distance(lat1, lon1, lat2, lon2))
    print(geodesic_azimuth(lat1, lon1, lat2, lon2))
    print(geodesic_distance_azimuth(lat1, lon1, lat2, lon2))
    azimuth = np.array([90, 45])
    maxdist = np.array([111.195, 111.195])
    print(shoot(lon1, lat1, azimuth, maxdist))
