from math import pi
from typing import Tuple, Union

import numpy as np

# Constants
FLATTENING = 1 / 298.257223563
EPS = 0.00000000005
DEG2RAD = pi / 180.0
RAD2DEG = 180.0 / pi


def convert_to_radians(*args: Union[float, NDArray]) -> tuple:
    """Convert degree inputs to radians.

    Args:
        *args: Variable number of degree inputs

    Returns:
        tuple: Input values converted to radians
    """
    return tuple(np.radians(arg) for arg in args)


def geodesic_distance(
    lat1: Union[float, NDArray],
    lon1: Union[float, NDArray],
    lat2: Union[float, NDArray],
    lon2: Union[float, NDArray],
) -> Union[float, NDArray]:
    """Calculate great circle distance between two points on Earth.

    Args:
        lat1, lon1: Latitude and longitude of first point in degrees
        lat2, lon2: Latitude and longitude of second point in degrees

    Returns:
        float: Great circle distance in degrees
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
    lat1: Union[float, NDArray],
    lon1: Union[float, NDArray],
    lat2: Union[float, NDArray],
    lon2: Union[float, NDArray],
) -> Union[float, NDArray]:
    """Calculate azimuth between two points on Earth.

    Args:
        lat1, lon1: Latitude and longitude of first point in degrees
        lat2, lon2: Latitude and longitude of second point in degrees

    Returns:
        float: Azimuth in degrees
    """
    lon1, lat1, lon2, lat2 = convert_to_radians(lon1, lat1, lon2, lat2)

    az = np.arctan2(
        np.cos(lat2) * np.sin(lon2 - lon1),
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1),
    )

    # Handle special cases at poles
    az = np.where(lat1 <= -pi/2, 0, az)
    az = np.where(lat2 >= pi/2, 0, az)
    az = np.where(lat2 <= -pi/2, pi, az)
    az = np.where(lat1 >= pi/2, pi, az)

    return np.degrees(az % (2 * pi))


def geodesic_distance_azimuth(
    lat1: Union[float, NDArray],
    lon1: Union[float, NDArray],
    lat2: Union[float, NDArray],
    lon2: Union[float, NDArray],
) -> Tuple[Union[float, NDArray], Union[float, NDArray]]:
    """Calculate both great circle distance and azimuth between two points.

    Args:
        lat1, lon1: Latitude and longitude of first point in degrees
        lat2, lon2: Latitude and longitude of second point in degrees

    Returns:
        tuple: (distance in degrees, azimuth in degrees)
    """
    return geodesic_distance(lat1, lon1, lat2, lon2), geodesic_azimuth(lat1, lon1, lat2, lon2)


def shoot(
    lon: float,
    lat: float,
    azimuth: float,
    maxdist: float,
) -> Tuple[float, float, float]:
    """Calculate endpoint given starting point, azimuth and distance.

    Args:
        lon: Starting longitude in degrees
        lat: Starting latitude in degrees
        azimuth: Initial azimuth in degrees
        maxdist: Distance to travel in kilometers

    Returns:
        tuple: (final longitude, final latitude, back azimuth) in degrees
    """
    glat1 = lat * DEG2RAD
    glon1 = lon * DEG2RAD
    s = maxdist / 1.852  # Convert km to nautical miles
    faz = azimuth * DEG2RAD

    if (abs(np.cos(glat1)) < EPS) and not (abs(np.sin(faz)) < EPS):
        raise ValueError("Only N-S courses are meaningful, starting at a pole!")

    r = 1 - FLATTENING
    tu = r * np.tan(glat1)
    sf = np.sin(faz)
    cf = np.cos(faz)
    
    if cf == 0:
        b = 0.0
    else:
        b = 2.0 * np.arctan2(tu, cf)

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
    y = tu

    # Iterative solution
    while True:
        sy = np.sin(y)
        cy = np.cos(y)
        cz = np.cos(b + y)
        e = 2.0 * cz * cz - 1.0
        c = y
        x = e * cy
        y = e + e - 1.0
        y = (((sy * sy * 4.0 - 3.0) * y * cz * d / 6.0 + x) * d / 4.0 - cz) * sy * d + tu
        
        if abs(y - c) <= EPS:
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

    return (
        glon2 * RAD2DEG,
        glat2 * RAD2DEG,
        baz * RAD2DEG
    )
