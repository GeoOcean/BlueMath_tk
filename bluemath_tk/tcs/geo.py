from math import acos, asin, atan2, cos, degrees, pi, radians, sin, sqrt

import numpy as np


def gc_distance(lat1, lon1, lat2, lon2):
    "Calculate great circle distance and azimuth (exact. parsed ml)"

    # distance
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    a = (
        sin((lat2 - lat1) / 2) ** 2
        + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    )
    if a < 0:
        a = 0
    if a > 1:
        a = 1

    r = 1
    rng = r * 2 * atan2(sqrt(a), sqrt(1 - a))
    rng = degrees(rng)

    # azimuth
    az = atan2(
        cos(lat2) * sin(lon2 - lon1),
        cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1),
    )
    if lat1 <= -pi / 2:
        az = 0
    if lat2 >= pi / 2:
        az = 0
    if lat2 <= -pi / 2:
        az = pi
    if lat1 >= pi / 2:
        az = pi

    az = az % (2 * pi)
    az = degrees(az)

    return rng, az


def shoot(lon, lat, azimuth, maxdist=None):
    """
    Shooter Function
    Original javascript on http://williams.best.vwh.net/gccalc.htm
    Translated to python by Thomas Lecocq
    """
    glat1 = lat * np.pi / 180.0
    glon1 = lon * np.pi / 180.0
    s = maxdist / 1.852  # from km to n mi
    faz = azimuth * np.pi / 180.0

    EPS = 0.00000000005
    if (np.abs(np.cos(glat1)) < EPS) and not (np.abs(np.sin(faz)) < EPS):
        print("Only N-S courses are meaningful, starting at a pole!")

    a = 6378.13 / 1.852
    f = 1 / 298.257223563
    r = 1 - f
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
    tu = s / (r * a * c)
    y = tu
    c = y + 1
    while np.abs(y - c) > EPS:
        sy = np.sin(y)
        cy = np.cos(y)
        cz = np.cos(b + y)
        e = 2.0 * cz * cz - 1.0
        c = y
        x = e * cy
        y = e + e - 1.0
        y = (
            ((sy * sy * 4.0 - 3.0) * y * cz * d / 6.0 + x) * d / 4.0 - cz
        ) * sy * d + tu

    b = cu * cy * cf - su * sy
    c = r * np.sqrt(sa * sa + b * b)
    d = su * cy + cu * sy * cf
    glat2 = (np.arctan2(d, c) + np.pi) % (2 * np.pi) - np.pi
    c = cu * cy - su * sy * cf
    x = np.arctan2(sy * sf, c)
    c = ((-3.0 * c2a + 4.0) * f + 4.0) * c2a * f / 16.0
    d = ((e * cy * c + cz) * sy * c + y) * sa
    glon2 = ((glon1 + x - (1.0 - c) * d * f + np.pi) % (2 * np.pi)) - np.pi

    baz = (np.arctan2(sa, b) + np.pi) % (2 * np.pi)

    glon2 *= 180.0 / np.pi
    glat2 *= 180.0 / np.pi
    baz *= 180.0 / np.pi

    return (glon2, glat2, baz)


def geo_distance_azimuth(lat_matrix, lon_matrix, lat_point, lon_point):
    """
    Returns geodesic distance and azimuth between lat,lon matrix and lat,lon
    point in degrees
    """

    arcl = np.zeros(lat_matrix.shape) * np.nan
    azi = np.zeros(lat_matrix.shape) * np.nan

    sh1, sh2 = lat_matrix.shape

    for i in range(sh1):
        for j in range(sh2):
            arcl[i, j], azi[i, j] = gc_distance(
                lat_point, lon_point, lat_matrix[i][j], lon_matrix[i][j]
            )

    return arcl, azi


def geo_distance_cartesian(y_matrix, x_matrix, y_point, x_point):
    """
    Returns cartesian distance between y,x matrix and y,x point
    """

    dist = np.zeros(y_matrix.shape) * np.nan

    sh1, sh2 = y_matrix.shape

    for i in range(sh1):
        for j in range(sh2):
            dist[i, j] = sqrt(
                (y_point - y_matrix[i][j]) ** 2 + (x_point - x_matrix[i][j]) ** 2
            )

    return dist


def degC2degN(degC):
    """
    Converts a wind direction on a unit circle (degrees cartesian) to
    a direction in nautical convention (degrees north).

    Angle in a unit circle:
    The angle between the horizontal east (E) and the head (pointing outwards), counter-clockwise

                ^ N (90)
                |
    W (180) <~~   ~~> E (0)
                |
                v S (270)

    Angle in nautical convention:
    The angle between the vertical up (N) and the tail (pointing inwards), clockwise

                | N (0)
                v
    W (270) ~~>   <~~ E (90)
                ^
                | S (180)
    """
    degN = np.mod(-degC + 270, 360)

    return degN


def degN2degC(degN):
    """
    Converts a wind direction in nautical convention (degrees north) to a
    a direction on a unit circle (degrees cartesian).
    """
    degC = np.mod(-degN + 270, 360)

    return degC


def GeoAzimuth(lat1, lon1, lat2, lon2):
    "Returns geodesic azimuth between point1 and point2"

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    az = atan2(
        cos(lat2) * sin(lon2 - lon1),
        cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1),
    )
    if lat1 <= -pi / 2:
        az = 0
    if lat2 >= pi / 2:
        az = 0
    if lat2 <= -pi / 2:
        az = pi
    if lat1 >= pi / 2:
        az = pi

    az = az % (2 * pi)
    az = degrees(az)

    return az


def GeoDistance(lat1, lon1, lat2, lon2):
    "Returns great circle distance between points in degrees"

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    a = (
        sin((lat2 - lat1) / 2) ** 2
        + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    )
    if a < 0:
        a = 0
    if a > 1:
        a = 1

    r = 1
    rng = r * 2 * atan2(sqrt(a), sqrt(1 - a))
    rng = degrees(rng)

    return rng
