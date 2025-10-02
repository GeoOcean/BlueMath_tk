from typing import List, Union

import numpy as np
from scipy import interpolate


def reef(
    dx: float,
    h0: float,
    Slope1: float,
    Slope2: float,
    Wreef: float,
    Wfore: float,
    bCrest: float,
    emsl: float,
) -> np.ndarray:
    """
    Generate a reef morphologic profile based on Pearson et al. 2017.

    This function creates a bathymetric profile for reef environments with
    distinct sections: fore reef, reef slope, reef flat, and inner slope.

    Parameters
    ----------
    dx : float
        Bathymetry mesh resolution at x axes (m)
    h0 : float
        Offshore depth (m)
    Slope1 : float
        Fore shore slope (dimensionless)
    Slope2 : float
        Inner shore slope (dimensionless)
    Wreef : float
        Reef bed width (m)
    Wfore : float
        Flume length before fore toe (m)
    bCrest : float
        Beach height (m)
    emsl : float
        Mean sea level (m)

    Returns
    -------
    np.ndarray
        Depth data values representing the reef profile (m)

    Notes
    -----
    The profile consists of several sections:
    - Fore reef: constant depth at offshore level
    - Reef slope: linear slope from offshore to reef flat
    - Reef flat: constant depth at mean sea level
    - Inner slope: linear slope from reef flat to beach
    - Plane beach: gentle slope for overtopping dissipation
    """

    # flume length
    W_inner = bCrest / Slope2
    W1 = int(abs(h0 - emsl) / Slope1)

    # sections length
    x1 = np.arange(0, Wfore, dx)
    x2 = np.arange(0, W1, dx)
    x3 = np.arange(0, Wreef, dx)
    x4 = np.arange(0, W_inner, dx)

    # curve equation
    y_fore = np.zeros(len(x1)) + [h0]
    y1 = -Slope1 * x2 + h0
    y2 = np.zeros(len(x3)) + emsl
    y_inner = -Slope2 * x4 + emsl

    # overtopping cases: an inshore plane beach to dissipate overtopped flux
    plane = 0.005 * np.arange(0, 150, 1) + y_inner[-1]

    # concatenate depth
    depth = np.concatenate([y_fore, y1, y2, y_inner, plane])

    return depth


def linear(dx: float, h0: float, bCrest: float, m: float, Wfore: float) -> np.ndarray:
    """
    Generate a simple linear profile (y = m * x + n).

    This function creates a bathymetric profile with a constant slope
    from offshore to the beach crest.

    Parameters
    ----------
    dx : float
        Bathymetry mesh resolution at x axes (m)
    h0 : float
        Offshore depth (m)
    bCrest : float
        Beach height (m)
    m : float
        Profile slope (dimensionless)
    Wfore : float
        Flume length before slope toe (m)

    Returns
    -------
    np.ndarray
        Depth data values representing the linear profile (m)

    Notes
    -----
    The profile consists of:
    - Fore section: constant depth at offshore level
    - Main slope: linear slope from offshore to beach
    - Beach slope: linear slope from sea level to beach crest
    - Plane beach: gentle slope for overtopping dissipation
    """

    # Flume length
    W1 = int(h0 / m)
    W2 = int(bCrest / m)

    # Sections length
    x1 = np.arange(0, Wfore, dx)
    x2 = np.arange(0, W1, dx)
    x3 = np.arange(0, W2, dx)

    # Curve equation
    y_fore = np.zeros(len(x1)) + [h0]
    y1 = -m * x2 + h0
    y2 = -m * x3

    # Overtopping cases: an inshore plane beach to dissipate overtopped flux
    plane = 0.005 * np.arange(0, len(y2), 1) + y2[-1]  # Length bed = 2 L

    # concatenate depth
    depth = np.concatenate([y_fore, y1, y2, plane])

    return depth


def parabolic(
    dx: float, h0: float, A: float, xBeach: float, bCrest: float
) -> np.ndarray:
    """
    Generate a parabolic profile (y = A * x^(2/3)).

    This function creates a bathymetric profile following the equilibrium
    beach profile theory with a parabolic shape.

    Parameters
    ----------
    dx : float
        Bathymetry mesh resolution at x axes (m)
    h0 : float
        Offshore depth (m)
    A : float
        Parabola coefficient (m^(1/3))
    xBeach : float
        Beach length (m)
    bCrest : float
        Beach height (m)

    Returns
    -------
    np.ndarray
        Depth data values representing the parabolic profile (m)

    Notes
    -----
    The profile follows the equilibrium beach profile theory where
    depth varies as x^(2/3) from the shoreline to the closure depth.
    The beach section is linear from the shoreline to the beach crest.
    """

    lx = np.arange(1, xBeach, dx)
    y = -(bCrest / xBeach) * lx

    depth, xl = [], []
    x, z = 0, 0

    while z <= h0:
        z = A * x ** (2 / 3)
        depth.append(z)
        xl.append(x)
        x += dx

    f = interpolate.interp1d(xl, depth)
    xnew = np.arange(0, int(np.round(len(depth) * dx)), 1)
    ynew = f(xnew)

    # concatenate depth
    depth = np.concatenate([ynew[::-1], y])

    return depth


def biparabolic(
    h0: float, hsig: float, omega_surf_list: Union[float, np.ndarray], TR: float
) -> np.ndarray:
    """
    Generate a biparabolic profile based on Bernabeu et al. 2013.

    This function creates a bathymetric profile with two parabolic sections
    separated by a discontinuity point, suitable for mixed-sediment beaches.

    Parameters
    ----------
    h0 : float
        Offshore water level (m)
    hsig : float
        Significant wave height (m)
    omega_surf_list : float or np.ndarray
        Intertidal dimensionless fall velocity (1 <= omega_surf <= 5)
    TR : float
        Tidal range (m)

    Returns
    -------
    np.ndarray
        Depth data values representing the biparabolic profile (m)

    Notes
    -----
    The biparabolic profile consists of:
    - Lower section: parabolic profile for fine sediments
    - Upper section: parabolic profile for coarse sediments
    - Discontinuity point: transition between the two sections
    - The profile is centered on mean tide level

    The empirical parameters A, B, C, D are adjusted based on the
    dimensionless fall velocity parameter.
    """

    # Discontinuity point
    hr = 1.1 * hsig + TR

    # Legal point
    _ha = 3 * hsig + TR

    # Empirical adjusted parameters
    A = 0.21 - 0.02 * omega_surf_list
    B = 0.89 * np.exp(-1.24 * omega_surf_list)
    C = 0.06 + 0.04 * omega_surf_list
    D = 0.22 * np.exp(-0.83 * omega_surf_list)

    # Different values for the height
    h = np.linspace(0, h0, 150)
    h_cont = []

    # Important points for the profile
    _xr = (hr / A) ** (3 / 2) + (B / (A ** (3 / 2))) * hr**3

    # Lines for the profile
    x, Xx, X, xX = [], [], [], []

    for hs in h:  # For each vertical point
        if hs < hr:
            x_max = 0
            xapp = (hs / A) ** (3 / 2) + (B / (A ** (3 / 2))) * hs**3
            x.append(xapp)
            x_max = max(xapp, x_max)
            if hs > (hr - 1.5):
                Xxapp = (hs / C) ** (3 / 2) + (D / (C ** (3 / 2))) * hs**3
                Xx.append(Xxapp)
                h_cont.append(hs)
        else:
            Xapp = (hs / C) ** (3 / 2) + (D / (C ** (3 / 2))) * hs**3
            if (hs - hr) < 0.1:
                x_diff = x_max - Xapp
            X.append(Xapp)
            if hs < (hr + 1.5):
                xXapp = (hs / A) ** (3 / 2) + (B / (A ** (3 / 2))) * hs**3
                xX.append(xXapp)
                h_cont.append(hs)

    h_cont = np.array(h_cont)
    x_tot = np.concatenate((np.array(x), np.array(X) + x_diff))
    # x_cont = np.concatenate((np.array(Xx)+x_diff, np.array(xX)))

    # Centering the y-axis in the mean tide
    xnew = np.arange(0, x_tot[-1], 1)
    # xnew_border = np.arange(x_tot[-1]-x_cont[0], x_cont[-1]-x_cont[-1], 1)
    depth = h - TR / 2
    # border = (-h_cont+TR/2)

    f = interpolate.interp1d(x_tot, depth)
    # f1 = interpolate.interp1d(x_cont, border)
    ynew = f(xnew)[::-1]
    # ynew_border = f1(xnew_border)[::-1]

    depth = (h - TR / 2)[::-1]
    # border = (-h_cont+TR/2)[::-1]

    # plot
    # TODO: move plot to plots.py
    # fig, ax = plt.subplots(1, figsize = (12, 4))
    # ax.plot(xnew, -ynew, color='k', zorder=3)
    # ax.fill_between(xnew, np.zeros((len(xnew)))+(-ynew[0]),
    #                -ynew,facecolor="wheat", alpha=1, zorder=2)
    # ax.scatter(x_tot[-1]-xr, -hr+TR/2, s=30, c='red', label='Discontinuity point', zorder=5)
    # ax.fill_between(xnew, -ynew, np.zeros(len(xnew)), facecolor="deepskyblue", alpha=0.5, zorder=1)
    # ax.axhline(-ha+TR/2, color='grey', ls='-.', label='Available region')
    # ax.axhline(TR/2, color='silver', ls='--', label='HT')
    # ax.axhline(0, color='lightgrey', ls='--', label='MSL')
    # ax.axhline(-TR/2, color='silver', ls='--', label='LT')
    # ax.scatter(xnew_border, -ynew_border, c='k', s=1, marker='_', zorder=4)

    # attrbs
    # ax.set_ylim(-ynew[0], -ynew[-1]+1)
    # ax.set_xlim(0, x_tot[-1])
    # set_title  = '$\Omega_{sf}$ = ' + str(omega_surf_list)
    # set_title += ', TR = ' + str(TR)
    # ax.set_title(set_title)
    # ax.legend(loc='upper left')
    # ax.set_ylabel('$Depth$ $[m]$', fontweight='bold')
    # ax.set_xlabel('$X$ $[m]$', fontweight='bold')

    # TODO: deph or ynew ?
    return ynew


def custom_profile(
    dx: float,
    emsl: float,
    xs: Union[List[float], np.ndarray],
    ys: Union[List[float], np.ndarray],
) -> np.ndarray:
    """
    Generate a custom N-point profile from user-defined coordinates.

    This function creates a bathymetric profile by interpolating between
    user-specified x,y coordinate pairs.

    Parameters
    ----------
    dx : float
        Bathymetry mesh resolution at x axes (m)
    emsl : float
        Mean sea level (m) - used for reference but not directly applied
    xs : list or np.ndarray
        X coordinate values (m)
    ys : list or np.ndarray
        Y coordinate values (m) - positive values represent elevation above MSL

    Returns
    -------
    np.ndarray
        Depth data values representing the custom profile (m)

    Notes
    -----
    The function uses linear interpolation between the provided points.
    The output depths are negative values (below sea level) as is
    conventional for bathymetric data.

    The xs and ys arrays must have the same length and be sorted
    in increasing x order for proper interpolation.
    """

    # flume length
    xnew = np.arange(xs[0], xs[-1], dx)
    f = interpolate.interp1d(xs, ys)
    ynew = f(xnew)

    depth = -ynew

    return depth
