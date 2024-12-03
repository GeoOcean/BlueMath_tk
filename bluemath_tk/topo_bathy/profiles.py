import numpy as np
from scipy import interpolate

# custom 2D profiles for SWASH numerical model cases


def reef(dx, h0, Slope1, Slope2, Wreef, Wfore, bCrest, emsl):
    """
    Reef morphologic profile (Pearson et al. 2017)

    dx:      bathymetry mesh resolution at x axes (m)
    h0:      offshore depth (m)
    Slope1:  fore shore slope
    Slope2:  inner shore slope
    Wreef:   reef bed width (m)
    Wfore:   flume length before fore toe (m)
    bCrest:  beach heigh (m)
    emsl:    mean sea level (m)

    return depth data values
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


def linear(dx, h0, bCrest, m, Wfore):
    """
    simple linear profile (y = m * x + n)

    dx:      bathymetry mesh resolution at x axes (m)
    h0:      offshore depth (m)
    bCrest:  beach heigh (m)
    m:       profile slope
    Wfore:   flume length before slope toe (m)

    return depth data values
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


def parabolic(dx, h0, A, xBeach, bCrest):
    """
    Parabolic profile (y = A * x^(2/3))

    dx:      bathymetry mesh resolution at x axes (m)
    h0:      offshore depth (m)
    A:       parabola coefficient
    xBeach:  beach length(m)
    bCrest:  beach heigh (m)
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


def biparabolic(h0, hsig, omega_surf_list, TR):
    """
    Biparabolic profile (Bernabeu et al. 2013)

    h0:          offshore water level (m)
    hsig:        significant wave height (m)
    omega_surf:  intertidal dimensionless fall velocity (1 <= omega_surf <= 5)
    TR:          tidal range (m)
    """

    # Discontinuity point
    hr = 1.1 * hsig + TR

    # Legal point
    ha = 3 * hsig + TR

    # Empirical adjusted parameters
    A = 0.21 - 0.02 * omega_surf_list
    B = 0.89 * np.exp(-1.24 * omega_surf_list)
    C = 0.06 + 0.04 * omega_surf_list
    D = 0.22 * np.exp(-0.83 * omega_surf_list)

    # Different values for the height
    h = np.linspace(0, h0, 150)
    h_cont = []

    # Important points for the profile
    xr = (hr / A) ** (3 / 2) + (B / (A ** (3 / 2))) * hr**3

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


def custom_profile(dx, emsl, xs, ys):
    """
    custom N points profile

    dx:   bathymetry mesh resolution at x axes (m)
    xs:    x values array
    ys:    y values array
    emsl:  mean sea level (m)
    """

    # flume length
    xnew = np.arange(xs[0], xs[-1], dx)
    f = interpolate.interp1d(xs, ys)
    ynew = f(xnew)

    depth = -ynew

    return depth
