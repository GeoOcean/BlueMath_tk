#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from .geo import shoot


def track_from_parameters(
    pmin, vmean, delta, gamma,
    x0, y0, x1, R,
    date_ini, hours,
    great_circle=False):
    '''
    Calculates storm track variables from storm track parameters

    pmin, vmean, delta, gamma  - storm track parameters
    x0, y0                     - site coordinates (longitude, latitude)
    x1                         - enter point in computational grid
    R                          - radius (º)
    date_ini                   - initial date 'yyyy-mm-dd HH:SS'
    hours                      - number of hours to generate
    great_circle               - True for using great circle lon,lat calculation
    '''

    RE = 6378.135  # earth radius

    # generation of storm track 
    xc = x0 + R * np.sin(delta * np.pi/180)  # enter point in the smaller radius
    yc = y0 + R * np.cos(delta * np.pi/180)

    d = (x1 - xc) / np.sin(gamma * np.pi/180)
    y1 = yc + d * np.cos(gamma * np.pi/180)

    # time array for SWAN input
    time_input = pd.date_range(date_ini, periods=hours, freq='H')

    # storm track (pd.DataFrame)
    st = pd.DataFrame(index=time_input, columns=['move', 'vf', 'pn', 'p0', 'lon', 'lat'])
    st['move'] = gamma
    st['vf'] = vmean
    st['pn'] = 1013
    st['p0'] = pmin

    # calculate lon and lat
    if not great_circle:
        st['lon'] = x1 - (st['vf']*180/(RE*np.pi)) * np.sin(gamma*np.pi/180) * list(range(len(st)))
        st['lat'] = y1 - (st['vf']*180/(RE*np.pi)) * np.cos(gamma*np.pi/180) * list(range(len(st)))

    else:
        x2 = x1 - (st['vf']*180/(RE*np.pi)) * np.sin(gamma*np.pi/180) * list(range(len(st)))
        y2 = y1 - (st['vf']*180/(RE*np.pi)) * np.cos(gamma*np.pi/180) * list(range(len(st)))
        xt, yt = [], []
        for i in list(range(0,hours)):
            glon, glat, baz = shoot(x1, y1, gamma+180, vmean * i)
            xt.append(glon)
            yt.append(glat)
        st['lon'] = xt
        st['lat'] = yt

    # add some metadata
    st.x0 = x0
    st.y0 = y0
    st.R = R

    return st

