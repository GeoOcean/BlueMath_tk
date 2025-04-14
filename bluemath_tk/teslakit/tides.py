#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr


def Calculate_MMSL(xda_tide, year_ini, year_end):
    '''
    Calculate monthly mean sea level

    xda_tide  - xarray.DataArray (time,)
    '''

    lout_mean = []
    lout_median = []
    lout_time = []
    for yy in range(year_ini, year_end+1):
        for mm in range(1,13):

            d1 = np.datetime64('{0:04d}-{1:02d}'.format(yy,mm))
            d2 = d1 + np.timedelta64(1, 'M')

            tide_sel_m = xda_tide.where(
                (xda_tide.time >= d1) & (xda_tide.time <= d2),
                drop = True)[:-2]
            time_sel = tide_sel_m.time.values

            if len(time_sel) >= 300:
                # mean, median and dates
                ts_mean = tide_sel_m.mean().values
                ts_median = tide_sel_m.median().values
                ts_time = time_sel[int(len(time_sel)/2)]

                lout_mean.append(ts_mean)
                lout_median.append(ts_median)
                lout_time.append(ts_time)


    # join output in xarray.Dataset
    xds_MMSL = xr.Dataset(
        {
            'mmsl':(('time',), lout_mean),
            'mmsl_median':(('time',), lout_median),
        },
        coords = {
            'time': lout_time
        }
    )

    return xds_MMSL

