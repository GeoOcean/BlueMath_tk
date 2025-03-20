#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from cftime._cftime import DatetimeGregorian
import calendar

import numpy as np
import pandas as pd
import xarray as xr

# TODO replace xds2datetime with date2datenum (needs new date type switch) 
# TODO make a fast_reindex daily option
# TODO optimize, refactor, and tests


# MATLAB TIMES

def datematlab2datetime(datenum_matlab):
    'Return python datetime for matlab datenum. Transform and adjust from matlab.'

    d = datetime.fromordinal(int(datenum_matlab)) + \
    timedelta(days=float(datenum_matlab % 1)) - \
        timedelta(days=366) + timedelta(microseconds=0)

    return d

def datevec2datetime(d_vec):
    '''
    Returns datetime list from a datevec matrix
    d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
    '''

    return [datetime(d[0], d[1], d[2], d[3], d[4]) for d in d_vec]

def DateConverter_Mat2Py(datearray_matlab):
    'Parses matlab datenum array to python datetime list'

    return [datematlab2datetime(x) for x in datearray_matlab]


# PYTHON TIMES

def xds2datetime(d64):
    'converts xr.Dataset np.datetime64[ns] into datetime'
    # TODO: hour minutes and seconds 

    return datetime(d64.dt.year, d64.dt.month, d64.dt.day)

def npdt64todatetime(dt64):
    'converts np.datetime64[ns] into datetime'

    ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime(1970, 1, 1) + timedelta(seconds=ts)

# declare aux function for generating datetime.datetime array
def generate_datetimes(t0, t1, dtype='datetime64[h]'):

    # lambda vectorized: datetime.datetime from utc timestamp
    lb_dfts = lambda t: datetime.utcfromtimestamp(t)
    np_dfts = np.vectorize(lb_dfts)

    # mount times array
    dtd = {
        'datetime64[h]': timedelta(hours=1),
        'datetime64[D]': timedelta(days=1),
    }
    tdd = dtd[dtype]

    tg = np.arange(t0, t1 + tdd, dtype=dtype)
    tg = (tg - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    tg = np_dfts(tg)

    return tg

def fast_reindex_hourly(xds_data):
    '''
    Fast and secure method to reindex (pad) xarray.Dataset to hourly data

    xds_data - xarray.Dataset with time coordinate
    '''

    # def aux function for getting timedeltas as int array
    def get_deltas(t):
        t_df = (np.diff(t))
        t_tp = type(t_df[0])

        # total number of hours each date interval
        if  t_tp == np.timedelta64:
            d = t_df.astype('timedelta64[h]') / np.timedelta64(1,'h')
            d = d.astype(int)

        else:
            # lambda vectorized: datetime.timedelta to number of hours 
            lb_ghs = lambda t: t.days*24
            np_ghs = np.vectorize(lb_ghs)
            d = np_ghs(t_df)

        return d

    # generate output time array
    time_base = xds_data.time.values[:]
    t0, t1 = date2datenum(time_base[0]), date2datenum(time_base[-1])
    time_h = generate_datetimes(t0, t1, dtype='datetime64[h]')

    # repeat data in each variable new lower time frame (pad) 
    ds = get_deltas(time_base)
    dv = {}
    for vn in xds_data.keys():
        b = xds_data[vn].values[:]
        p = np.append(np.repeat(b[:-1], ds), b[-1])
        dv[vn] = (('time',), p)

    # return xarray.Dataset
    xds_out = xr.Dataset(dv, coords={'time': time_h})

    return xds_out

def fast_reindex_hourly_nsim(xds_data):
    '''
    Fast and secure method to reindex (pad) xarray.Dataset to hourly data

    xds_data - xarray.Dataset with time, n_sim coordinates
    '''

    # def aux function for getting timedeltas as int array
    def get_deltas(t):
        t_df = (np.diff(t))
        t_tp = type(t_df[0])

        # total number of hours each date interval
        if  t_tp == np.timedelta64:
            d = t_df.astype('timedelta64[h]') / np.timedelta64(1,'h')
            d = d.astype(int)

        else:
            # lambda vectorized: datetime.timedelta to number of hours 
            lb_ghs = lambda t: t.days*24
            np_ghs = np.vectorize(lb_ghs)
            d = np_ghs(t_df)

        return d

    # generate output time array
    time_base = xds_data.time.values[:]
    t0, t1 = date2datenum(time_base[0]), date2datenum(time_base[-1])
    time_h = generate_datetimes(t0, t1, dtype='datetime64[h]')

    # repeat data in each variable new lower time frame (pad) 
    ds = get_deltas(time_base)

    # iterate n_sim dimension
    l_sim = []
    for s in xds_data.n_sim:
        xds_d = xds_data.sel(n_sim=s)

        # fast reindex
        dv = {}
        for vn in xds_d.keys():
            b = xds_d[vn].values[:]
            p = np.append(np.repeat(b[:-1], ds), b[-1])
            dv[vn] = (('time',), p)
        # return xarray.Dataset
        xds_rd = xr.Dataset(dv, coords={'time': time_h})
        l_sim.append(xds_rd)

    # concat simulations
    xds_out = xr.concat(l_sim, 'n_sim')

    return xds_out

def xds_reindex_daily(xds_data,  dt_lim1=None, dt_lim2=None):
    '''
    Reindex xarray.Dataset to daily data between optional limits
    '''

    # TODO: remove limits from inside function

    # TODO: remove this swich and use date2datenum
    if isinstance(xds_data.time.values[0], datetime):
        xds_dt1 = xds_data.time.values[0]
        xds_dt2 = xds_data.time.values[-1]
    else:
        # parse xds times to python datetime
        xds_dt1 = xds2datetime(xds_data.time[0])
        xds_dt2 = xds2datetime(xds_data.time[-1])

    # cut data at limits
    if dt_lim1:
        xds_dt1 = max(xds_dt1, dt_lim1)
    if dt_lim2:
        xds_dt2 = min(xds_dt2, dt_lim2)

    # number of days
    num_days = (xds_dt2-xds_dt1).days+1

    # reindex xarray.Dataset
    return xds_data.reindex(
        {'time': [xds_dt1 + timedelta(days=i) for i in range(num_days)]},
        method = 'pad',
    )

def xds_reindex_monthly(xds_data):
    '''
    Reindex xarray.Dataset to monthly data
    '''

    # TODO: remove this swich and use date2datenum
    if isinstance(xds_data.time.values[0], datetime):
        xds_dt1 = xds_data.time.values[0]
        xds_dt2 = xds_data.time.values[-1]
    else:
        # parse xds times to python datetime
        xds_dt1 = xds2datetime(xds_data.time[0])
        xds_dt2 = xds2datetime(xds_data.time[-1])

    # number of months
    num_months = (xds_dt2.year - xds_dt1.year)*12 + \
            xds_dt2.month - xds_dt1.month

    # reindex xarray.Dataset
    return xds_data.reindex(
        {'time': [xds_dt1 + relativedelta(months=i) for i in range(num_months)]},
        method = 'pad',
    )

def xds_common_dates_daily(xds_list):
    '''
    returns daily datetime array between a list of xarray.Dataset comon date
    limits
    '''

    d1, d2 = xds_limit_dates(xds_list)

    return [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]

def xds_limit_dates(xds_list):
    '''
    returns datetime common limits between a list of xarray.Dataset
    '''

    d1 = None
    d2 = None

    for xds_e in xds_list:

        # TODO: remove this swich and use date2datenum
        if isinstance(xds_e.time.values[0], datetime):
            xds_e_dt1 = xds_e.time.values[0]
            xds_e_dt2 = xds_e.time.values[-1]
        else:
            # parse xds times to python datetime
            xds_e_dt1 = xds2datetime(xds_e.time[0])
            xds_e_dt2 = xds2datetime(xds_e.time[-1])

        if d1 == None:
            d1 = xds_e_dt1
            d2 = xds_e_dt2

        d1 = max(xds_e_dt1, d1)
        d2 = min(xds_e_dt2, d2)

    return d1, d2

def xds_further_dates(xds_list):
    '''
    returns datetime further date limits between a list of xarray.Dataset
    '''

    d1 = None
    d2 = None

    for xds_e in xds_list:

        # TODO: remove this swich and use date2datenum
        if isinstance(xds_e.time.values[0], datetime):
            xds_e_dt1 = xds_e.time.values[0]
            xds_e_dt2 = xds_e.time.values[-1]
        else:
            # parse xds times to python datetime
            xds_e_dt1 = xds2datetime(xds_e.time[0])
            xds_e_dt2 = xds2datetime(xds_e.time[-1])

        if d1 == None:
            d1 = xds_e_dt1
            d2 = xds_e_dt2

        d1 = min(xds_e_dt1, d1)
        d2 = max(xds_e_dt2, d2)

    return d1, d2

def date2yearfrac(d):
    'Returns date d in fraction of the year'

    # get timetuple obj 
    if isinstance(d, datetime):
        ttup = d.timetuple()

    elif isinstance(d, np.datetime64):
        ttup = npdt64todatetime(d).timetuple()

    elif isinstance(d, DatetimeGregorian):
        ttup = d.timetuple()

    # total number of days in year
    year_ndays = 366.0 if calendar.isleap(ttup.tm_year) else 365.0

    return ttup.tm_yday / year_ndays

def date2datenum(d):
    'Returns date d (any format) in datetime'

    # TODO: new type switch for xarray.core numpy.datetime64
    # TODO: rename to date2datetime

    if isinstance(d, datetime):
        return d

    # else get timetuple
    elif isinstance(d, np.datetime64):
        ttup = npdt64todatetime(d).timetuple()

    elif isinstance(d, DatetimeGregorian):
        ttup = d.timetuple()

    # return datetime 
    return datetime(*ttup[:6])

def get_years_months_days(time):
    '''
    Returns years, months, days of time in separete lists

    (Used to avoid problems with dates type)
    '''

    t0 = time[0]
    if isinstance(t0, (date, datetime, DatetimeGregorian)):
        ys = np.asarray([x.year for x in time])
        ms = np.asarray([x.month for x in time])
        ds = np.asarray([x.day for x in time])

    else:
        tpd = pd.DatetimeIndex(time)
        ys = tpd.year
        ms = tpd.month
        ds = tpd.day

    return ys, ms, ds


# aux. functions for teslakit hourly output

def repair_times_hourly(xds):
    'ensures that xarray.Dataset time index is rounded to nearest hour and does not repeat values'

    # round times to hour (only for np.datetime64)
    if  isinstance(xds.time.values[0], np.datetime64):
        xds['time'] = xds['time'].dt.round('H')

    # remove duplicates
    _, ix = np.unique(xds['time'], return_index=True); xds = xds.isel(time=ix)

    return xds

def hours_since(base_date, target_dates):
    'fast method for locating "target_dates" hours since "base_date"'

    return (target_dates - base_date).astype('timedelta64[h]').astype(int)

def add_max_storms_mask(xds, times_max_storms, name_mask='max_storms'):
    'fast method for adding a "max_storms" mask to a hourly xarray.Dataset'

    # find max storm indexes and make boolean mask
    ixs = hours_since(xds.time.values[0], times_max_storms)
    np_mask = np.zeros(len(xds.time.values), dtype='bool'); np_mask[ixs] = True

    # add mask to dataset
    xds[name_mask] = (('time'), np_mask)

    return xds

