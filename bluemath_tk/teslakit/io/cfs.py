#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op

# pip
import xarray as xr
import numpy as np
import netCDF4 as nc4


def ReadSLP(p_db, lat1, lat2, lon1, lon2, resample, p_save = None):
    'Read data from CFS SLP database: netCDF files'
    # TODO: poder parar y retomar la extraccion

    print('Reading SLP data from files...')

    ncfiles_1 = sorted(
        [op.join(p_db,f) for f in os.listdir(p_db) \
         if f.endswith('.nc') and 'gdas' in f]
    )
    ncfiles_2 = sorted(
        [op.join(p_db,f) for f in os.listdir(p_db) \
         if f.endswith('.nc') and 'cdas1' in f]
    )
    ncfiles = ncfiles_1 + ncfiles_2

    if not ncfiles:
        print('No files for extraction')
        return None

    # get time array
    time = []
    for f in ncfiles:
        with nc4.Dataset(f,'r') as ds:
            time_var = ds.variables['time']
            dtime = nc4.num2date(time_var[:],time_var.units)
            time += list(dtime)

    # get longitude and latitude indexes
    with nc4.Dataset(ncfiles[0],'r') as ds:
        lon = ds.variables['lon'][:]
        lat = ds.variables['lat'][:]
        ix_lon1 = np.where(lon==lon1)[0][0]
        ix_lon2 = np.where(lon==lon2)[0][0]
        ix_lat1 = np.where(lat==lat1)[0][0]
        ix_lat2 = np.where(lat==lat2)[0][0]
        longitude = lon[ix_lon1:ix_lon2+resample:resample]
        latitude = lat[ix_lat1:ix_lat2+resample:resample]

    # read slp file by file
    np_SLP = np.nan * np.ones((len(time), len(latitude), len(longitude)))
    ti = 0
    for f in ncfiles:
        print(f)
        with nc4.Dataset(f,'r') as ds:
            time_len = len(ds.variables['time'])
            slp = ds.variables['PRMSL_L101'][:]
            np_SLP[ti:ti+time_len,:,:] = slp[
                :,
                ix_lat1:ix_lat2+resample:resample,
                ix_lon1:ix_lon2+resample:resample
            ]
            ti += time_len

    # mount output dataset
    xds_SLP = xr.Dataset(
        {
            'SLP': (('time','latitude','longitude'), np_SLP),
        },
        coords = {
            'time': time,
            'longitude': longitude,
            'latitude': latitude,
        },
    )

    # save
    if p_save:
        xds_SLP.to_netcdf(p_save)

    return xds_SLP


