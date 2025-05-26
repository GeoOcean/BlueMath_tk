#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#from mpl_toolkits.basemap import Basemap
from datetime import datetime, timedelta

def WorldMap(
    bk='simple',
    lon_1=-180, lat_1=-70,
    lon_2=180, lat_2=70,
    d_parallel=10., d_meridian=10.):
    'Returns customized world map (mercator)'

    # basemap: mercator
    m = Basemap(
        projection='merc',
        llcrnrlat=lat_1, urcrnrlat=lat_2,
        llcrnrlon=lon_1, urcrnrlon=lon_2,
        lat_ts=20,
        resolution='c')

    # draw parallels.
    parallels = np.arange(-90.,90.,d_parallel)
    m.drawparallels(parallels, labels=[1,0,0,0],fontsize=6)
    # draw meridians
    meridians = np.arange(0.,360.,d_meridian)
    m.drawmeridians(meridians, labels=[0,0,0,1],fontsize=6)

    # draw background
    if bk == 'simple':
        m.drawcoastlines()
        m.fillcontinents(color='coral',lake_color='aqua')
        m.drawmapboundary(fill_color='aqua')
    elif bk == 'shaderelief':
        m.shadedrelief()
    elif bk == 'etopo':
        m.etopo()
    elif bk == 'bluemarble':
        m.bluemarble()

    # return basemap
    return m


def WorldMap_Stations(
    xds_stations, bk='simple',
    lon_1=-180, lat_1=-70,
    lon_2=180, lat_2=70,
    d_parallel=10., d_meridian=10.,
    p_export=None):
    '''
    Plot mercator world map with CSIRO spec stations
    bk (background)= 'simple', 'shaderelief', 'etopo', 'bluemarble'
    lonlat1, lonlat2: basemap limits
    d_parallel, d_meridian: parallels and meridians interval
    '''

    # xds_station lon lat
    lon = xds_stations.longitude.values[:]
    lat = xds_stations.latitude.values[:]
    nms = xds_stations.station_name[:]

    # plot figure
    fig = plt.figure(figsize=(16,9))

    # Get customized basemap
    m = WorldMap(
        bk,
        lon_1, lat_1,
        lon_2, lat_2,
        d_parallel, d_meridian)

    # add stations
    m.scatter(lon, lat, s=24, c='r', latlon=True)

    # more info
    plt.title('CSIRO spec stations')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=300)
        plt.close()

def WorldGlobe_Stations(
    xds_stations, bk='simple',
    lon_0=0, lat_0=0,
    lon_1=None, lat_1=None,
    lon_2=None, lat_2=None,
    p_export=None):
    '''
    Plot orthogonal world globe with CSIRO spec stations
    bk (background): 'simple', 'shaderelief', 'etopo', 'bluemarble'
    lonlat0: center focus
    lonlat1, lonlat2: basemap limits
    '''
    # TODO: area zoom not working

    # xds_station lon lat
    lon = xds_stations.longitude.values[:]
    lat = xds_stations.latitude.values[:]
    nms = xds_stations.station_name[:]

    # plot figure
    fig = plt.figure(figsize=(12,12))

    # basemap: ortho
    m = Basemap(
        projection='ortho',
        lon_0=lon_0,
        lat_0=lat_0,
        lon_1=lon_1,
        lat_1=lat_1,
        lon_2=lon_2,
        lat_2=lat_2,
        resolution='l',
    )

    # draw parallels.
    parallels = np.arange(-90.,90.,10.)
    m.drawparallels(parallels, labels=[1,0,0,0],fontsize=6)
    # draw meridians
    meridians = np.arange(0.,360.,10.)
    m.drawmeridians(meridians, labels=[0,0,0,1],fontsize=6)

    # draw background
    if bk == 'simple':
        m.drawcoastlines()
        m.fillcontinents(color='coral',lake_color='aqua')
        m.drawmapboundary(fill_color='aqua')
    elif bk == 'shaderelief':
        m.shadedrelief()
    elif bk == 'etopo':
        m.etopo()
    elif bk == 'bluemarble':
        m.bluemarble()

    # add stations
    m.scatter(lon, lat, s=6, c='r', latlon=True)

    # more info
    plt.title('CSIRO spec stations')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=96)
        plt.close()

def WorldGlobeZoom_Stations(xds_stations, bk='simple', lon0=0, lat0=0, p_export=None):
    '''
    Plot orthogonal world globe zoom with CSIRO spec stations
    bk (background): 'simple', 'shaderelief', 'etopo', 'bluemarble'
    lonlat0: center focus, tuple (lon0, lat0)
    '''
    # TODO: no sale bien

    delta_x = 5000000
    delta_y = 5000000

    # xds_station lon lat
    lon = xds_stations.longitude.values[:]
    lat = xds_stations.latitude.values[:]
    nms = xds_stations.station_name[:]

    # plot figure
    fig = plt.figure(figsize=(12,12))

    # basemap: ortho
    m1 = Basemap(
            projection='ortho',
            lon_0=lon0,
            lat_0=lat0,
            resolution=None
        )
    # zoom
    #ax = fig.add_axes([0.1,0.1,0.8,0.8],facecolor='k')
    xc,yc = m1(lon0, lat0)
    m = Basemap(
        projection='ortho',
        lon_0=lon0,
        lat_0=lat0,
        resolution='l',
        llcrnrx=xc-delta_x, llcrnry=yc-delta_y,
        urcrnrx=yc+delta_x, urcrnry=yc+delta_y,
    )

    # draw parallels.
    parallels = np.arange(-90.,90.,10.)
    # m1.drawparallels(parallels, labels=[1,0,0,0],fontsize=6)
    # draw meridians
    meridians = np.arange(0.,360.,10.)
    #m1.drawmeridians(meridians, labels=[0,0,0,1],fontsize=6)

    # draw background
    if bk == 'simple':
        m.drawcoastlines()
        m.fillcontinents(color='coral',lake_color='aqua')
        m.drawmapboundary(fill_color='aqua')
    elif bk == 'shaderelief':
        m.shadedrelief()
    elif bk == 'etopo':
        m.etopo()
    elif bk == 'bluemarble':
        m.bluemarble()

    # add stations
    m.scatter(lon, lat, s=6, c='r', latlon=True)

    # more info
    plt.title('CSIRO spec stations')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=96)
        plt.close()


def WorldMap_GriddedCoords(
    xds_gridded, bk='simple',
    lon_1=-180, lat_1=-70,
    lon_2=180, lat_2=70,
    d_parallel=10.,d_meridian=10.,
    p_export=None):
    '''
    Plot mercator world map with CSIRO gridded coordinates
    bk (background)= 'simple', 'shaderelief', 'etopo', 'bluemarble'
    '''

    # xds_station lon lat
    lon = xds_gridded.longitude.values[:]
    lat = xds_gridded.latitude.values[:]

    # land mask 
    use_mask = 'mask' in xds_gridded.variables

    # fix lon for plotting
    lon[np.where(lon>180.0)] = lon[np.where(lon>180.0)]-360

    # plot figure
    fig = plt.figure(figsize=(16,9))

    # Get customized basemap
    m = WorldMap(
        bk,
        lon_1, lat_1,
        lon_2, lat_2,
        d_parallel, d_meridian)

    # add stations
    xx,yy = np.meshgrid(lon, lat)
    if use_mask:
        mask = xds_gridded.mask.values[:]
        xx_plot = xx[mask]
        yy_plot = yy[mask]
    else:
        xx_plot = xx
        yy_plot = yy

    m.scatter(xx_plot, yy_plot, s=6, c='r', latlon=True)

    # more info
    plt.title('CSIRO gridded coords')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=96)
        plt.close()


