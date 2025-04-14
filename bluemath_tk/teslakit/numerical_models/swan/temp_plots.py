#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import pandas as pd

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# TODO: updated plot library, temporal
# new develops at swan/plots/ common.py stat.py nonstat.py

def aux_quiver(X, Y, var, vdir):
    '''
    interpolates var and plots quiver with var_dir. Requires open figure

    var  - variable module
    vdir - variable direction (º clockwise relative to North)
    '''

    size = 30  # quiver mesh size

    # var and dir interpolators 
    vdir_f = vdir.copy()
    vdir_f[np.isnan(vdir_f)] = 0
    f_dir = interpolate.interp2d(X, Y, vdir_f, kind='linear')

    var_f = var.copy()
    var_f[np.isnan(var_f)] = 0
    f_var = interpolate.interp2d(X, Y, var_f, kind='linear')

    # generate quiver mesh
    x_q = np.linspace(X[0], X[-1], num = size)
    y_q = np.linspace(Y[0], Y[-1], num = size)

    # interpolate data to quiver mesh
    vdir_q = f_dir(x_q, y_q)
    var_q = f_var(x_q, y_q)

    # u and v dir components
    u = np.sin(np.deg2rad(vdir_q))
    v = np.cos(np.deg2rad(vdir_q))

    # plot quiver
    plt.quiver(x_q, y_q, -u*var_q, -v*var_q) #, width=0.003, scale=1, scale_units='inches')

def plot_var_times(xds_out_case, var_name, p_export_case, quiver=False,
                   np_shore=np.array([]), cmap='jet'):
    '''
    Plots non-stationary SWAN execution output for selected var and case

    xds_out_case    - swan output (xarray.Dataset)
    var_name        - 'Hsig', 'Tm02', 'Tpsmoo'
    p_export        - path for exporting figures

    opt. args
    quiver     - True for adding directional quiver plot
    np_shore   - shoreline, np.array x = np_shore[:,0] y = np.shore[:,1]
    cmap       - matplotlib colormap
    '''

    if not op.isdir(p_export_case): os.makedirs(p_export_case)

    # iterate case output over time
    for t in xds_out_case.time.values[:]:
        xds_oct = xds_out_case.sel(time=t)

        # time string
        t_str = pd.to_datetime(str(t)).strftime('%Y%m%d-%H%M')

        # get mesh data from output dataset
        X = xds_oct.X.values[:]
        Y = xds_oct.Y.values[:]

        # get variable and units
        var = xds_oct[var_name].values[:]
        var_units = xds_oct[var_name].attrs['units']

        # new figure
        fig, ax0 = plt.subplots(nrows=1, figsize=(12, 12))
        var_title = '{0}'.format(var_name)  # title

        # pcolormesh
        ocean = plt.get_cmap('jet')  # colormap
        im = plt.pcolormesh(X, Y, var, cmap=cmap)

        # add quiver plot 
        if quiver:
            var_title += '-Dir'
            var_dir = xds_oct.Dir.values[:]
            aux_quiver(X, Y, var, var_dir)

        # shoreline
        if np_shore.any():
            x_shore = np_shore[:,0]
            y_shore = np_shore[:,1]
            plt.plot(x_shore, y_shore,'.', color='dimgray', markersize=3)

        # customize pcolormesh
        plt.title('{0} (t={1})'.format(var_title, t_str),
                  fontsize = 12, fontweight='bold')
        plt.xlabel(xds_oct.attrs['xlabel'], fontsize = 12)
        plt.ylabel(xds_oct.attrs['ylabel'], fontsize = 12)

        plt.axis('scaled')
        plt.xlim(X[0], X[-1])
        plt.ylim(Y[0], Y[-1])

        # add custom colorbar
        divider = make_axes_locatable(ax0)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        plt.ylabel('{0} ({1})'.format(var_name, var_units), fontsize = 12)

        # export fig
        p_ex = op.join(p_export_case, 'outmap_{0}_{1}.png'.format(var_name, t_str))
        fig.savefig(p_ex)

        # close fig 
        plt.close()

def plot_output_nonstat(xds_out, var_name, p_export, quiver=False,
                        np_shore=np.array([])):
    '''
    Plots non-stationary SWAN execution output for selected var, for every case

    xds_out    - swan output (xarray.Dataset)
    var_name   - 'Hsig', 'Tm02', 'Tpsmoo'
    p_export   - path for exporting figures

    opt. args
    quiver     - True for adding directional quiver plot
    np_shore   - shoreline, np.array x = np_shore[:,0] y = np.shore[:,1]
    '''

    # make export dir
    if not op.isdir(p_export): os.makedirs(p_export)

    for case_ix in xds_out.case.values[:]:

        # select case
        xds_out_case = xds_out.sel(case=case_ix)

        # output case subfolder
        case_id = '{0:04d}'.format(case_ix)
        p_export_case = op.join(p_export, case_id)

        # plot variable times
        plot_var_times(
            xds_out_case, var_name, p_export_case,
            quiver=quiver, np_shore=np_shore)

def plot_points_times(xds_out_case, p_export_case):
    '''
    Plots non-stationary SWAN points output for selected case

    xds_out_case   - swan case output (xarray.Dataset)
    p_export_case  - path for exporting figures
    '''

    if not op.isdir(p_export_case): os.makedirs(p_export_case)

    # iterate over points
    n_pts = len(xds_out_case.point)
    for i in range(n_pts):

        # get point variables
        xds_pt = xds_out_case.isel(point=i)

        hs = xds_pt.HS.values[:]
        tm = xds_pt.TM02.values[:]
        tp = xds_pt.RTP.values[:]
        dr = xds_pt.DIR.values[:]

        # plot and save figure series of each output point
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))

        ax1.plot(hs, '.', color = 'b', markersize=2, label="Hs [m]")
        #ax1.set_xlim([time[0][0], time[0][-1]])
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set_title('Significant Wave Height [m]', fontweight = 'bold')

        ax2.plot(tm, '.', color = 'b', markersize=2)
        #ax3.set_xlim([time[0][0], time[0][-1]])
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax2.set_title('Mean Period [s]', fontweight = 'bold')

        ax3.plot(tp, '.', color = 'b', markersize=2)
        #ax3.set_xlim([time[0][0], time[0][-1]])
        plt.setp(ax3.get_xticklabels(), visible=False)
        ax3.set_title('Peak Period [s]', fontweight = 'bold')

        ax4.plot(dr, '.', color = 'b', markersize=2)
        ax4.set_ylim([0, 360])
        #ax4.set_xlim([time[0][0], time[0][-1]])
        plt.setp(ax4.get_xticklabels(), visible=True)
        ax4.set_title('Wave direction [º]', fontweight = 'bold')

        # export fig
        p_ex = op.join(p_export_case, 'point_{0}.png'.format(i))
        fig.savefig(p_ex)

        # close fig 
        plt.close()

def plot_output_points(xds_out, p_export):
    '''
    Plots SWAN execution output table points time series

    xds_out    - swan points output (xarray.Dataset)
    p_export   - path for exporting figures
    '''

    # make export dir
    if not op.isdir(p_export): os.makedirs(p_export)

    for case_ix in xds_out.case.values[:]:

        # select case
        xds_out_case = xds_out.sel(case=case_ix)

        # output case subfolder
        case_id = '{0:04d}'.format(case_ix)
        p_export_case = op.join(p_export, case_id)

        # plot variable times
        plot_points_times(xds_out_case, p_export_case)

def plot_storm_track(lon0, lon1, lat0, lat1, pd_storm, p_export, np_shore=np.array([])):
    '''
    Plots SWAN execution output table points time series

    lon0, lon1  - longitude axes limits (lon0, lon1)
    lat0, lat1  - latitude axes limits (lat0, lat1)
    pd_storm    - storm track pandas.DataFrame (x0, y0, R as metadata)
    p_export    - path for exporting figure

    opt. args
    np_shore   - shoreline, np.array x = np_shore[:,0] y = np.shore[:,1]
    '''

    # make export dir
    if not op.isdir(p_export): os.makedirs(p_export)

    # get storm track data
    xt = pd_storm.lon
    yt = pd_storm.lat
    pmin = pd_storm.p0[0]
    vmean = pd_storm.vf[0]

    # get storm metadata
    x0 = pd_storm.x0
    y0 = pd_storm.y0
    R = pd_storm.R

    # circle angles
    ang = tuple(np.arange(0, 2*np.pi, 2*np.pi/1000))

    # circle coordinates
    x = R * np.cos(ang) + x0
    y = R * np.sin(ang) + y0

    # plot and save figure
    fig = plt.figure(figsize=(12, 12))

    # plot track
    plt.plot(xt, yt, 'o-', linewidth=2, color='purple', label='Great Circle')

    # plot small circle and center
    plt.plot(x, y, '-', linewidth=2, color='green', label='')
    plt.plot(x0, y0, '.', markersize=10, color='dodgerblue', label='')

    # plot shoreline
    if np_shore.any():
        x_shore = np_shore[:,0]
        y_shore = np_shore[:,1]
        plt.plot(x_shore, y_shore,'.', color='dimgray', markersize=3, label='')

    # plot parameters
    plt.axis('scaled')
    plt.xlim(lon0, lon1)
    plt.ylim(lat0, lat1)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Pmin: {0} hPa  /  V: {1} km/h'.format(pmin, vmean))
    plt.legend()

    # export fig
    p_save = op.join(p_export, 'track_coords.png')
    fig.savefig(p_save)

    # close fig
    plt.close()

