#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

from .common import GetBestRowsCols, calc_quiver

# import constants
from .config import _faspect, _fsize, _fdpi


def axplot_var_map(ax, XX, YY, vv, vd,
                   quiver=True, np_shore=np.array([]),
                   vmin=None, vmax=None):
    'plot 2D map with variable data'

    # parameters
    cmap = plt.get_cmap('seismic')

    # cplot v lims
    if vmin == None: vmin = vv.min()
    if vmax == None: vmax = vv.max()


    # plot variable 2D map
    pm = ax.pcolormesh(
        XX, YY, vv,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
    )

    # optional quiver
    if quiver:
        x_q, y_q, var_q, u, v = calc_quiver(XX[0,:], YY[:,0], vv, vd, size=12)
        ax.quiver(
            x_q, y_q, -u*var_q, -v*var_q,
            width=0.003,
            #scale = 0.5,
            scale_units='inches',
        )

    # optional shoreline
    if np_shore.any():
        xs = np_shore[:,0]
        ys = np_shore[:,1]
        ax.plot(
            np_shore[:,0], np_shore[:,1],
            '.', color='dimgray',
            markersize=3, label=''
        )

        # fix axes
        ax.set_xlim(XX[0,0], XX[0,-1])
        ax.set_ylim(YY[0,0], YY[-1,0])

    # return last pcolormesh
    return pm

def scatter_maps(xds_out, var_list=[], n_cases=None, quiver=True, var_limits={},
                 np_shore=np.array([])):
    '''
    scatter plots stationary SWAN execution output for first "n_cases"

    xds_out    - swan stationary output (xarray.Dataset)

    opt. args
    var_list   - swan output variables ['Hsig', 'Tm02', 'Tpsmoo'] (default all vars)
    n_cases    - number of cases to plot (default all cases)
    quiver     - True for adding directional quiver plot
    var_limits  - dictionary with variable names as keys and a (min, max) tuple for limits
    np_shore   - shoreline, np.array x = np_shore[:,0] y = np.shore[:,1]
    '''

    # TODO improve xticks, yticks 
    # TODO legend box with info ?
    # TODO diff colormap with diff variables

    # number of cases
    if n_cases == None:
        n_cases = len(xds_out.case.values)

    # get number of rows and cols for gridplot
    n_cols, n_rows = GetBestRowsCols(n_cases)

    # allowed vars
    avs =['Hsig', 'Tm02', 'Dspr'] #, 'TPsmoo']

    # variable list 
    if var_list == []:
        var_list = dict(xds_out.variables).keys()
        var_list = [vn for vn in var_list if vn in avs]  # filter only allowed

    # mesh data to plot
    if 'lon' in xds_out.dims:
        X, Y = xds_out.lon, xds_out.lat
        xlab, ylab = 'Longitude (º)', 'Latitude (º)'
    else:
        X, Y = xds_out.X, xds_out.Y
        xlab, ylab = 'X', 'Y'
    XX, YY = np.meshgrid(X, Y)

    # iterate: one figure for each variable
    l_figs = []
    for vn in var_list:

        # plot figure
        fig, (axs) = plt.subplots(
            nrows=n_rows, ncols=n_cols,
            sharex=True, sharey=True,
            constrained_layout=False,
            figsize=(_fsize*_faspect, _fsize*_faspect),
        )

        fig.subplots_adjust(wspace=0, hspace=0)

        # common vlimits
        vmin = xds_out[vn].min()
        vmax = xds_out[vn].max()

        # optional vlimits
        if vn in var_limits.keys():
            vmin = var_limits[vn][0]
            vmax = var_limits[vn][1]

        # plot cases output
        gr, gc = 0, 0
        for ix in range(n_cases):
            out_case = xds_out.isel(case=ix)

            # variable and direction 
            vv = out_case[vn].values[:].T
            vd = out_case['Dir'].values[:].T

            # plot variable times
            ax = axs[gr, gc]
            pm = axplot_var_map(
                ax, XX, YY, vv, vd,
                quiver=quiver, np_shore=np_shore,
                vmin=vmin, vmax=vmax
            )

            # row,col counter
            gc += 1
            if gc >= n_cols:
                gc = 0
                gr += 1


        # add custom common axis labels
        fig.text(0.5, 0.04, xlab, ha='center')
        fig.text(0.04, 0.5, ylab, va='center', rotation='vertical')

        # add custom common colorbar
        cbar_ax = fig.add_axes([0.93, 0.11, 0.02, 0.77])
        fig.colorbar(pm, cax=cbar_ax)
        cbar_ax.set_ylabel(vn)


        l_figs.append(fig)

    return l_figs

