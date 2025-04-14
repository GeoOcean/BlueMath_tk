#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
from datetime import datetime, timedelta

# pip
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase

# teslakit
from ..plotting.custom_colors import GetClusterColors

# import constants
from ..plotting.config import _faspect, _fsize, _fdpi

# TODO: Plot_Complete, add xticks at last axs to show years

def axplot_bmus(ax, bmus, num_clusters, lab):
    'axes plot bmus series using colors'

    # colors
    wt_colors = GetClusterColors(num_clusters)

    # prepare nans as white 
    bmus = bmus.fillna(num_clusters+1)
    wt_colors = np.vstack([wt_colors, [0,0,0]])

    # bmus colors
    vp = wt_colors[bmus.astype(int)-1]

    # listed colormap
    ccmap = mcolors.ListedColormap(vp)

    # color series
    cbar_x = ColorbarBase(
        ax, cmap = ccmap, orientation='horizontal',
        norm = mcolors.Normalize(vmin=0, vmax=num_clusters),
    )
    cbar_x.set_ticks([])

    # customize axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(lab, rotation=0, fontweight='bold', labelpad=35)

def axplot_series(ax, vv, ls, lc, lab, xticks=False):
    'axes plot variables series'

    # find start and end index
    nna = np.where(~np.isnan(vv))[0]
    x_0, x_1 = nna[0], nna[-1]

    # plot series
    ax.plot(nna, vv[nna], linestyle=ls, linewidth=0.5, color=lc)

    # customize axes
    ax.set_xlim(0, len(vv))

    # TODO add optional xticks option
    if not xticks:
        ax.set_xticks([])

    ax.yaxis.tick_right()
    ax.tick_params(axis='both', which='both', labelsize=7)
    ax.set_ylabel(lab, rotation=0, fontweight='bold', labelpad=35)

def Plot_Complete(xds, show=True):
    '''
    Plot complete data variables inside xds

    xds - xarray.Dataset: (time) AWT, MJO, DWT, Hs, Tp, Dir, SS, AT, MMSL, TWL
    '''
    # TODO: auto parameters inside xds

    # parameters
    n_cs_AWT = 6
    n_cs_MJO = 25
    n_cs_DWT = 42


    # plot figure
    fig, axs = plt.subplots(
        10, 1,
        figsize=(_faspect*_fsize, _fsize),
    )

    # AWT
    axplot_bmus(axs[0], xds.AWT, n_cs_AWT, 'AWT')

    # MJO
    axplot_bmus(axs[1], xds.MJO, n_cs_MJO, 'MJO')

    # DWT
    axplot_bmus(axs[2], xds.DWT, n_cs_DWT, 'DWT')

    # HS
    axplot_series(axs[3], xds.Hs, 'solid', 'cyan', 'Hs(m)')

    # TP
    axplot_series(axs[4], xds.Tp, 'solid', 'red', 'Tp(s)')

    # DIR
    axplot_series(axs[5], xds.Dir, 'dotted', 'black', 'Dir(º)')

    # SS
    if 'SS' in xds.variables:
        axplot_series(axs[6], xds.SS, 'solid', 'orange', 'SS(m)')
    else:
        axs[6].set_xticks([])
        axs[6].set_yticks([])
        axs[6].set_ylabel('SS(m)', rotation=0, fontweight='bold', labelpad=35)

    # AT
    axplot_series(axs[7], xds.AT, 'solid', 'purple', 'AT(m)')

    # MMSL
    axplot_series(axs[8], xds.MMSL, 'solid', 'green', 'MMSL(m)')

    # TWL
    axplot_series(axs[9], xds.TWL, 'solid', 'blue', 'TWL(m)')


    # suptitle
    fig.suptitle('Complete Data', fontweight='bold', fontsize=12)

    # reduce first 3 subplots
    gs = gridspec.GridSpec(10, 1, height_ratios = [1,1,1,3,3,3,3,3,3,3])
    for i in range(10):
        axs[i].set_position(gs[i].get_position(fig))

    # show and return figure
    if show: plt.show()
    return fig


def axplot_compare_histograms(ax, var_1, var_2, n_bins=40,
                      label_1='Historical', label_2='Simulation', ttl='',
                      alpha_1=0.9, alpha_2=0.7,
                      color_1='white', color_2='skyblue',
                      density=False):
    'axes plot histogram comparison between fit-sim variables'

    (_, bins, _) = ax.hist(var_1, n_bins, weights=np.ones(len(var_1)) / len(var_1),
            alpha=alpha_1, color=color_1, ec='k', label = label_1, density=density)

    ax.hist(var_2, bins=bins, weights=np.ones(len(var_2)) / len(var_2),
            alpha=alpha_2, color=color_2, ec='k', label = label_2, density=density)

    # customize axes
    ax.legend(prop={'size':8})
    ax.set_title(ttl)

    if density:
        ax.set_ylabel('Probability')

def Plot_FitSim_Histograms(data_fit, data_sim, vns, n_bins=40,
                           color_1='white', color_2='skyblue',
                           alpha_1=0.7, alpha_2=0.4,
                           label_1='Historical', label_2 = 'Simulation',
                           gs_1 = 1, gs_2 = None, supt=False, vns_lims={},
                           density=False, show=True):
    'Plots fit vs sim histograms for variables "vns"'

    # grid spec default number of columns
    if gs_2 == None: gs_2 = len(vns)

    # plot figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize*gs_1/2.3))

    # grid spec
    gs = gridspec.GridSpec(gs_1, gs_2)  #, wspace=0.0, hspace=0.0)

    # variables
    cr, cc = 0, 0
    for c, vn in enumerate(vns):

        dh = data_fit[vn].values[:]; dh = dh[~np.isnan(dh)]
        ds = data_sim[vn].values[:]; ds = ds[~np.isnan(ds)]

        # variable max and min optional limits
        if vn in vns_lims.keys():
            vl_1, vl_2 = vns_lims[vn]
            dh = dh[np.where((dh >= vl_1) & (dh <= vl_2))[0]]
            ds = ds[np.where((ds >= vl_1) &( ds <= vl_2))[0]]

        ax = plt.subplot(gs[cr, cc])
        axplot_compare_histograms(
            ax, dh, ds, ttl=vn, density=density, n_bins=n_bins,
            color_1=color_1, color_2=color_2,
            alpha_1=alpha_1, alpha_2=alpha_2,
            label_1=label_1, label_2=label_2,
        )

        # grid spec counter
        cc+=1
        if cc >= gs_2:
            cc=0
            cr+=1

    # fig suptitle
    if supt:
        fig.suptitle(
            '{0} - {1} Comparison: {2}'.format(label_1, label_2, ', '.join(vns)),
            fontsize=13, fontweight = 'bold',
        )

    # show and return figure
    if show: plt.show()
    return fig

def Plot_LevelVariables_Histograms(data_hist, data_sim, label_1='Historical', label_2 = 'Simulation', show=True):
    'Plots histogram comparison (historical - simulation) for level related variables'

    # Compare histograms 
    Plot_FitSim_Histograms(
        data_hist, data_sim, ['level', 'AT', 'MMSL', 'TWL'],
        color_1='white', color_2='skyblue', alpha_1=0.9, alpha_2=0.7,
        label_1= label_1, label_2 = label_2,
        gs_1 = 2, gs_2 = 2,
        density=False, show=True
    )

