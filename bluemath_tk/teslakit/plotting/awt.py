#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
import itertools
import calendar
from datetime import datetime

# pip
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D

# teslakit
from ..util.operations import GetBestRowsCols
from ..util.time_operations import xds_reindex_daily as xr_daily
from ..util.time_operations import xds_common_dates_daily as xcd_daily
from ..util.time_operations import get_years_months_days
from ..kma import ClusterProbabilities
from .custom_colors import colors_awt
from .pcs import axplot_PC_hist, axplot_PCs_3D


# import constants
from .config import _faspect, _fsize, _fdpi

def axplot_AWT_2D(ax, var_2D, num_wts, id_wt, color_wt):
    'axes plot AWT variable (2D)'

    # plot 2D AWT
    ax.pcolormesh(
        var_2D,
        cmap='RdBu_r', shading='gouraud',
        vmin=-1.5, vmax=+1.5,
    )

    # title and axis labels/ticks
    ax.set_title(
        'WT #{0} --- {1} years'.format(id_wt, num_wts),
        {'fontsize': 14, 'fontweight':'bold'}
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('month', {'fontsize':8})
    ax.set_xlabel('lon', {'fontsize':8})

    # set WT color on axis frame
    plt.setp(ax.spines.values(), color=color_wt, linewidth=4)
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=color_wt)

def axplot_AWT_years(ax, dates_wt, bmus_wt, color_wt, xticks_clean=False,
                     ylab=None, xlims=None):
    'axes plot AWT dates'

    # date axis locator
    yloc5 = mdates.YearLocator(5)
    yloc1 = mdates.YearLocator(1)
    yfmt = mdates.DateFormatter('%Y')

    # get years string
    ys_str = np.array([str(d).split('-')[0] for d in dates_wt])

    # use a text bottom - top cycler
    text_cycler_va = itertools.cycle(['bottom', 'top'])
    text_cycler_ha = itertools.cycle(['left', 'right'])

    # plot AWT dates and bmus
    ax.plot(
        dates_wt, bmus_wt,
        marker='+',markersize=9, linestyle='', color=color_wt,
    )
    va = 'bottom'
    for tx,ty,tt in zip(dates_wt, bmus_wt, ys_str):
        ax.text(
            tx, ty, tt,
            {'fontsize':8},
            verticalalignment = next(text_cycler_va),
            horizontalalignment = next(text_cycler_ha),
            rotation=45,
        )

    # configure axis
    ax.set_yticks([])
    ax.xaxis.set_major_locator(yloc5)
    ax.xaxis.set_minor_locator(yloc1)
    ax.xaxis.set_major_formatter(yfmt)
    ax.grid(True, which='both', axis='x', linestyle='--', color='grey')
    ax.tick_params(axis='x', which='major', labelsize=8)

    # optional parameters
    if xticks_clean:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Year', {'fontsize':8})

    if ylab: ax.set_ylabel(ylab)

    if xlims is not None:
        ax.set_xlim(xlims[0], xlims[1])

def axplot_EOF_evolution(ax, years, EOF_evol):
    'axes plot EOFs evolution'

    # date axis locator
    yloc5 = mdates.YearLocator(5)
    yloc1 = mdates.YearLocator(1)
    yfmt = mdates.DateFormatter('%Y')

    # get years datetime
    ys_dt = np.array([datetime(y,1,1) for y in years])

    # plot EOF evolution 
    ax.plot(
        ys_dt, EOF_evol,
        linestyle='-', color='black',
    )

    # configure axis
    ax.set_xlim(ys_dt[0], ys_dt[-1])
    ax.xaxis.set_major_locator(yloc5)
    ax.xaxis.set_minor_locator(yloc1)
    ax.xaxis.set_major_formatter(yfmt)
    ax.grid(True, which='both', axis='x', linestyle='--', color='grey')
    ax.tick_params(axis='both', which='major', labelsize=8)

def axplot_EOF(ax, EOF_value, lon, ylbl, ttl):
    'axes plot EOFs evolution'

    # EOF pcolormesh 
    ax.pcolormesh(
        lon, range(12), np.transpose(EOF_value),
        cmap='RdBu_r', shading='gouraud',
        clim=2,
    )

    # axis and title
    ax.set_yticklabels(ylbl)
    ax.set_title(
        ttl,
        {'fontsize': 14, 'fontweight':'bold'}
    )
    ax.tick_params(axis='x', which='major', labelsize=8)
    ax.tick_params(axis='y', which='major', labelsize=10)


def Plot_AWT_Validation_Cluster(AWT_2D, AWT_num_wts, AWT_ID, AWT_dates,
                                AWT_bmus, AWT_PCs_fit, AWT_PCs_rnd, AWT_color,
                                show=True):


    # figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize))

    # layout
    gs = gridspec.GridSpec(4, 4, wspace=0.10, hspace=0.15)
    ax_AWT_2D = plt.subplot(gs[:2, :2])
    ax_PCs3D_fit = plt.subplot(gs[2, 0], projection='3d')
    ax_PCs3D_rnd = plt.subplot(gs[2, 1], projection='3d')
    ax_AWT_y = plt.subplot(gs[3, :])
    ax_PC1_hst_fit = plt.subplot(gs[0, 2])
    ax_PC1_hst_rnd = plt.subplot(gs[0, 3])
    ax_PC2_hst_fit = plt.subplot(gs[1, 2])
    ax_PC2_hst_rnd = plt.subplot(gs[1, 3])
    ax_PC3_hst_fit = plt.subplot(gs[2, 2])
    ax_PC3_hst_rnd = plt.subplot(gs[2, 3])

    # plot AWT 2D
    axplot_AWT_2D(ax_AWT_2D, AWT_2D, AWT_num_wts, AWT_ID, AWT_color)

    # plot AWT years
    axplot_AWT_years(ax_AWT_y, AWT_dates, AWT_bmus, AWT_color)

    # compare PCs fit - sim with 3D plot
    axplot_PCs_3D(ax_PCs3D_fit, AWT_PCs_fit,  AWT_color, ttl='PCs fit')
    axplot_PCs_3D(ax_PCs3D_rnd, AWT_PCs_rnd,  AWT_color, ttl='PCs sim')

    # compare PC1 histograms
    axplot_PC_hist(ax_PC1_hst_fit, AWT_PCs_fit[:,0], AWT_color)
    axplot_PC_hist(ax_PC1_hst_rnd, AWT_PCs_rnd[:,0], AWT_color, ylab='PC1')

    axplot_PC_hist(ax_PC2_hst_fit, AWT_PCs_fit[:,1], AWT_color)
    axplot_PC_hist(ax_PC2_hst_rnd, AWT_PCs_rnd[:,1], AWT_color, ylab='PC2')

    axplot_PC_hist(ax_PC3_hst_fit, AWT_PCs_fit[:,2], AWT_color)
    axplot_PC_hist(ax_PC3_hst_rnd, AWT_PCs_rnd[:,2], AWT_color, ylab='PC3')

    # show
    if show: plt.show()
    return fig

def Plot_AWTs_Validation(bmus, dates, Km, n_clusters, lon, d_PCs_fit,
                         d_PCs_rnd, show=True):
    '''
    Plot Annual Weather Types Validation

    bmus, dates, Km, n_clusters, lon - from KMA_simple()
    d_PCs_fit, d_PCs_rnd - historical and simulated PCs by WT
    '''

    # get cluster colors
    cs_awt = colors_awt()

    # each cluster has a figure
    l_figs = []
    for ic in range(n_clusters):

        # get cluster data
        id_AWT = ic + 1           # cluster ID
        index = np.where(bmus==ic)[0][:]
        dates_AWT = dates[index]  # cluster dates
        bmus_AWT = bmus[index]    # cluster bmus
        var_AWT = Km[ic,:]
        var_AWT_2D = var_AWT.reshape(-1, len(lon))
        num_WTs = len(index)      # number of cluster ocurrences
        clr = cs_awt[ic]          # cluster color
        PCs_fit = d_PCs_fit['{0}'.format(id_AWT)]
        PCs_rnd = d_PCs_rnd['{0}'.format(id_AWT)]

        # plot cluster figure
        fig = Plot_AWT_Validation_Cluster(
            var_AWT_2D, num_WTs, id_AWT,
            dates_AWT, bmus_AWT,
            PCs_fit, PCs_rnd,
            clr, show=show)

        l_figs.append(fig)

    return l_figs

def Plot_AWTs(bmus, Km, n_clusters, lon, show=True):
    '''
    Plot Annual Weather Types

    bmus, Km, n_clusters, lon - from KMA_simple()
    '''

    # get number of rows and cols for gridplot 
    n_cols, n_rows = GetBestRowsCols(n_clusters)

    # get cluster colors
    cs_awt = colors_awt()

    # plot figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize))

    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.10, hspace=0.15)
    gr, gc = 0, 0

    for ic in range(n_clusters):

        id_AWT = ic + 1           # cluster ID
        index = np.where(bmus==ic)[0][:]
        var_AWT = Km[ic,:]
        var_AWT_2D = var_AWT.reshape(-1, len(lon))
        num_WTs = len(index)
        clr = cs_awt[ic]          # cluster color

        # AWT var 2D 
        ax = plt.subplot(gs[gr, gc])
        axplot_AWT_2D(ax, var_AWT_2D, num_WTs, id_AWT, clr)

        gc += 1
        if gc >= n_cols:
            gc = 0
            gr += 1

    # show and return figure
    if show: plt.show()
    return fig

def Plot_AWTs_Dates(bmus, dates, n_clusters, show=True):
    '''
    Plot Annual Weather Types dates

    bmus, dates, n_clusters - from KMA_simple()
    '''

    # get cluster colors
    cs_awt = colors_awt()

    # plot figure
    fig, axs = plt.subplots(nrows=n_clusters, figsize=(_faspect*_fsize, _fsize))

    # each cluster has a figure
    for ic in range(n_clusters):

        id_AWT = ic + 1           # cluster ID
        index = np.where(bmus==ic)[0][:]
        dates_AWT = dates[index]  # cluster dates
        bmus_AWT = bmus[index]    # cluster bmus
        clr = cs_awt[ic]          # cluster color

        ylabel = "WT #{0}".format(id_AWT)
        xlims = [dates[0].astype('datetime64[Y]')-np.timedelta64(3, 'Y'), dates[-1].astype('datetime64[Y]')+np.timedelta64(3, 'Y')]

        xaxis_clean=True
        if ic == n_clusters-1:
            xaxis_clean=False

        # axs plot
        axplot_AWT_years(
            axs[ic], dates_AWT, bmus_AWT,
            clr, xaxis_clean, ylabel, xlims
        )

    # show and return figure
    if show: plt.show()
    return fig

def Plot_AWTs_EOFs(PCs, EOFs, variance, time, lon, n_plot, show=True):
    '''
    Plot annual EOFs for PCA_LatitudeAverage predictor

    PCs, EOFs, variance, time, lon - from PCA_LatitudeAverage()
    n_plot                         - number of EOFs plotted
    '''

    # transpose
    EOFs = np.transpose(EOFs)
    PCs = np.transpose(PCs)

    # get start and end month
    ys, ms, _ = get_years_months_days(time)

    # PCA latavg metadata
    y1 = ys[0]
    y2 = ys[-1]
    m1 = ms[0]
    m2 = ms[-1]

    # mesh data
    len_x = len(lon)

    # time data
    years = range(y1, y2+1)
    l_months = [calendar.month_name[x] for x in range(1,13)]
    ylbl = l_months[m1-1:] + l_months[:m2]

    # percentage of variance each field explains
    n_percent = variance / np.sum(variance)

    l_figs = []
    for it in range(n_plot):

        # map of the spatial field
        spatial_fields = EOFs[:,it]*np.sqrt(variance[it])

        # reshape from vector to matrix with separated months
        C = np.reshape(
            spatial_fields[:len_x*12], (12, len_x)
        ).transpose()

        # plot figure
        fig = plt.figure(figsize=(_faspect*_fsize, _fsize))

        # layout
        gs = gridspec.GridSpec(4, 4, wspace=0.10, hspace=0.2)
        ax_EOF = plt.subplot(gs[:3, :])
        ax_evol = plt.subplot(gs[3, :])

        # EOF pcolormesh
        ttl = 'EOF #{0}  ---  {1:.2f}%'.format(it+1, n_percent[it]*100)
        axplot_EOF(ax_EOF, C, lon, ylbl, ttl)

        # time series EOF evolution
        evol =  PCs[it,:]/np.sqrt(variance[it])
        axplot_EOF_evolution(ax_evol, years, evol)

        l_figs.append(fig)

        # show
        if show: plt.show()

    return l_figs

