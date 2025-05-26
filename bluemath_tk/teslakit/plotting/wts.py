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
from matplotlib.ticker import MaxNLocator

# teslakit
from ..util.operations import GetBestRowsCols
from .custom_colors import GetClusterColors
from ..kma import ClusterProbabilities, ChangeProbabilities
#from .outputs import axplot_compare_histograms


# import constants
from .config import _faspect, _fsize, _fdpi


def GenOneYearDaily(yy=1981, month_ini=1):
    'returns one generic year in a list of datetimes. Daily resolution'

    dp1 = datetime(yy, month_ini, 1)
    dp2 = dp1+timedelta(days=365)

    return [dp1 + timedelta(days=i) for i in range((dp2-dp1).days)]

def Generate_PerpYear_Matrix(num_clusters, bmus_values, bmus_dates,
                             num_sim=1, month_ini=1):
    '''
    Calculates and returns matrix for stacked bar plotting

    bmus_dates - datetime.datetime (only works if daily resolution)
    bmus_values has to be 2D (time, nsim)
    '''

    # generate perpetual year list
    list_pyear = GenOneYearDaily(month_ini=month_ini)

    # generate aux arrays
    m_plot = np.zeros((num_clusters, len(list_pyear))) * np.nan
    bmus_dates_months = np.array([d.month for d in bmus_dates])
    bmus_dates_days = np.array([d.day for d in bmus_dates])

    # sort data
    for i, dpy in enumerate(list_pyear):
        _, s = np.where(
            [(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)]
        )
        b = bmus_values[s,:]
        b = b.flatten()

        for j in range(num_clusters):
            _, bb = np.where([(j+1 == b)])  # j+1 starts at 1 bmus value!

            m_plot[j,i] = float(len(bb)/float(num_sim))/len(s)

    return m_plot


def axplot_PerpYear(ax, num_clusters, bmus_values, bmus_dates, num_sim, month_ini):
    'axes plot bmus perpetual year'

    # get cluster colors for stacked bar plot
    np_colors_int = GetClusterColors(num_clusters)

    # generate dateticks
    x_val = GenOneYearDaily(month_ini = month_ini)

    # generate plot matrix
    m_plot = Generate_PerpYear_Matrix(
        num_clusters, bmus_values, bmus_dates,
        num_sim = num_sim, month_ini = month_ini)

    # plot stacked bars
    bottom_val = np.zeros(m_plot[1,:].shape)
    for r in range(num_clusters):
        row_val = m_plot[r,:]
        ax.bar(
            x_val, row_val, bottom = bottom_val,
            width = 1, color = np.array([np_colors_int[r]])
        )

        # store bottom
        bottom_val += row_val

    # customize  axis
    months = mdates.MonthLocator()
    monthsFmt = mdates.DateFormatter('%b')

    ax.set_xlim(x_val[0], x_val[-1])
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.set_ylim(0, 1)
    ax.set_ylabel('')

def axplot_ChangeProbs(ax, change_probs, wt_colors,
                       ttl = '', vmin = 0, vmax = 1,
                       cmap = 'Blues', cbar_ttl = ''):
    'axes plot cluster change probabilities'

    num_clusters = change_probs.shape[0]

    # clsuter transition plot
    pc = ax.pcolor(
        change_probs,
        cmap=cmap, vmin=vmin, vmax=vmax,
    )

    # add colorbar
    cbar = plt.colorbar(pc, ax=ax)
    cbar.ax.tick_params(labelsize=8)
    if vmin != 0 or vmax !=1:
        cbar.set_ticks(np.linspace(vmin, vmax, 6))
    cbar.ax.set_ylabel(cbar_ttl, rotation=270, labelpad=20)

    # customize axes
    ax.set_xticks(np.arange(num_clusters)+0.5)
    ax.set_yticks(np.arange(num_clusters)+0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(ttl, {'fontsize':10, 'fontweight':'bold'})

    # add custom color axis
    ccmap = mcolors.ListedColormap([tuple(r) for r in wt_colors])

    # custom color axis positions
    ax_pos = ax.get_position()
    cax_x_pos = [ax_pos.x0, ax_pos.y0-0.03, ax_pos.width, 0.02]
    cax_y_pos = [ax_pos.x0-0.03/_faspect, ax_pos.y0, 0.02/_faspect, ax_pos.height]

    # custom color axis X
    cax_x = ax.figure.add_axes(cax_x_pos)
    cbar_x = ColorbarBase(
        cax_x, cmap = ccmap, orientation='horizontal',
        norm = mcolors.Normalize(vmin=0, vmax=num_clusters),
    )
    cbar_x.set_ticks([])

    # custom color axis Y
    cax_y= ax.figure.add_axes(cax_y_pos)
    cbar_y = ColorbarBase(
        cax_y, cmap = ccmap, orientation='vertical',
        norm = mcolors.Normalize(vmin=0, vmax=num_clusters),
    )
    cbar_y.set_ticks([])

def axplot_Transition(ax, probs_change_fit, probs_change_sim,
                      ttl='', color='black'):
    'axes plot for scatter historical - simulation transition matrix comparison'

    # clsuter transition fit vs. sim scatter plot
    pc = ax.scatter(
        probs_change_fit, probs_change_sim,
        color=color, s=5, zorder=2,
    )

    # get max value for axis lim
    axlim = np.ceil(10*np.max(probs_change_sim[:]))/10

    # customize axes
    ax.plot(([0,0],[1,1]), linestyle='dashed', linewidth=0.5, color='grey',
            zorder=1)
    ax.set_ylabel('Simulated', {'fontsize':8})
    ax.set_xlabel('Historical', {'fontsize':8})
    ax.grid(True, which='major', axis='both', linestyle='--', color='grey')
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.set_xlim([0, axlim])
    ax.set_ylim([0, axlim])
    ax.set_title(ttl, {'fontsize':10, 'fontweight':'bold'})

def axplot_ClusterProbs(ax, cluster_probs_fit, cluster_probs_sim, wt_colors,
                       ttl=''):
    'axes plot for scatter historical - simulation cluster probs comparison'

    # clsuter transition plot
    for cf, cs, wc in zip(cluster_probs_fit, cluster_probs_sim, wt_colors):
        ax.plot(cf, cs, marker='.', markersize=8, color=wc, zorder=2)

    # get max value for axis lim
    axlim = np.ceil(10*np.max(cluster_probs_sim[:]))/10

    # customize axes
    ax.plot(([0,0],[1,1]), linestyle='dashed', linewidth=0.5, color='grey',
            zorder=1)
    ax.set_ylabel('Simulated', {'fontsize':8})
    ax.set_xlabel('Historical', {'fontsize':8})
    ax.grid(True, which='major', axis='both', linestyle='--', color='grey')
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.set_xlim([0, axlim])
    ax.set_ylim([0, axlim])
    ax.set_title(ttl, {'fontsize':10, 'fontweight':'bold'})

def axplot_WT_Probs(ax, wt_probs,
                     ttl = '', vmin = 0, vmax = 0.1,
                     cmap = 'Blues', caxis='black'):
    'axes plot WT cluster probabilities'

    # clsuter transition plot
    pc = ax.pcolor(
        np.flipud(wt_probs),
        cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolors='k',
    )

    # customize axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(ttl, {'fontsize':10, 'fontweight':'bold'})

    # axis color
    plt.setp(ax.spines.values(), color=caxis)
    plt.setp(
        [ax.get_xticklines(), ax.get_yticklines()],
        color=caxis,
    )

    # axis linewidth
    if caxis != 'black':
        plt.setp(ax.spines.values(), linewidth=3)

    return pc

def axplot_WT_Hist(ax, bmus, n_clusters, ttl=''):
    'axes plot WT cluster count histogram'

    # cluster transition plot
    ax.hist(
        bmus,
        bins = np.arange(1, n_clusters+2),
        edgecolor='k'
    )

    # customize axes
    #ax.grid('y')

    ax.set_xticks(np.arange(1,n_clusters+1)+0.5)
    ax.set_xticklabels(np.arange(1,n_clusters+1))
    ax.set_xlim([1, n_clusters+1])
    ax.tick_params(axis='both', which='major', labelsize=6)

    ax.set_title(ttl, {'fontsize':10, 'fontweight':'bold'})


def Plot_Compare_PerpYear(num_clusters,
                          bmus_values_sim, bmus_dates_sim,
                          bmus_values_hist, bmus_dates_hist,
                          n_sim = 1, month_ini = 1, show=True):
    '''
    Plot simulated - historical bmus comparison in a perpetual year

    bmus_dates requires 1 day resolution time
    bmus_values set min value has to be 1 (not 0)
    '''

    # check dates have 1 day time resolution 
    td_h = bmus_dates_hist[1] - bmus_dates_hist[0]
    td_s = bmus_dates_sim[1] - bmus_dates_sim[0]
    
    if isinstance(td_h, np.int32):
        return
    if td_h.days != 1 or td_s.days != 1:
        print('PerpetualYear bmus comparison skipped.')
        print('timedelta (days): Hist - {0}, Sim - {1})'.format(
            td_h, td_s.days))
        return

    # plot figure
    fig, (ax_hist, ax_sim) = plt.subplots(2,1, figsize=(_faspect*_fsize, _fsize))

    # historical perpetual year
    axplot_PerpYear(
        ax_hist, num_clusters, bmus_values_hist, bmus_dates_hist,
        num_sim = 1, month_ini = month_ini,
    )
    ax_hist.set_title('Historical')

    # simulated perpetual year
    axplot_PerpYear(
        ax_sim, num_clusters, bmus_values_sim, bmus_dates_sim,
        num_sim = n_sim, month_ini = month_ini,
    )
    ax_sim.set_title('Simulation')

    # add custom colorbar
    np_colors_int = GetClusterColors(num_clusters)
    ccmap = mcolors.ListedColormap(
        [tuple(r) for r in np_colors_int]
    )
    cax = fig.add_axes([0.92, 0.125, 0.025, 0.755])
    cbar = ColorbarBase(
        cax, cmap=ccmap,
        norm = mcolors.Normalize(vmin=0, vmax=num_clusters),
        ticks = np.arange(num_clusters) + 0.5,
    )
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticklabels(range(1, num_clusters + 1))

    # text
    fig.suptitle('Perpetual Year', fontweight='bold', fontsize=12)

    # show and return figure
    if show: plt.show()
    return fig

def Plot_Compare_Transitions(num_clusters, bmus_values_hist, bmus_values_sim,
                             sttl=None, show=True):
    '''
    Plot many probabilities transition comparisons between 2 bmus series

    bmus_values_hist, bmus_values_sim (series x n_sim) - bmus series to compare
    num_clusters  - total number of clusters at series
    '''

    # if n_sim > 1 lets safe-join simulated bmus for probs calcs.
    n_sim = bmus_values_sim.shape[1]
    if n_sim > 1:
        # add nan splitter and then flatten
        bmus_values_sim = np.append(
            bmus_values_sim, np.zeros((1, n_sim))*np.nan,
            axis = 0,
        )
        bmus_values_sim = bmus_values_sim.flatten('F')

    # get cluster colors
    wt_colors = GetClusterColors(num_clusters)
    wt_set = np.arange(num_clusters) + 1

    # cluster probabilities
    probs_c_fit = ClusterProbabilities(bmus_values_hist, wt_set)
    probs_c_sim = ClusterProbabilities(bmus_values_sim, wt_set)

    # cluster change probabilities
    _, chprobs_fit = ChangeProbabilities(bmus_values_hist, wt_set)
    _, chprobs_rnd = ChangeProbabilities(bmus_values_sim, wt_set)
    chprobs_dif = chprobs_rnd - chprobs_fit

    # figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize))

    # layout
    gs = gridspec.GridSpec(2, 6, wspace=0.60, hspace=0.35)
    ax_chpr_fit = plt.subplot(gs[0, :2])
    ax_chpr_rnd = plt.subplot(gs[0, 2:4])
    ax_chpr_dif = plt.subplot(gs[0, 4:])
    ax_scat_tra = plt.subplot(gs[1, 1:3])
    ax_scat_pbs = plt.subplot(gs[1, 3:5])

    # cluster change probs axes 
    axplot_ChangeProbs(ax_chpr_fit, chprobs_fit, wt_colors,
                       'Historical WT Transition Probabilities')
    axplot_ChangeProbs(ax_chpr_rnd, chprobs_rnd, wt_colors,
                       'Simulation WT Transition Probabilities')
    axplot_ChangeProbs(ax_chpr_dif, chprobs_dif, wt_colors,
                       'Change in WT Transition Probabilities',
                       vmin=-0.05, vmax=0.05, cmap='RdBu_r',
                       cbar_ttl='Simulation - Historical',
                      )

    # scatter plot transitions
    axplot_Transition(ax_scat_tra, chprobs_fit, chprobs_rnd,
                      'Sim. vs. Hist. WT Transition Probabilities'
                     )

    # scatter plot cluster probs
    axplot_ClusterProbs(ax_scat_pbs, probs_c_fit, probs_c_sim, wt_colors,
                       'Sim. vs. Hist. WT Probabilities',
                       )

    # suptitle
    if sttl != None:
        fig.suptitle(sttl, fontweight='bold', fontsize=12)

    # show and return figure
    if show: plt.show()
    return fig

def Plot_Probs_WT_WT(series_1, series_2, n_clusters_1, n_clusters_2, ttl='',
                     wt_colors=False, show=True):
    '''
    Plot WTs_1 / WTs_2 probabilities

    both categories series should start at 0
    '''

    # set of daily weather types
    set_2 = np.arange(n_clusters_2)

    # dailt weather types matrix rows and cols
    n_rows, n_cols = GetBestRowsCols(n_clusters_2)

    # get cluster colors
    cs_wt =  GetClusterColors(n_clusters_1)

    # plot figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize/3))
    gs = gridspec.GridSpec(1, n_clusters_1, wspace=0.10, hspace=0.15)

    for ic in range(n_clusters_1):

        # select DWT bmus at current AWT indexes
        index_1 = np.where(series_1==ic)[0][:]
        sel_2 = series_2[index_1]

        # get DWT cluster probabilities
        cps = ClusterProbabilities(sel_2, set_2)
        C_T = np.reshape(cps, (n_rows, n_cols))

        # axis colors
        if wt_colors:
            caxis = cs_wt[ic]
        else:
            caxis = 'black'

        # plot axes
        ax = plt.subplot(gs[0, ic])
        axplot_WT_Probs(
            ax, C_T,
            ttl = 'WT {0}'.format(ic+1),
            cmap = 'Reds', caxis = caxis,
        )
        ax.set_aspect('equal')

    # add fig title
    fig.suptitle(ttl, fontsize=14, fontweight='bold')

    # show and return figure
    if show: plt.show()
    return fig

def Plot_Probs_WT_WT_anomaly(series_1, series_2, n_clusters_1, n_clusters_2, ttl='',
                     wt_colors=False, show=True):
    '''
    Plot WTs_1 / WTs_2 probabilities

    both categories series should start at 0
    '''

    # set of daily weather types
    set_2 = np.arange(n_clusters_2)

    # dailt weather types matrix rows and cols
    n_rows, n_cols = GetBestRowsCols(n_clusters_2)

    # get cluster colors
    cs_wt =  GetClusterColors(n_clusters_1)

    # plot figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize/3))
    gs = gridspec.GridSpec(1, n_clusters_1, wspace=0.10, hspace=0.15)

    #----------------------
    # best rows cols combination
    n_rows, n_cols = GetBestRowsCols(n_clusters_2)

    c_T = ClusterProbabilities(series_2, set_2)
    C_T_mean = np.reshape(c_T, (n_rows, n_cols))
    #----------------------

    for ic in range(n_clusters_1):

        # select DWT bmus at current AWT indexes
        index_1 = np.where(series_1==ic)[0][:]
        sel_2 = series_2[index_1]

        # get DWT cluster probabilities
        cps = ClusterProbabilities(sel_2, set_2)
        C_T = np.reshape(cps, (n_rows, n_cols))

        #-----------
        C_T = C_T-C_T_mean
        #-----------

        # axis colors
        if wt_colors:
            caxis = cs_wt[ic]
        else:
            caxis = 'black'

        # plot axes
        ax = plt.subplot(gs[0, ic])
        axplot_WT_Probs(
            ax, C_T,
            ttl = 'WT {0}'.format(ic+1),vmin=-0.1,vmax=0.1,
            cmap = 'seismic', caxis = caxis,
        )
        ax.set_aspect('equal')

    # add fig title
    fig.suptitle(ttl, fontsize=14, fontweight='bold')

    # show and return figure
    if show: plt.show()
    return fig

def Plot_Compare_Persistences(n_clusters, pers_hist, pers_sim,
                              color_1='plum', color_2='dodgerblue',
                              alpha_1=0.7, alpha_2=0.4,
                              label_1='Historical', label_2 = 'Simulation',
                              show=True):
    '''
    Plot historical - simulation WT persistences comparison

    pers_sim, pers_hist - dictionary {WT: ocurrences durations list}
    '''

    # dailt weather types matrix rows and cols
    n_rows, n_cols = GetBestRowsCols(n_clusters)

    # plot figure
    fig = plt.figure(figsize=(_fsize*n_cols/2, _fsize*n_rows/2.3))
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.2, hspace=0.2)

    for c in range(n_clusters):

        # select wt data
        wt = c+1

        # persistences
        p_h = pers_hist[wt]
        p_s = pers_sim[wt]

        # common set
        ss = np.array(sorted(list(set(p_h).union(set(p_s)))))+0.5
        # filter when only 1 bin 
        if len(ss) == 1: continue

        # compare histograms
        ax = fig.add_subplot(gs[c])
        axplot_compare_histograms(
            ax, sorted(p_h), sorted(p_s), ttl='WT: {0}'.format(wt),
            density=True,
            n_bins=ss,
            color_1=color_1, color_2=color_2,
            alpha_1=alpha_1, alpha_2=alpha_2,
            label_1=label_1, label_2=label_2,
        )

        # customize axis
        ax.set_ylabel('')
        ax.legend(prop={'size':8})
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # suptitle
    fig.suptitle('WTs Persistences (days)', fontweight='bold', fontsize=12)

    # show and return figure
    if show: plt.show()
    return fig


# TODO: following functions are not finished / tested

def Plot_PerpYear(bmus_values, bmus_dates, num_clusters, num_sim=1,
                  show=True):
    'Plots ARL bmus simulated in a perpetual_year stacked bar chart'

    # TODO: UPDATE


    # get cluster colors for stacked bar plot
    np_colors_int = GetClusterColors(num_clusters)

    # generate plot matrix
    m_plot = Generate_PerpYear_Matrix(
        num_clusters, bmus_values, bmus_dates, num_sim)

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(_faspect*_fsize, _fsize))

    bottom_val = np.zeros(m_plot[1,:].shape)
    for r in range(num_clusters):
        row_val = m_plot[r,:]
        plt.bar(
            range(365), row_val, bottom=bottom_val,
            width=1, color = np.array([np_colors_int[r]])
               )

        # store bottom
        bottom_val += row_val

    # axis
    # TODO: REFINAR AXIS
    plt.xlim(1, 365)
    plt.ylim(0, 1)
    plt.xlabel('Perpetual year')
    plt.ylabel('')

    # show and return figure
    if show: plt.show()
    return fig

