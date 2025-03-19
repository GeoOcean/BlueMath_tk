#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op

# pip
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import (MultipleLocator)

# teslakit
from .util import MidpointNormalize
from .custom_colors import GetClusterColors

# import constants
from .config import _faspect, _fsize, _fdpi


def Generate_Covariate_Matrix(
    bmus_values, covar_values,
    bmus_dates, covar_dates,
    num_clusters, covar_rng, num_sim=1):
    'Calculates and returns matrix for stacked bar plotting'

    # generate aux arrays
    #bmus_years = [d.year for d in bmus_dates]
    m_plot = np.zeros((num_clusters, len(covar_rng)-1)) * np.nan

    for i in range(len(covar_rng)-1):

        # find years inside range
        _, s = np.where(
            [(covar_values >= covar_rng[i]) & (covar_values <= covar_rng[i+1])]
        )

        # TODO: usando alr_wrapper las fechas covar y bmus coinciden
        b = bmus_values[s,:]
        b = b.flatten()

        # TODO: mejorar, no usar los years y posicion. 
        # usar la fecha

        #ys = [covar_dates[x].year for x in s]
        # find data inside years found
        #sb = np.where(np.in1d(bmus_years, ys))[0]
        #b = bmus_values[sb,:]
        #b = b.flatten()

        for j in range(num_clusters):
            _, bb = np.where([(j+1 == b)])
            # TODO sb se utiliza para el test de laura
            #if len(sb) > 0:
                #m_plot[j,i] = float(len(bb)/float(num_sim))/len(sb)
            if len(s) > 0:
                m_plot[j,i] = float(len(bb)/float(num_sim))/len(s)
            else:
                m_plot[j,i] = 0

    return m_plot

def Generate_Covariate_rng(covar_name, cov_values):
    'Returns covar_rng and interval depending on covar_name'

    if covar_name.startswith('PC'):
        delta = 5
        n_rng = 7

        covar_rng = np.linspace(
            np.min(cov_values)-delta,
            np.max(cov_values)+delta,
            n_rng
        )

    elif covar_name.startswith('MJO'):
        delta = 0.5
        n_rng = 7

        covar_rng = np.linspace(
            np.min(cov_values)-delta,
            np.max(cov_values)+delta,
            n_rng
        )

    else:
        print('Cant plot {0}, missing rng params in plotting library'.format(
            name_covar
        ))
        return None, None

    # interval
    interval = covar_rng[1]-covar_rng[0]

    return covar_rng, interval


def Plot_PValues(p_values, term_names, show=True):
    'Plot ARL/BMUS p-values'

    n_wts = p_values.shape[0]
    n_terms = p_values.shape[1]

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(_faspect*_fsize, _fsize))

    c = ax.pcolor(p_values, cmap='coolwarm_r', clim=(0,1),
                  norm=MidpointNormalize(midpoint=0.1, vmin=0, vmax=1))
    #c.cmap.set_over('w')
    fig.colorbar(c, ax=ax)

    # Pval text
    #for i in range(p_values.shape[1]):
    #    for j in range(p_values.shape[0]):
    #        v = p_values[j,i]
    #        if v<=0.1:
    #            ax.text(i+0.5, j+0.5, '{0:.2f}'.format(v),
    #                    va='center', ha='center', size=6)

    # axis
    ax.set_title('p-value', fontweight='bold')
    ax.set_ylabel('WT')

    ax.xaxis.tick_bottom()
    ax.set_xticks(np.arange(n_terms), minor=True)
    ax.set_xticks(np.arange(n_terms)+0.5, minor=False)
    ax.set_xticklabels(term_names, minor=False, rotation=90, fontsize=7)

    ax.set_yticks(np.arange(n_wts), minor=True)
    ax.set_yticks(np.arange(n_wts)+0.5, minor=False)
    ax.set_yticklabels(np.arange(n_wts)+1, minor=False)

    # add grid
    ax.grid(True, which='minor', axis='both', linestyle='-', color='k')

    # show and return figure
    if show: plt.show()
    return fig

def Plot_Params(params, term_names, show=True):
    'Plot ARL/BMUS params'

    n_wts = params.shape[0]
    n_terms = params.shape[1]

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(_faspect*_fsize, _fsize))

    # text table and color
    #c = ax.pcolor(params, cmap=plt.cm.bwr)
    c = ax.pcolor(
        params, cmap='coolwarm_r',
        norm=MidpointNormalize(midpoint=0)
    )
    #for i in range(params.shape[1]):
    #    for j in range(params.shape[0]):
    #        v = params[j,i]
    #        ax.text(i+0.5, j+0.5, '{0:.1f}'.format(v),
    #                va='center', ha='center', size=6)
    fig.colorbar(c, ax=ax)

    # axis
    ax.set_title('params', fontweight='bold')
    ax.set_ylabel('WT')

    ax.xaxis.tick_bottom()
    ax.set_xticks(np.arange(n_terms), minor=True)
    ax.set_xticks(np.arange(n_terms)+0.5, minor=False)
    ax.set_xticklabels(term_names, minor=False, rotation=90, fontsize=7)

    ax.set_yticks(np.arange(n_wts), minor=True)
    ax.set_yticks(np.arange(n_wts)+0.5, minor=False)
    ax.set_yticklabels(np.arange(n_wts)+1, minor=False)

    # add grid
    ax.grid(True, which='minor', axis='both', linestyle='-', color='k')

    # show and return figure
    if show: plt.show()
    return fig


def Plot_Log_Sim(log, show=True):
    '''
    Plot ALR simulation log

    log - xarray.Dataset from alr wrapper (n_sim already selected)
    '''

    # plot figure
    #fig = plt.figure(figsize=(_faspect*_fsize, _fsize))
    fig = plt.figure(figsize=[18.5,9])

    # figure gridspec
    gs1 = gridspec.GridSpec(4,1)
    ax1 = fig.add_subplot(gs1[0])
    ax2 = fig.add_subplot(gs1[1], sharex=ax1)
    ax3 = fig.add_subplot(gs1[2], sharex=ax1)
    ax4 = fig.add_subplot(gs1[3], sharex=ax1)

    # Plot evbmus values
    ax1.plot(
        log.time, log.evbmus_sims, ':',
        linewidth=0.5, color='grey',
        marker='.', markersize=3,
        markerfacecolor='crimson', markeredgecolor='crimson'
    )

    ax1.yaxis.set_major_locator(MultipleLocator(4))
    ax1.grid(which='major', linestyle=':', alpha=0.5)
    ax1.set_xlim(log.time[0], log.time[-1])
    ax1.set_ylabel('Bmus', fontsize=12)

    # Plot evbmus probabilities
    z = np.diff(np.column_stack(
        ([np.zeros([len(log.time),1]), log.probTrans.values])
    ), axis=1)
    z1 = np.column_stack((z, z[:,-1])).T
    z2 = np.column_stack((z1, z1[:,-1]))
    p1 = ax2.pcolor(
        np.append(log.time, log.time[-1]),
        np.append(log.n_clusters, log.n_clusters[-1]), z2,
        cmap='PuRd', edgecolors='grey', linewidth=0.05
    )
    ax2.set_ylabel('Bmus',fontsize=12)

    # TODO: gestionar terminos markov 
    # TODO: no tengo claro si el primero oel ultimo
    alrt0 = log.alr_terms.isel(mk=0)

    # Plot Terms
    for v in range(len(log.terms)):
        if log.terms.values[v].startswith('ss'):
            ax3.plot(log.time, alrt0[:,v], label=log.terms.values[v])

        if log.terms.values[v].startswith('PC'):
            ax4.plot(log.time, alrt0[:,v], label=log.terms.values[v])

        if log.terms.values[v].startswith('MJO'):
            ax4.plot(log.time, alrt0[:,v], label=log.terms.values[v])

    # TODO: plot terms markov??

    ax3.set_ylim(-1.8,1.2)
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(loc='lower left',ncol=len(handles))
    ax3.set_ylabel('Seasonality',fontsize=12)

    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(loc='lower left',ncol=len(handles))
    ax4.set_ylabel('Covariates',fontsize=12)
    # cbar=plt.colorbar(p1,ax=ax2,pad=0)
    # cbar.set_label('Transition probability')

    gs1.tight_layout(fig, rect=[0.05, [], 0.95, []])

    # custom colorbar for probability
    gs2 = gridspec.GridSpec(1,1)
    ax1 = fig.add_subplot(gs2[0])
    plt.colorbar(p1, cax=ax1)
    ax1.set_ylabel('Probability')
    gs2.tight_layout(fig, rect=[0.935, 0.52, 0.99, 0.73])

    # show and return figure
    if show: plt.show()
    return fig


# TODO: following functions are not finished / tested

def Plot_Covariate(bmus_values, covar_values,
                   bmus_dates, covar_dates,
                   num_clusters, name_covar,
                   num_sims=1,
                   p_export=None):
    'Plots ARL covariate related to bmus stacked bar chart'

    # get cluster colors for stacked bar plot
    np_colors_int = GetClusterColors(num_clusters)

    # generate common covar_rng
    covar_rng, interval = Generate_Covariate_rng(
        name_covar, covar_values)

    # generate plot matrix
    m_plot = Generate_Covariate_Matrix(
        bmus_values, covar_values,
        bmus_dates, covar_dates,
        num_clusters, covar_rng, num_sims)

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(_faspect*_fsize, _fsize))
    x_val = covar_rng[:-1]

    bottom_val = np.zeros(m_plot[1,:].shape)
    for r in range(num_clusters):
        row_val = m_plot[r,:]
        plt.bar(
            x_val, row_val, bottom=bottom_val,
            width=interval, color = np.array([np_colors_int[r]])
               )

        # store bottom
        bottom_val += row_val

    # axis
    plt.xlim(np.min(x_val)-interval/2, np.max(x_val)+interval/2)
    plt.ylim(0, 1)
    plt.xlabel(name_covar)
    plt.ylabel('')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

def Plot_Terms(terms_matrix, terms_dates, terms_names, show=True):
    'Plot terms used for ALR fitting'

    # number of terms
    n_sps = terms_matrix.shape[1]

    # custom fig size
    fsy = n_sps * 2

    # plot figure
    fig, ax_list = plt.subplots(
        n_sps, 1, sharex=True, figsize=(_faspect*_fsize, fsy)
    )

    x = terms_dates
    for i in range(n_sps):
        y = terms_matrix[:,i]
        n = terms_names[i]
        ax = ax_list[i]
        ax.plot(x, y, '.b')
        ax.set_title(n, loc='left', fontweight='bold', fontsize=10)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(-1, 1)
        ax.grid(True, which='both')

        if n=='intercept':
            ax.set_ylim(0, 2)

    # date label
    fig.text(0.5, 0.04, 'date (y)', ha='center', fontweight='bold')
    fig.text(0.04, 0.5, 'value (-)', va='center', rotation='vertical',
             fontweight='bold')

    # show and return figure
    if show: plt.show()
    return fig

def Plot_Compare_Covariate(num_clusters,
                           bmus_values_sim, bmus_dates_sim,
                           bmus_values_hist, bmus_dates_hist,
                           cov_values_sim, cov_dates_sim,
                           cov_values_hist, cov_dates_hist,
                           name_covar,
                           n_sim = 1, p_export=None):
    'Plot simulated - historical bmus comparison related to covariate'

    # get cluster colors for stacked bar plot
    np_colors_int = GetClusterColors(num_clusters)

    # generate common covar_rng
    covar_rng, interval = Generate_Covariate_rng(
        name_covar, np.concatenate((cov_values_sim, cov_values_hist)))

    # generate plot matrix
    m_plot_sim = Generate_Covariate_Matrix(
        bmus_values_sim, cov_values_sim,
        bmus_dates_sim, cov_dates_sim,
        num_clusters, covar_rng, n_sim)

    m_plot_hist = Generate_Covariate_Matrix(
        bmus_values_hist, cov_values_hist,
        bmus_dates_hist, cov_dates_hist,
        num_clusters, covar_rng, 1)

    # plot figure
    fig, (ax_hist, ax_sim) = plt.subplots(2,1, figsize=(_faspect*_fsize, _fsize))
    x_val = covar_rng[:-1]

    # sim
    bottom_val = np.zeros(m_plot_sim[1,:].shape)
    for r in range(num_clusters):
        row_val = m_plot_sim[r,:]
        ax_sim.bar(
            x_val, row_val, bottom=bottom_val,
            width=interval, color = np.array([np_colors_int[r]])
               )

        # store bottom
        bottom_val += row_val

    # hist
    bottom_val = np.zeros(m_plot_hist[1,:].shape)
    for r in range(num_clusters):
        row_val = m_plot_hist[r,:]
        ax_hist.bar(
            x_val, row_val, bottom=bottom_val,
            width=interval, color = np_colors_int[r]
               )

        # store bottom
        bottom_val += row_val

    # axis
    # MEJORAR Y METER EL NOMBRE DE LA COVARIATE
    ax_sim.set_xlim(np.min(x_val)-interval/2, np.max(x_val)+interval/2)
    ax_sim.set_ylim(0, 1)
    ax_sim.set_title('Simulation')
    ax_sim.set_ylabel('')
    ax_sim.set_xlabel('')

    ax_hist.set_xlim(np.min(x_val)-interval/2, np.max(x_val)+interval/2)
    ax_hist.set_ylim(0, 1)
    ax_hist.set_title('Historical')
    ax_hist.set_ylabel('')
    ax_sim.set_xlabel('')

    fig.suptitle(name_covar, fontweight='bold', fontsize=12)

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

