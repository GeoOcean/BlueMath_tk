#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op

# pip
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

# teslakit
from .custom_colors import GetClusterColors

# import constants
from .config import _faspect, _fsize, _fdpi


def axplot_PC_hist(ax, pc_wt, color_wt, nb=30, ylab=None):
    'axes plot AWT singular PC histogram'

    ax.hist(pc_wt, nb, density=True, color=color_wt)

    # gridlines and axis properties
    ax.grid(True, which='both', axis='both', linestyle='--', color='grey')
    ax.set_xlim([-3,3])
    ax.set_yticklabels([])
    ax.tick_params(axis='x', which='major', labelsize=5)
    if ylab:
        ax.set_ylabel(ylab, {'fontweight':'bold'}, labelpad=-3)

def axplot_PCs_2D(ax, PC1, PC2, d_wts, c_wts):
    'axes plot AWT PCs 1,2,3 (3D)'

    # calculate PC centroids
    pc1_wt = [np.mean(PC1[d_wts[i]]) for i in sorted(d_wts.keys())]
    pc2_wt = [np.mean(PC2[d_wts[i]]) for i in sorted(d_wts.keys())]

    # scatter  plot
    ax.scatter(
        PC1, PC2,
        c = 'silver',
        s = 3,
    )

    # WT centroids
    for x,y,c in zip(pc1_wt, pc2_wt, c_wts):
        ax.scatter(x, y, c=[c], s=10)

    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])
    ax.set_xticks([])
    ax.set_yticks([])

def axplot_PCs_3D(ax, pcs_wt, color_wt, ttl='PCs'):
    'axes plot AWT PCs 1,2,3 (3D)'

    PC1 = pcs_wt[:,0]
    PC2 = pcs_wt[:,1]
    PC3 = pcs_wt[:,2]

    # scatter  plot
    ax.scatter(
        PC1, PC2, PC3,
        c = [color_wt],
        s = 2,
    )

    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])
    ax.set_zlim([-3,3])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_title(ttl, {'fontsize':8, 'fontweight':'bold'})

def axplot_PCs_3D_WTs(ax, d_PCs, wt_colors, ttl='PCs'):
    'axes plot AWT PCs 1,2,3 (3D)'

    # plot each weather type
    wt_keys = sorted(d_PCs.keys())
    for ic, k in enumerate(wt_keys):
        PC1 = d_PCs[k][:,0]
        PC2 = d_PCs[k][:,1]
        PC3 = d_PCs[k][:,2]

        # scatter  plot
        ax.scatter(
            PC1, PC2, PC3,
            c = [wt_colors[ic]],
            label = k,
            s = 6,
        )

    ax.set_xlabel('PC1', {'fontsize':10})
    ax.set_ylabel('PC2', {'fontsize':10})
    ax.set_zlabel('PC3', {'fontsize':10})
    
    lim = np.nanmax([np.nanmax(np.abs(PC1)), np.nanmax(np.abs(PC2)), np.nanmax(np.abs(PC3))])
    
#     ax.set_xlim([-3,3])
#     ax.set_ylim([-3,3])
#     ax.set_zlim([-3,3])

    
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_title(ttl, {'fontsize':10, 'fontweight':'bold'})


def Plot_PCs_WT(PCs, variance, bmus, n_clusters, n=3, show=True):
    '''
    Plot Annual Weather Types PCs using 2D axis

    PCs, variance, bmus - from KMA_simple() or KMA_regression_guided()
    n                   - number of PCs to plot
    '''

    # get cluster colors
    cs_wt = GetClusterColors(n_clusters)

    # get cluster - bmus indexes
    d_wts = {}
    for i in range(n_clusters):
        d_wts[i] = np.where(bmus == i)[:]

    # figure
    fig = plt.figure(figsize=(_faspect*_fsize, _faspect*_fsize))
    gs = gridspec.GridSpec(n-1, n-1, wspace=0.0, hspace=0.0)

    for i in range(n):
        for j in range(i+1, n):

            # get PCs to plot
            PC1 = np.divide(PCs[:,i], np.sqrt(variance[i]))
            PC2 = np.divide(PCs[:,j], np.sqrt(variance[j]))

            # plot PCs (2D)
            ax = plt.subplot(gs[i, j-1])
            axplot_PCs_2D(ax, PC1, PC2, d_wts, cs_wt)

            # custom labels
            if i==0:
                ax.set_xlabel(
                    'PC {0}'.format(j+1),
                    {'fontsize':10, 'fontweight':'bold'}
                )
                ax.xaxis.set_label_position('top')
            if j==n-1:
                ax.set_ylabel(
                    'PC {0}'.format(i+1),
                    {'fontsize':10, 'fontweight':'bold'}
                )
                ax.yaxis.set_label_position('right')

    # show and return figure
    if show: plt.show()
    return fig

def Plot_PCs_Compare_3D(d_PCs_fit, d_PCs_rnd, show=True):
    '''
    Plot Annual Weather Types PCs fit - PCs rnd comparison (3D)
    '''

    # get cluster colors
    n_clusters = len(d_PCs_fit.keys())
    cs_wt = GetClusterColors(n_clusters)

    # figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize/1.66))
    gs = gridspec.GridSpec(1, 2, wspace=0.10, hspace=0.35)
    ax_fit = plt.subplot(gs[0, 0], projection='3d')
    ax_sim = plt.subplot(gs[0, 1], projection='3d')

    # Plot PCs (3D)
    axplot_PCs_3D_WTs(ax_fit, d_PCs_fit,  cs_wt, ttl='PCs fit')
    axplot_PCs_3D_WTs(ax_sim, d_PCs_rnd,  cs_wt, ttl='PCs sim')

    # show and return figure
    if show: plt.show()
    return fig

def Plot_WT_PCs_3D(d_PCs, n_clusters, show=True):
    '''
    Plot Weather Types PCs (3D)
    '''

    # get cluster colors
    cs_wt = GetClusterColors(n_clusters)

    # figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize*2/1.66))
    gs = gridspec.GridSpec(1, 1, wspace=0.10, hspace=0.35)
    ax_pcs = plt.subplot(gs[0, 0], projection='3d')

    # Plot PCs (3D)
    axplot_PCs_3D_WTs(ax_pcs, d_PCs,  cs_wt, ttl='PCs')

    # show and return figure
    if show: plt.show()
    return fig
