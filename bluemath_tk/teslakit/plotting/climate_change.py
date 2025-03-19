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
from ..util.operations import GetBestRowsCols
from .custom_colors import GetClusterColors
from ..kma import ClusterProbabilities, ChangeProbabilities

# import constants
from .config import _faspect, _fsize, _fdpi


def Plot_RCP_ocurrence(lon_grid, lat_grid, ocurrence, site_lon_index,
                       site_lat_index, show=True):
    '''
    Plot global map with RCP TCs ocurrence and study site point

    lon_grid, lat_grid - 2D numpy array. ocurrence coordinates meshgrid
    '''

    # parameters
    ttl = 'Tropical Cyclone tracks: [prob.RCP85/prob.HIST - 1] (%)'
    ttl += '\nProjected change in TCs occurrence rate (%)'

    # site location
    lon_site = lon_grid[site_lon_index, site_lat_index]
    lat_site = lat_grid[site_lon_index, site_lat_index]

    # plot figure
    fig, ax = plt.subplots(figsize=(_faspect*_fsize, _fsize))

    # Plot global map and location of Site
    pc = ax.pcolor(lon_grid, lat_grid, ocurrence, cmap='bwr', vmin=-40, vmax=40, label='')
    ax.plot(lon_site, lat_site, 'ok',label='Site')

    # colorbar
    plt.colorbar(pc, ax=ax)

    # titles and labels
    ax.set_title(ttl)
    ax.set_xlabel('Longitude (º)')
    ax.set_ylabel('Latitude (º)')

    plt.legend()

    # show and return figure
    if show: plt.show()
    return fig


