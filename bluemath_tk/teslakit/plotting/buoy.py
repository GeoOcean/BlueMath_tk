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

from scipy.stats import gaussian_kde, probplot
from sklearn.metrics import mean_squared_error


# import constants
from .config import _faspect, _fsize, _fdpi

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def bias(predictions, targets):
    return sum(predictions-targets)/len(predictions)

def si(predictions, targets):
    S = predictions.mean()
    O = targets.mean()
    return np.sqrt(sum(((predictions-S)-(targets-O))**2)/((sum(targets**2))))


def scatter_QQ(xds_1, xds_2, vn, xlabel='', ylabel=''):
    'Scatter plot v_1 - v_2 and adds rmse, si, bias and QQ plot'

    cmap='jet'

    # data
    x = xds_1[vn]
    y = xds_2[vn]

    # figure
    fig, ax = plt.subplots(1,1, constrained_layout=True,
                           figsize=(_faspect*_fsize,_faspect*_fsize))

    # stack data and gaussian kde
    xy = np.vstack([x, y])
    xy = xy[~np.isnan(xy).any(axis=1)]  # remove nans

    z = gaussian_kde(xy)(xy)

    # sort data
    idx = z.argsort()
    x2, y2, z = x[idx], y[idx], z[idx]

    ax.scatter(x2, y2, c=z, s=0.6, cmap=cmap)
    ax.set_xlabel('{0} {1}'.format(vn, xlabel))
    ax.set_ylabel('{0} {1}'.format(vn, ylabel))

    xmax = max(x) + 0.1
    ymax = max(y) + 0.1
    maxt = np.ceil(max(xmax, ymax))
    ax.set_xlim(0, maxt)
    ax.set_ylim(0, maxt)

    ax.plot([0, maxt],[0, maxt],'-k', linewidth=0.6)
    ax.set_xticks(np.linspace(0, maxt, 5))
    ax.set_yticks(np.linspace(0, maxt, 5))
    ax.set_aspect('equal')

    xqq = probplot(x2, dist="norm")
    yqq = probplot(y2, dist="norm")
    ax.plot(xqq[0][1], yqq[0][1], "o", markersize=1, color='gold' ,label='Q-Q plot')

    mse = mean_squared_error(x2, y2)
    rmse_e = rmse(x2, y2)
    BIAS = bias(x2, y2)
    SI = si(x2, y2)
    label = '\n'.join((
             r'RMSE = %.2f' % (rmse_e, ),
             r'mse =  %.2f' % (mse,  ),
             r'BIAS = %.2f' % (BIAS,  ),
             R'SI = %.2f' % (SI,  )))
    ax.text(0.05, 0.8, label, transform=ax.transAxes)
    ax.legend(loc='upper right')

def compare_series(xds_1, xds_2, vns, label_1='', label_2=''):

    # one plot for each variable at vns
    n_plots = len(vns)

    # figure
    fig, axs = plt.subplots(
        n_plots, 1, sharex=True,
        figsize=(_faspect*_fsize, _fsize)
    )

    # plot variables
    for c, vn in enumerate(vns):
        axs[c].scatter(xds_1['time'], xds_1[vn], label=label_1, color='r',
                    marker='.', s=6)
        axs[c].scatter(xds_2['time'], xds_2[vn],  label=label_2, color='k',
                    marker='.', s=6)
        axs[c].set_ylabel(vn)

    # config first plot
    axs[0].set_xlim(xds_1['time'][0], xds_1['time'][-1])
    axs[0].legend()

