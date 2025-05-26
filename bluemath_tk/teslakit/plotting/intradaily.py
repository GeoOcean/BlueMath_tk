
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

from ..util.time_operations import date2datenum as d2d

# fig aspect and size
_faspect = (1+5**0.5)/2.0
_fsize = 7


# TODO: DEVELOP

def Plot_Hydrographs(dict_bins, p_export=None):
    ''

    # plot figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize))

    #plt.title('t')
    #plt.xlabel('x')
    #plt.ylabel('y')

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=128)
        plt.close()

def Plot_Hist_TAU(dict_bins, p_export=None):
    ''

    # plot figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize))

    #plt.title('t')
    #plt.xlabel('x')
    #plt.ylabel('y')

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=128)
        plt.close()

def Plot_Hist_MU(dict_bins, p_export=None):
    ''

    # plot figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize))

    #plt.title('t')
    #plt.xlabel('x')
    #plt.ylabel('y')

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=128)
        plt.close()

def Plot_Scatter_MU_TAU(dict_bins, p_export=None):
    ''

    # plot figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize))

    #plt.title('t')
    #plt.xlabel('x')
    #plt.ylabel('y')

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=128)
        plt.close()

