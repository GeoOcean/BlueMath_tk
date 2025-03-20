#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
from datetime import datetime

# pip
import numpy as np
import xarray as xr

def MJO_Categories(rmm1, rmm2, phase):
    '''
    Divides MJO data in 25 categories.

    rmm1, rmm2, phase - MJO parameters

    returns array with categories time series
    and corresponding rmm
    '''

    rmm = np.sqrt(rmm1**2 + rmm2**2)
    categ = np.empty(rmm.shape) * np.nan

    for i in range(1,9):
        s = np.squeeze(np.where(phase == i))
        rmm_p = rmm[s]

        # categories
        categ_p = np.empty(rmm_p.shape) * np.nan
        categ_p[rmm_p <=1] =  25
        categ_p[rmm_p > 1] =  i + 8*2
        categ_p[rmm_p > 1.5] =  i + 8
        categ_p[rmm_p > 2.5] =  i
        categ[s] = categ_p

    # get rmm_categ
    rmm_categ = {}
    for i in range(1,26):
        s = np.squeeze(np.where(categ == i))
        rmm_categ['cat_{0}'.format(i)] = np.column_stack((rmm1[s],rmm2[s]))

    return categ.astype(int), rmm_categ

def MJO_Phases(rmm1, rmm2):
    'calculates and returns MJO phases (1-8) and degrees'

    # mjo degrees phase
    degr = (np.arctan2(rmm2, rmm1))*360/(2*np.pi)
    degr[degr<0] = degr[degr<0] + 360

    # degree to mjo phase (1-8)
    phases_sector = [
        (5, [0, 45]),
        (6, [45, 90]),
        (7, [90, 135]),
        (8, [135, 180]),
        (1, [180, 225]),
        (2, [225, 270]),
        (3, [270, 315]),
        (4, [315, 360]),
    ]

    # calculate phase
    phase = np.zeros(len(degr))*np.nan
    for p, s in phases_sector:
        phase[np.where((degr[:]>s[0]) & (degr[:]<=s[1]))] = p
    phase = phase.astype(int)

    return phase, degr

