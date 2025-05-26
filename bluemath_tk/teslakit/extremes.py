#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pip
from datetime import timedelta
import numpy as np
import xarray as xr
from scipy.stats import  gumbel_l, genextreme

from .util.time_operations import date2datenum as d2d

def FitGEV_KMA_Frechet(bmus, n_clusters, var):
    '''
    Returns stationary GEV/Gumbel_L params for KMA bmus and varible series

    bmus        - KMA bmus (time series of KMA centroids)
    n_clusters  - number of KMA clusters
    var         - time series of variable to fit to GEV/Gumbel_L

    returns np.array (n_clusters x parameters). parameters = (shape, loc, scale)
    for gumbel distributions shape value will be ~0 (0.0000000001)
    '''

    param_GEV = np.empty((n_clusters, 3))
    for i in range(n_clusters):
        c = i+1
        pos = np.where((bmus==c))[0]

        if len(pos) == 0:
            param_GEV[i,:] = [np.nan, np.nan, np.nan]

        else:

            # get variable at cluster position
            var_c = var[pos]
            var_c = var_c[~np.isnan(var_c)]

            # fit to Gumbel_l and get negative loglikelihood
            loc_gl, scale_gl = gumbel_l.fit(-var_c)
            theta_gl = (0.0000000001, -1*loc_gl, scale_gl)
            nLogL_gl = genextreme.nnlf(theta_gl, var_c)

            # fit to GEV and get negative loglikelihood
            c = -0.1
            shape_gev, loc_gev, scale_gev = genextreme.fit(var_c, c)
            theta_gev = (shape_gev, loc_gev, scale_gev)
            nLogL_gev = genextreme.nnlf(theta_gev, var_c)

            # store negative shape
            theta_gev_fix = (-shape_gev, loc_gev, scale_gev)

            # apply significance test if Frechet
            if shape_gev < 0:

                # TODO: cant replicate ML exact solution
                if nLogL_gl - nLogL_gev >= 1.92:
                    param_GEV[i,:] = list(theta_gev_fix)
                else:
                    param_GEV[i,:] = list(theta_gl)
            else:
                param_GEV[i,:] = list(theta_gev_fix)

    return param_GEV

def Smooth_GEV_Shape(cenEOFs, param):
    '''
    Smooth GEV shape parameter (for each KMA cluster) by promediation
    with neighbour EOFs centroids

    cenEOFs  - (n_clusters, n_features) KMA centroids
    param    - GEV shape parameter for each KMA cluster

    returns smoothed GEV shape parameter as a np.array (n_clusters)
    '''

    # number of clusters
    n_cs = cenEOFs.shape[0]

    # calculate distances (optimized)
    cenEOFs_b = cenEOFs.reshape(cenEOFs.shape[0], 1, cenEOFs.shape[1])
    D = np.sqrt(np.einsum('ijk, ijk->ij', cenEOFs-cenEOFs_b, cenEOFs-cenEOFs_b))
    np.fill_diagonal(D, np.nan)

    # sort distances matrix to find neighbours
    sort_ord = np.empty((n_cs, n_cs), dtype=int)
    D_sorted = np.empty((n_cs, n_cs))
    for i in range(n_cs):
        order = np.argsort(D[i,:])
        sort_ord[i,:] = order
        D_sorted[i,:] = D[i, order]

    # calculate smoothed parameter
    denom = np.sum(1/D_sorted[:,:4], axis=1)
    param_c = 0.5 * np.sum(np.column_stack(
        [
            param[:],
            param[sort_ord[:,:4]] * (1/D_sorted[:,:4])/denom[:,None]
        ]
    ), axis=1)

    return param_c

def ACOV(f, theta, x):
    '''
    Returns asyntotyc variance matrix using Fisher Information matrix inverse
    Generalized functions, parameters and data.

    f      - function to evaluate: GEV, GUMBELL, ...
    theta  - function parameters: for GEV (shape, location, scale)
    x      - data used for function evaluation

    Second derivative evaluation - variance and covariance
    dxx = (f(x+dt_x) - 2f(x) + f(x-dt_x)) / (dt_x**2)
    dxy = (f(x,y) - f(x-dt_x,y) - f(x,y-dt_y) + f(x-dt_x, u-dt_y)) / (dt_x*dt_y)
    '''

    # parameters differential
    pm = 0.00001
    params = np.asarray(theta)
    dt_p = pm * params

    # Fisher information matrix holder 
    ss = len(params)
    FI = np.ones((ss,ss)) * np.nan

    # TODO: evaluar f falla en algunos casos?? 
    if np.isinf(f(theta, x)):
        #print ('ACOV error: nLogL = Inf. {0}'.format(theta))
        return np.ones((ss,ss))*0.0001

    # variance and covariance
    for i in range(ss):

        # diferential parameter FI evaluation
        p1 = np.asarray(theta); p1[i] = p1[i] + dt_p[i]
        p2 = np.asarray(theta); p2[i] = p2[i] - dt_p[i]

        # variance
        FI[i,i] = (f(tuple(p1), x) - 2*f(theta,x) + f(tuple(p2), x))/(dt_p[i]**2)

        for j in range(i+1,ss):

            # diferential parameter FI evaluation
            p1 = np.asarray(theta); p1[i] = p1[i] - dt_p[i]
            p2 = np.asarray(theta); p2[j] = p2[j] - dt_p[j]
            p3 = np.asarray(theta); p3[i] = p3[i] - dt_p[i]; p3[j] = p3[j] - dt_p[j]

            # covariance
            cov = (f(theta,x) - f(tuple(p1),x) - f(tuple(p2),x) + f(tuple(p3),x)) \
                    / (dt_p[i]*dt_p[j])
            FI[i,j] = cov
            FI[j,i] = cov

    # asynptotic variance covariance matrix
    acov = np.linalg.inv(FI)

    return acov

def Peaks_Over_Threshold(xds, var_name, percentile=99, threshold=None,
                         window_days=3):
    '''

    Peaks Over Threshold methodology to find extreme values.

    xds         - xarray.Dataset (time dim)
    var_name    - variable to apply POT
    percentile  - if threshold not given, calculate it with this percentile
    threshold   - optional, threshold to apply POT
    window_days - minimum number of days between consecutive independent peaks

    returns xarray.Dataset ('time', ) vars: peaks, excedeence, area, and duration
    '''

    # TODO: refactor with times
    def to_hours(time_diff):

        # single
        if isinstance(time_diff, (timedelta, np.timedelta64)):
            if isinstance(time_diff, np.timedelta64):
                dt_h = time_diff / np.timedelta64(1, 'h')

            elif isinstance(time_diff, timedelta):
                dt_h = np.array(time_diff.total_seconds()/3600.0)

            return dt_h

        # else: array
        if isinstance(time_diff[0], np.timedelta64):
            dt_h = time_diff / np.timedelta64(1, 'h')

        elif isinstance(time_diff[0], timedelta):
            dt_h = np.array([x.total_seconds()/3600.0 for x in time_diff])

        return dt_h


    # get variable
    vv = xds[var_name].values[:]
    vt = xds['time'].values[:]

    # data delta time (hours)
    npdf = np.diff(vt)
    dt_h = to_hours(npdf)

    # threshold
    if threshold == None:
        threshold = np.percentile(vv, percentile)

    # consecutive peaks over threshold
    ind_mask = np.where(vv >= threshold, 1, 0)
    ind_dif = np.diff(ind_mask)

    # peaks start and end
    ind_ini = np.where(ind_dif == 1)[0] + 1
    ind_end = np.where(ind_dif == -1)[0] + 1

    # start and end corrections
    if ind_end[0]<ind_ini[0]: ind_ini = np.insert(ind_ini,0,0)
    if ind_end[-1]<ind_ini[-1]: ind_end = np.append(ind_end, len(ind_dif))

    # store variable max, time max, duration and area above threshold for each peak
    time_max, vv_max, area, durac = [], [], [], []

    for i, f in zip(ind_ini, ind_end):

        vv_max.append(np.max(vv[i:f]))

        area.append(np.sum(vv[i:f] * dt_h[i:f] - threshold))

        ind_max = np.argmax(vv[i:f])
        ind_time = list(range(i, f))
        time_max.append(vt[ind_time[ind_max]])

        durac.append(to_hours(vt[f] - vt[i]))


    # ensure independence between maxs
    ind_window_mask = np.where(to_hours(np.diff(time_max)) < 24*window_days, 1, 0)
    if np.any(ind_window_mask):

        # solve dependent events
        ind_dif = np.diff(ind_window_mask)

        # first event corrections 
        ind_dif = np.insert(ind_dif, 0, 0)
        if ind_dif[1] == -1: ind_dif[0] = 1

        ind_ini = np.where(ind_dif == 1)[0]
        ind_fin = np.where(ind_dif == -1)[0] + 1

        # Keep maximum of dependent events
        time_max_indep = []
        vv_max_indep = []
        durac_indep = []
        area_indep = []
        ind_delete = []
        for ind_i, ind_f in zip(ind_ini, ind_fin):

            vv_temp = np.max(vv_max[ind_i:ind_f])
            vv_max_indep.append(vv_temp)

            vv_ind_max = np.argmax(vv_max[ind_i:ind_f])
            ind_time = list(range(ind_i,ind_f))
            time_temp = time_max[ind_time[vv_ind_max]]
            time_max_indep.append(time_temp)

            durac_indep.append(
                to_hours(time_max[ind_f]-time_max[ind_i])
            )

            area_indep.append(np.sum(area[ind_i:ind_f]))

            ind_delete.extend(ind_time)

        # remove all dependent events
        vv_max = np.delete(vv_max, [ind_delete])
        time_max = np.delete(time_max, [ind_delete])
        durac = np.delete(durac, [ind_delete])
        area = np.delete(area, [ind_delete])

        # add independent events
        vv_max = np.concatenate((vv_max, vv_max_indep))
        time_max = np.concatenate((time_max, time_max_indep))
        durac = np.concatenate((durac, durac_indep))  # hours
        area = np.concatenate((area, area_indep))

    # output
    peaks = xr.Dataset(
        {
            '{0}'.format(var_name): (('time',), vv_max),
            '{0}_exceedances'.format(var_name): (('time',), vv_max-threshold),
            'duration': (('time',), durac),
            'area': (('time',), area),
        },
        coords = {'time': time_max},
    )
    peaks = peaks.sortby('time')

    # add some attrs
    peaks.attrs['threshold'] = threshold
    peaks.attrs['percentile'] = percentile

    return peaks



