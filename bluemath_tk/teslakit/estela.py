#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import os
import os.path as op

# pip
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path

# tk 
from .pca import PCA_EstelaPred
from .kma import KMA_regression_guided
from .kma import SimpleMultivariateRegressionModel as SMRM
from .intradaily import Calculate_Hydrographs

from .plotting.estela import Plot_EOFs_EstelaPred, Plot_DWTs_Mean_Anom, \
Plot_DWTs_Probs
from .plotting.pcs import Plot_PCs_WT, Plot_WT_PCs_3D


def spatial_gradient(xdset, var_name):
    '''
    Calculate spatial gradient

    xdset:
        (longitude, latitude, time), var_name

    returns xdset with new variable "var_name_gradient"
    '''

    # TODO:check/ ADD ONE ROW/COL EACH SIDE
    var_grad = np.zeros(xdset[var_name].shape)

    Mx = len(xdset.longitude)
    My = len(xdset.latitude)
    lat = xdset.latitude.values

    for it in range(len(xdset.time)):
        var_val = xdset[var_name].isel(time=it).values

        # calculate gradient (matrix)
        m_c = var_val[1:-1,1:-1]
        m_l = np.roll(var_val, -1, axis=1)[1:-1,1:-1]
        m_r = np.roll(var_val, +1, axis=1)[1:-1,1:-1]
        m_u = np.roll(var_val, -1, axis=0)[1:-1,1:-1]
        m_d = np.roll(var_val, +1, axis=0)[1:-1,1:-1]
        m_phi = np.pi*np.abs(lat)/180.0
        m_phi = m_phi[1:-1]

        dpx1 = (m_c - m_l)/np.cos(m_phi[:,None])
        dpx2 = (m_r - m_c)/np.cos(m_phi[:,None])
        dpy1 = m_c - m_d
        dpy2 = m_u - m_c

        vg = (dpx1**2+dpx2**2)/2 + (dpy1**2+dpy2**2)/2
        var_grad[it, 1:-1, 1:-1] = vg

        # calculate gradient (for). old code
        #for i in range(1, Mx-1):
        #    for j in range(1, My-1):
        #        phi = np.pi*np.abs(lat[j])/180.0
        #        dpx1 = (var_val[j,i]   - var_val[j,i-1]) / np.cos(phi)
        #        dpx2 = (var_val[j,i+1] - var_val[j,i])   / np.cos(phi)
        #        dpy1 = (var_val[j,i]   - var_val[j-1,i])
        #        dpy2 = (var_val[j+1,i] - var_val[j,i])
        #        var_grad[it, j, i] = (dpx1**2+dpx2**2)/2 + (dpy1**2+dpy2**2)/2

    # store gradient
    xdset['{0}_gradient'.format(var_name)]= (
        ('time', 'latitude', 'longitude'), var_grad)

    return xdset

def mask_from_poly(xdset, ls_poly, name_mask='mask'):
    '''
    Generate mask from list of tuples (lon, lat)

    xdset dimensions:
        (longitude, latitude, )

    returns xdset with new variable "mask"
    '''

    lon = xdset.longitude.values
    lat = xdset.latitude.values
    mesh_lat, mesh_lon = np.meshgrid(lat, lon)
    mask = np.zeros(mesh_lat.shape)

    mesh_points = np.array(
        [mesh_lon.flatten(), mesh_lat.flatten()]
    ).T

    for pol in ls_poly:
        p = path.Path(pol)
        inside = np.array(p.contains_points(mesh_points))
        inmesh = np.reshape(inside, mask.shape)
        mask[inmesh] = 1

    xdset[name_mask]=(('latitude','longitude'), mask.T)

    return xdset

def dynamic_estela_predictor(xdset, var_name, estela_D):
    '''
    Generate dynamic predictor using estela

    xdset:
        (time, latitude, longitude), var_name, mask

    returns similar xarray.Dataset with variables:
        (time, latitude, longitude), var_name_comp
        (time, latitude, longitude), var_name_gradient_comp
    '''

    # first day is estela max
    first_day = int(np.floor(np.nanmax(estela_D)))+1

    # output will start at time=first_day
    shp = xdset[var_name].shape
    comp_shape = (shp[0]-first_day, shp[1], shp[2])
    var_comp = np.ones(comp_shape) * np.nan
    var_grd_comp = np.ones(comp_shape) * np.nan

    # get data using estela for each cell
    for i_lat in range(len(xdset.latitude)):
        for i_lon in range(len(xdset.longitude)):
            ed = estela_D[i_lat, i_lon]
            if not np.isnan(ed):

                # mount estela displaced time array 
                i_times = np.arange(
                    first_day, len(xdset.time)
                ) - np.int(ed)

                # select data from displaced time array positions
                xdselec = xdset.isel(
                    time = i_times,
                    latitude = i_lat,
                    longitude = i_lon)

                # get estela predictor values
                var_comp[:, i_lat, i_lon] = xdselec[var_name].values
                var_grd_comp[:, i_lat, i_lon] = xdselec['{0}_gradient'.format(var_name)].values

    # return generated estela predictor
    return xr.Dataset(
        {
            '{0}_comp'.format(var_name):(
                ('time','latitude','longitude'), var_comp),
            '{0}_gradient_comp'.format(var_name):(
                ('time','latitude','longitude'), var_grd_comp),

        },
        coords = {
            'time':xdset.time.values[first_day:],
            'latitude':xdset.latitude.values,
            'longitude':xdset.longitude.values,
        }
    )


class Predictor(object):
    '''
    tesla-kit custom dataset handler

    used for 3D dataset (lon,lat,time) and related
    statistical classification calculations and figures.
    '''

    def __init__(self, p_store):

        # file paths
        self.p_store = p_store
        self.p_data = op.join(p_store, 'data.nc')
        self.p_pca = op.join(p_store, 'pca.nc')
        self.p_kma = op.join(p_store, 'kma.nc')
        self.p_plots = op.join(p_store, 'figs')

        # data (xarray.Dataset)
        self.data = None
        self.PCA = None
        self.KMA = None

    def Load(self):
        if op.isfile(self.p_data):
            self.data = xr.open_dataset(self.p_data)
        if op.isfile(self.p_pca):
            self.PCA = xr.open_dataset(self.p_pca)
        if op.isfile(self.p_kma):
            self.KMA = xr.open_dataset(self.p_kma)

    def Save(self):
        try:
            os.makedirs(self.p_store)
        except:
            pass
        if self.data:
            if op.isfile(self.p_data): os.remove(self.p_data)
            self.data.to_netcdf(self.p_data,'w')
        if self.PCA:
            if op.isfile(self.p_pca): os.remove(self.p_pca)
            self.PCA.to_netcdf(self.p_pca,'w')
        if self.KMA:
            if op.isfile(self.p_kma): os.remove(self.p_kma)
            self.KMA.to_netcdf(self.p_kma,'w')

    def Calc_PCA_EstelaPred(self, var_name, xds_estela):
        'Principal components analysis using estela predictor'

        # generate estela predictor
        # TODO STORE DYNAMIC ESTELA PREDICTOR (too big..)
        xds_estela_pred = dynamic_estela_predictor(
            self.data, var_name, xds_estela)

        # Calculate PCA
        self.PCA = PCA_EstelaPred(
            xds_estela_pred, var_name)

        # save data
        self.Save()

    def Calc_KMA_regressionguided(
        self, num_clusters, xds_waves, waves_vars, alpha, min_group_size=None, repres = 0.95):
        'KMA regression guided with waves data'

        # we have to miss some days of data due to ESTELA
        tcut = self.PCA.pred_time.values[:]

        # calculate regresion model between predictand and predictor
        xds_waves = xds_waves.sel(time = slice(tcut[0], tcut[-1]))
        xds_Yregres = SMRM(self.PCA, xds_waves, waves_vars)

        # classification: KMA regresion guided
        self.KMA = KMA_regression_guided(
            self.PCA, xds_Yregres,
            num_clusters, repres, alpha, min_group_size
        )

        # store time array with KMA
        self.KMA['time'] = (('n_components',), self.PCA.pred_time.values[:])

        # save data
        self.Save()

    def Calc_MU_TAU_Hydrographs(self, xds_WAVES):
        '''
        Calculates TWL hydrographs

        returns list of xarray.Dataset with TWL hydrographs MU,TAU arrays for each WT
        '''

        # TODO: SACAR DE AQUI, replantear hydrographs

        # get sorted bmus from kma
        xds_BMUS = xr.Dataset(
            {'bmus':(('time', self.KMA.sorted_bmus.values[:]))},
            coords = {'time': self.KMA.time.values[:]}
        )

        # Calculate hydrographs for each WT
        _, l_xds_MUTAU = Calculate_Hydrographs(xds_BMUS, xds_WAVES)

        return l_xds_MUTAU

    def Mod_KMA_AddStorms(self, storm_dates, storm_categories):
        '''
        Modify KMA bmus series adding storm category (6 new groups)
        '''

        n_clusters = len(self.KMA.n_clusters.values[:])
        kma_dates = self.KMA.time.values[:]
        bmus_storms = np.copy(self.KMA.sorted_bmus.values[:])  # copy numpy.array

        for sd, sc in zip(storm_dates, storm_categories):
            sdr =  np.array(sd, dtype='datetime64[D]')  # round to day
            pos_date = np.where(kma_dates==sdr)[0]
            if pos_date:
                bmus_storms[pos_date[0]] = n_clusters + sc

        # copy kma and add bmus_storms
        self.KMA['sorted_bmus_storms'] = (('n_components',), bmus_storms)

        # store changes
        if op.isfile(self.p_kma): os.remove(self.p_kma)
        self.KMA.to_netcdf(self.p_kma,'w')

    def Plot_EOFs_EstelaPred(self, n_plot=3, show=True):
        'Plot EOFs generated in PCA_EstelaPred'

        # optional land mask
        mask_land = None
        if 'mask_land' in self.data.keys():
            mask_land = self.data['mask_land'].values[:]

        l_fs = Plot_EOFs_EstelaPred(self.PCA, n_plot, mask_land=mask_land, show=show)
        return l_fs

    def Plot_DWTs(self, var_name, kind='mean', show=True):
        '''
        Plot KMA clusters generated in PCA_EstelaPred (DWTs)

        kind = mean / anom

        uses database means/anomalies at cluster location (bmus corrected)
        '''

        # data to plot
        xds_DWTs = self.KMA
        var_data = self.data[var_name]

        # optional land mask
        mask_land = None
        if 'mask_land' in self.data.keys():
            mask_land = self.data['mask_land'].values[:]

        # Plot DWTs mean using var_data
        fig = Plot_DWTs_Mean_Anom(
            xds_DWTs, var_data,
            kind = kind,
            mask_land = mask_land,
            show = show
        )

        return fig

    def Plot_DWTs_Probs(self, show=True, field='sorted_bmus', n_clusters=None):
        '''
        Plot DWTs bmus probabilities
            - histogram for ocurrences
            - probs. all series
            - probs by month
            - probs by 3month
        '''

        # Plot DWTs mean using var_data
        bmus = self.KMA[field].values[:] + 1 # index to DWT id
        bmus_time = self.KMA['time'].values[:]
        if not n_clusters:
            n_clusters = len(self.KMA.n_clusters.values[:])

        fig = Plot_DWTs_Probs(bmus, bmus_time, n_clusters, show=show)

        return fig

    def Plot_DWT_PCs(self, n=3, show=True):
        '''
        Plot Daily Weather Types PCs using 2D axis
        '''

        # get estela data
        PCs = self.PCA.PCs.values[:]
        variance = self.PCA.variance.values[:]
        bmus = self.KMA.sorted_bmus.values[:]  # sorted_bmus
        n_clusters = len(self.KMA.n_clusters.values[:])

        # Plot DWTs PCs
        fig = Plot_PCs_WT(PCs, variance, bmus, n_clusters, n, show=show)

        return fig

    def Plot_PCs_3D(self, show=True):
        'Plots Predictor first 3 PCs'

        # first 3 PCs
        bmus = self.KMA['sorted_bmus'].values[:]
        PCs = self.PCA.PCs.values[:]
        variance = self.PCA.variance.values[:]

        n_clusters = len(self.KMA.n_clusters.values[:])

        PC1 = np.divide(PCs[:,0], np.sqrt(variance[0]))
        PC2 = np.divide(PCs[:,1], np.sqrt(variance[1]))
        PC3 = np.divide(PCs[:,2], np.sqrt(variance[2]))

        # dictionary of DWT PCs 123 
        d_PCs = {}
        for ic in range(n_clusters):
            ind = np.where(bmus == ic)[:]

            PC123 = np.column_stack((PC1[ind], PC2[ind], PC3[ind]))
            d_PCs['{0}'.format(ic+1)] = PC123

        # Plot DWTs PCs 3D 
        fig = Plot_WT_PCs_3D(d_PCs, n_clusters, show=show)

        return fig

