import warnings
warnings.simplefilter('ignore')

import os
import os.path as op

import pickle

import glob
from datetime import datetime, timedelta, date
from cftime._cftime import DatetimeGregorian

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import regionmask


from ..datamining.pca import PCA
from ..datamining.kma import KMA
from ..core.plotting.colors import get_config_variables, get_cluster_colors

config_variables = get_config_variables()

class DWT:

    '''
    This class implements the Princial Component Algorithm (PCA)
    
    '''

    def __init__(self):
        
        self.data = []
        self.p_data = []
        self.area = [0, -90, 360, 90]
        self.dim_pca = 'time'
        self

    def generate_dataset(self):
    
        p_data = self.p_data
        area = self.area
        resample_factor = self.resample_factor
        remove_land = self.remove_land
        estela = self.estela
        mslp_grad = self.mslp_grad
    
        for ivar, var in enumerate(self.predictor):
    
            p_file = glob.glob(os.path.join(p_data, '*{0}*1day*'.format(var)))[0]
            ds = xr.open_dataset(p_file)

            if var == 'sst':
                 if ds.sst.mean().values > 150:
                     ds['sst'] = ds['sst'] - 273.15 #transform kelvin to celsius

            if var == 'mslp':
                 if ds.mslp.mean().values > 10000:
                     ds['mslp'] = ds['mslp']/100 #transform Pa to HPa/mbar
            
    
            if not var in ds.data_vars.keys():
                raise DWTError("Variable {0} not found in dataset {1}. \
                Available vars: {2}. Rename variables to match file name".format(
                    var, p_file, list(ds.data_vars.keys())))
    
            expected_dims = ['time', 'longitude', 'latitude']
            correct_dims = all(dim in ds.dims for dim in expected_dims)
            if not correct_dims:
                print(ds.dims)
                raise DWTError("Incorrect dimensions found in dataset. Expected dimensions: {0}'".format(expected_dims))
    
            
            ds['longitude'] = (ds.longitude + 360) % 360 #Make sure longitudes are 0-360
            ds = ds.sortby(['longitude', 'latitude', 'time'])
            ds = ds.transpose('time', 'latitude', 'longitude')

            if area[1]<area[0]: #Area goes over 0
                ds = ds.sel(latitude = slice(area[2]-5, area[3]+5))
                ds = ds.isel(longitude = np.where((ds.longitude.values>area[0]-5) | (ds.longitude.values<area[1]+5))[0])
            else:
                ds = ds.sel(longitude = slice(area[0]-5, area[1]+5,),
                            latitude = slice(area[2]-5, area[3]+5))
    
            if ivar == 0:
                DS = ds.copy()
            else:
                ds = ds.interp_like(DS)
                DS[var] = (('time', 'latitude', 'longitude'), ds[var].values)
    
            #Coarsen
            if self.resample_factor>1:
                DS = DS.coarsen(longitude = resample_factor, 
                                latitude = resample_factor, 
                                boundary = 'pad').mean()

        self.data = DS

        
        #If mslp_grad calculate spatial sea level pressure gradients (proportional to wind)
        if mslp_grad:
            self.spatial_gradient(var_name = 'mslp')
        
        self.data = DS

        
        # Call dynamic_estela_predictor if estela is True
        if estela:
            DS = self.data
            ESTELA = xr.open_dataset('estela_sea.nc',decode_times=False)   # ESTELA file
            #ESTELA = ESTELA.sel(time='ALL')
            ESTELA = ESTELA.assign({'estela_mask':(('latitude','longitude'),np.where(ESTELA.F.values>0,1,np.nan))})
    
            # select ESTELA at site grid
            ESTELA_site = ESTELA.sortby(ESTELA.latitude,ascending=False).interp(coords={'latitude':DS.latitude,'longitude':DS.longitude})
            
            # apply ESTELA mask to SLP data
            DS['mslp'] = DS.mslp * ESTELA_site.estela_mask
            DS['mslp_grad'] = DS.mslp_grad * ESTELA_site.estela_mask
            DS['mask_land'] = ESTELA_site.estela_mask

            # generate estela predictor
            estela_D = ESTELA_site.traveltime # .drop('time')
            self.estela_D = estela_D
            self.data = DS
        
            self.data = self.dynamic_estela_predictor()




        #Remove land areas (if only ocean is of interest)
        if remove_land:
            
            mask = 1 - regionmask.defined_regions.natural_earth_v4_1_0.land_110.mask(
                    DS, lon_name = 'longitude', lat_name = 'latitude').fillna(1).rename("lsm")
    
            DS = DS.where(mask.where(mask == 0) == 0)        

        
        if area[1]<area[0]: #Area goes over 0
            DS = DS.sel(latitude = slice(area[2]-5, area[3]+5))
            DS = DS.isel(longitude = np.where((DS.longitude.values>area[0]) | (DS.longitude.values<area[1]))[0])
            #order from -180 to 180
            DS['longitude'] = np.where(DS.longitude.values>180, DS.longitude.values - 360, DS.longitude.values)
            DS = DS.sortby('longitude')
        
        else:
            DS = DS.sel(longitude = slice(area[0]-5, area[1]+5,),
                        latitude = slice(area[2]-5, area[3]+5))
        



    def dynamic_estela_predictor(self):
        '''
        Generate dynamic predictor using estela
    
        xdset:
            (time, latitude, longitude), var_name, mask
    
        returns similar xarray.Dataset with variables:
            (time, latitude, longitude), var_name_comp
            (time, latitude, longitude), var_name_gradient_comp
        '''

        xdset = self.data
        var_name = 'mslp'
        estela_D = self.estela_D 


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
                    ) - int(ed) # np.int
    
                    # select data from displaced time array positions
                    xdselec = xdset.isel(
                        time = i_times,
                        latitude = i_lat,
                        longitude = i_lon)
    
                    # get estela predictor values
                    var_comp[:, i_lat, i_lon] = xdselec[var_name].values
                    var_grd_comp[:, i_lat, i_lon] = xdselec['{0}_grad'.format(var_name)].values
    
        # return generated estela predictor
        return xr.Dataset(
            {
                '{0}'.format(var_name):(
                    ('time','latitude','longitude'), var_comp),
                '{0}_grad'.format(var_name):(
                    ('time','latitude','longitude'), var_grd_comp),
    
            },
            coords = {
                'time':xdset.time.values[first_day:],
                'latitude':xdset.latitude.values,
                'longitude':xdset.longitude.values,
            }
        )



    def spatial_gradient(self, var_name):
        
        '''
        Calculate spatial gradient
    
        xdset:
            (time, 'latitude', 'longitude'), var_name
    
        returns xdset with new variable "var_name_gradient"
        '''
    
        ds = self.data

        var_val = ds[var_name].values
        var_grad = np.zeros_like(var_val)
    
        lat_phi = np.pi * np.abs(ds.latitude.values) / 180.0
    
        for it in range(len(ds.time)):
            m_c = var_val[it, 1:-1, 1:-1]
            m_l = var_val[it, 1:-1, :-2]
            m_r = var_val[it, 1:-1, 2:]
            m_u = var_val[it, :-2, 1:-1]
            m_d = var_val[it, 2:, 1:-1]
    
            dpx1 = (m_c - m_l) / np.cos(lat_phi[None, None, 1:-1, None])
            dpx2 = (m_r - m_c) / np.cos(lat_phi[None, None, 1:-1, None])
            dpy1 = m_c - m_d
            dpy2 = m_u - m_c
    
            vg = (dpx1**2 + dpx2**2) / 2 + (dpy1**2 + dpy2**2) / 2
            var_grad[it, 1:-1, 1:-1] = vg.squeeze()
    
        ds[f'{var_name}_grad'] = (('time', 'latitude', 'longitude'), var_grad)
        self.data = ds[sorted(ds.data_vars)]


        
    def pca(self, n_data = 10, n_pca=5):
    
        pca_ob = PCA()
        pca_ob.dataset = self.data
        pca_ob.dim_pca = self.dim_pca # coordinate to do PCA over
        pca_ob.generate_data_matrix()
        pca_ob.area = self.area
    
        cp = {
               'cmap':'RdBu_r'
                }
        
        pca_ob.plot_data(N=n_data, custom_params = cp)
    
        pca_ob.pca()
    
        pca_ob.plot_eofs_pcs(n_pca = n_pca)

        #TODO: decide wether pca_ob is too much or not
        self.pca_ob = pca_ob
        #self.pca = pca_ob.pca
        
    def sort_bmus(self):
    
        if 'mslp' in list(self.dwt_centroids.data_vars.keys()):
            var_order = 'mslp'
        else:
            var_order = list(self.dwt_centroids.data_vars.keys())[0]
           
        order = np.argsort(self.dwt_centroids[var_order].mean(dim = ('longitude', 'latitude')).values)
        print(order)
    
        data = self.data
        mapping = {val: i for i, val in enumerate(order)}
        v_ordered = [mapping[val] for val in data.bmus.values]
        data['bmus'] = (('time'), v_ordered)
    
        centroids = self.dwt_centroids
        centroids = centroids.isel(bmus = order)
        centroids['bmus'] = (('bmus'), range(len(centroids.bmus)))
        self.dwt_centroids = centroids.sortby('bmus')

    def kma(self, sort_bmus = True):
        
        kma_ob = KMA()
        kma_ob.n_pcs = np.where(self.pca_ob.pca.APEV <= self.explained_variance)[0][-1]
    
        kma_ob.data = self.pca_ob.pca.PCs.isel(n_components = range(kma_ob.n_pcs)).values
        kma_ob.ix_scalar = range(kma_ob.data.shape[1])  # Scalar columns
        kma_ob.ix_directional = [] #Directional columns 
        
        cu, cont = 0, 0
        while np.nanmin(cu) < self.min_data:
            
            kma_ob.kma(self.n_dwt)
            u, cu = np.unique(kma_ob.bmus, return_counts=True)
            cont+=1
            
        kma_ob.num_iter = cont
    
        self.kma_ob = kma_ob
    
        self.data['bmus'] = (('time'), self.kma_ob.bmus)
        self.dwt_centroids = self.data.groupby('bmus').mean(dim='time')
    
        if sort_bmus:
            self.sort_bmus()
            
    
        
    ### Plotting ###  

    def plot_map_features(self, ax, land_color = cfeature.COLORS['land']):
        
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black', color = land_color)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha = .5)

    def plot_map_mean(self):

        ds = self.data
        p_site = self.p_site
        vars = list(ds.data_vars.keys())
        area = self.area
        estela = self.estela
        
        if area[1]<area[0]:
            central_longitude = 0
        else:
            central_longitude = 180
    
    
        if len(vars)>3:
            nrows = ncols = int(np.ceil(np.sqrt(len(vars))))
        else:
            ncols = len(vars); nrows = 1
        
        fig, axs = plt.subplots(nrows, ncols, figsize = [15, 7], 
                                subplot_kw={'projection': ccrs.PlateCarree(central_longitude = central_longitude)})
        
        plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)
        
        if (nrows==1) & (ncols == 1):
            unique_ax = True 
        else:
            axs_list = axs.flatten()
            unique_ax = False
        
        for iv, var in enumerate(vars):

            if not var in config_variables.keys():
                config_variables[var] = config_variables['other']
        
            if unique_ax:
                ax = axs
            else:
                ax = axs_list[iv]
            
            self.plot_map_features(ax)

            vmin, vmax = config_variables[var]['limits'][0], config_variables[var]['limits'][1]
            
            contour = ax.contourf(ds.longitude, ds.latitude, ds[var].mean(dim = 'time'), 50,
                                  cmap=config_variables[var]['cmap'], transform=ccrs.PlateCarree(), 
                                  vmin = vmin, vmax = vmax)

            if not estela: 
                ax.set_extent([ds.longitude.values.min()-10, ds.longitude.values.max()+10, 
                              ds.latitude.values.min()-10, ds.latitude.values.max()+10, ], crs=ccrs.PlateCarree())
                
            else:                    
                if area[1]<area[0]: #Area goes over 0:
                    ax.set_extent([area[0]-360-5, area[1]+5, 
                                  area[2]-10, area[3]+10, ], crs=ccrs.PlateCarree())
                else: 
                    ax.set_extent([area[1]-10, area[0]+10, 
                                  area[2]-10, area[3]+10, ], crs=ccrs.PlateCarree())
            
                

            plt.colorbar(contour, ax = ax, orientation = 'horizontal', shrink = .7)
            ax.set_title(var)

        fig.savefig(op.join(p_site, 'plot_map_mean.png'), bbox_inches='tight')

        with open(op.join(p_site, 'plot_map_mean.pkl'), 'wb') as f:
            pickle.dump(fig, f)



    def plot_dwts(self, anomaly = False):   

        data_means = self.data.mean(dim = 'time')
        ds = self.dwt_centroids
        vars = list(ds.data_vars.keys())
        n_dwts = len(self.dwt_centroids.bmus.values)

        p_site = self.p_site
        
        area = self.area
        if area[1]<area[0]:
            central_longitude = 0
        else:
            central_longitude = 180
        
        if n_dwts>3:
            nrows = ncols = int(np.ceil(np.sqrt(n_dwts)))
        else:
            ncols = n_dwts; nrows = 1

        
        for iv, var in enumerate(vars):  
        
            fig, axs = plt.subplots(nrows, ncols, figsize = [20, 18/(len(ds.longitude.values)/len(ds.latitude.values))], 
                                subplot_kw={'projection': ccrs.PlateCarree(central_longitude = central_longitude)})
            
            if (nrows==1) & (ncols == 1):
                unique_ax = True 
            else:
                axs_list = axs.flatten()
                unique_ax = False
        
            plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.01, wspace=0.05, hspace=0.05)
        
            if not var in config_variables.keys():
                config_variables[var] = config_variables['other']

            norm = None
            if anomaly:
                total = ds[var].values - data_means[var].values
                lim = np.nanmax([np.abs(np.nanpercentile(total, 1)), np.abs(np.nanpercentile(total, 99))])
                vmin, vmax = -lim, lim
                cmap = 'RdBu_r'
            else:
                #vmin, vmax = config_variables[var]['limits'][0], config_variables[var]['limits'][1]
                vmin, vmax = np.nanpercentile(ds[var],1), np.nanpercentile(ds[var],99)
                if var == 'mslp':
                    norm = TwoSlopeNorm(vmin=vmin, vcenter=1014, vmax=vmax)
                cmap = config_variables[var]['cmap']

            np_colors_int = get_cluster_colors(n_dwts)
            
            for d in range(n_dwts):
                
                if unique_ax:
                    ax = axs
                else:
                    ax = axs_list[d]

                color_css = mcolors.to_hex(np.array([np.append(np_colors_int[d], .5) ]), keep_alpha=True)
                
                self.plot_map_features(ax, land_color = color_css)
        
                if anomaly:
                    var_plot = ds[var].isel(bmus = d) - data_means[var]
                else:
                    var_plot = ds[var].isel(bmus = d)

                if norm:
                    contour = ax.pcolor(ds.longitude, ds.latitude, var_plot, 
                                          cmap=cmap, transform=ccrs.PlateCarree(), 
                                          norm = norm)
                else:
                    contour = ax.pcolor(ds.longitude, ds.latitude, var_plot, 
                                          cmap=cmap, transform=ccrs.PlateCarree(), 
                                          vmin = vmin, vmax = vmax)

                # Add the number of each DWT
                ax.text(0.05, 0.05, d + 1, ha='left', va='bottom', fontsize=15, fontweight='bold', color='navy', transform=ax.transAxes)
            
            cbar = plt.colorbar(contour, ax = axs_list, orientation = 'vertical', shrink = .6)
            cbar.set_label(config_variables[var]['label'], fontsize = 16)





            fig.savefig(op.join(p_site, 'plot_dwts_{}_anomaly_{}.png'.format(var, anomaly)), bbox_inches='tight')
            with open(op.join(p_site, 'plot_dwts_{}_anomaly_{}.pkl'.format(var, anomaly)), 'wb') as f:
                pickle.dump(fig, f)
            
            plt.show()
            print('\n\n')
            

    def get_years_months_days(self, time):
        '''
        Returns years, months, days of time in separete lists
    
        (Used to avoid problems with dates type)
        '''
    
        t0 = time[0]
        if isinstance(t0, (date, datetime, DatetimeGregorian)):
            ys = np.asarray([x.year for x in time])
            ms = np.asarray([x.month for x in time])
            ds = np.asarray([x.day for x in time])
    
        else:
            tpd = pd.DatetimeIndex(time)
            ys = tpd.year
            ms = tpd.month
            ds = tpd.day
    
        return ys, ms, ds
        
    def ClusterProbabilities(self, series, set_values):
        'return series probabilities for each item at set_values'
    
        us, cs = np.unique(series, return_counts=True)
        d_count = dict(zip(us,cs))
    
        # cluster probabilities
        cprobs = np.zeros((len(set_values)))
        for i, c in enumerate(set_values):
           cprobs[i] = 1.0*d_count[c]/len(series) if c in d_count.keys() else 0.0
    
        return cprobs


    def ClusterProbs_Month(self, bmus, time, wt_set, month_ix):
        'Returns Cluster probs by month_ix'
    
        # get months
        _, months, _ = self.get_years_months_days(time)
    
        if isinstance(month_ix, list):
    
            # get each month indexes
            l_ix = []
            for m_ix in month_ix:
                ixs = np.where(months == m_ix)[0]
                l_ix.append(ixs)
    
            # get all indexes     
            ix = np.unique(np.concatenate(tuple(l_ix)))
    
        else:
            ix = np.where(months == month_ix)[0]
    
        bmus_sel = bmus[ix]
    
        return self.ClusterProbabilities(bmus_sel, wt_set)

    def axplot_wt_probs(self, ax, wt_probs,
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

    def axplot_wt_hist(self, ax, bmus, n_clusters, ttl=''):
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
    
    

    def ClusterProbabilities(self, series, set_values):
        'return series probabilities for each item at set_values'
    
        us, cs = np.unique(series, return_counts=True)
        d_count = dict(zip(us,cs))
    
        # cluster probabilities
        cprobs = np.zeros((len(set_values)))
        for i, c in enumerate(set_values):
           cprobs[i] = 1.0*d_count[c]/len(series) if c in d_count.keys() else 0.0
    
        return cprobs


    def plot_dwts_probs(self, vmax = 0.15, vmax_seasonality = 0.15):
    
        '''
        Plot Daily Weather Types bmus probabilities
        '''
    
        bmus = self.data.bmus.values
        bmus_time = self.data.time.values
        n_clusters = len(self.dwt_centroids.bmus.values)
        p_site = self.p_site
    
        wt_set = np.arange(n_clusters) + 1
    
        # best rows cols combination
        if n_clusters>3:
            n_rows = n_cols = int(np.ceil(np.sqrt(n_clusters)))
        else:
            n_cols = n_clusters; n_rows = 1
    
        # figure
        fig = plt.figure(figsize=(15, 9))
    
        # layout
        gs = gridspec.GridSpec(4, 7, wspace=0.10, hspace=0.25)
    
        # list all plots params
        l_months = [
            (1, 'January',   gs[1,3]),
            (2, 'February',  gs[2,3]),
            (3, 'March',     gs[0,4]),
            (4, 'April',     gs[1,4]),
            (5, 'May',       gs[2,4]),
            (6, 'June',      gs[0,5]),
            (7, 'July',      gs[1,5]),
            (8, 'August',    gs[2,5]),
            (9, 'September', gs[0,6]),
            (10, 'October',  gs[1,6]),
            (11, 'November', gs[2,6]),
            (12, 'December', gs[0,3]),
        ]
    
        l_3months = [
            ([12, 1, 2],  'DJF', gs[3,3]),
            ([3, 4, 5],   'MAM', gs[3,4]),
            ([6, 7, 8],   'JJA', gs[3,5]),
            ([9, 10, 11], 'SON', gs[3,6]),
        ]
    
        # plot total probabilities
        c_T = self.ClusterProbabilities(bmus, wt_set)
        C_T = np.reshape(c_T, (n_rows, n_cols))
    
        ax_probs_T = plt.subplot(gs[:2, :2])
        pc = self.axplot_wt_probs(ax_probs_T, C_T, ttl = 'DWT Probabilities')
    
        # plot counts histogram
        ax_hist = plt.subplot(gs[2:, :3])
        self.axplot_wt_hist(ax_hist, bmus, n_clusters, ttl = 'DWT Counts')
    
        # plot probabilities by month
        
        for m_ix, m_name, m_gs in l_months:
    
            # get probs matrix
            c_M = self.ClusterProbs_Month(bmus, bmus_time, wt_set, m_ix)
            C_M = np.reshape(c_M, (n_rows, n_cols))
    
            # plot axes
            ax_M = plt.subplot(m_gs)
            self.axplot_wt_probs(ax_M, C_M, ttl = m_name, vmax=vmax)
    
    
        # plot probabilities by 3 month sets
        
        for m_ix, m_name, m_gs in l_3months:
    
            # get probs matrix
            c_M = self.ClusterProbs_Month(bmus, bmus_time, wt_set, m_ix)
            C_M = np.reshape(c_M, (n_rows, n_cols))
    
            # plot axes
            ax_M = plt.subplot(m_gs)
            self.axplot_wt_probs(ax_M, C_M, ttl = m_name, vmax=vmax_seasonality, cmap='Greens')
    
        # add custom colorbar
        pp = ax_probs_T.get_position()
        cbar_ax = fig.add_axes([pp.x1+0.02, pp.y0, 0.02, pp.y1 - pp.y0])
        cb = fig.colorbar(pc, cax=cbar_ax, cmap='Blues')
        cb.ax.tick_params(labelsize=8)

        fig.savefig(op.join(p_site, 'plot_dwts_probs.png'), bbox_inches='tight')
        with open(op.join(p_site, 'plot_dwts_probs.pkl'), 'wb') as f:
            pickle.dump(fig, f)

    def Generate_PerpYear_Matrix(self, num_clusters, bmus_values, bmus_dates,
                                 num_sim=1, month_ini=1):
        '''
        Calculates and returns matrix for stacked bar plotting
    
        bmus_dates - datetime.datetime (only works if daily resolution)
        bmus_values has to be 2D (time, nsim)
        '''
    
        # generate perpetual year list
        list_pyear = self.GenOneYearDaily(month_ini=month_ini)
    
        # generate aux arrays
        m_plot = np.zeros((num_clusters, len(list_pyear))) * np.nan
        bmus_dates_months = np.array([pd.to_datetime(d).month for d in bmus_dates])
        bmus_dates_days = np.array([pd.to_datetime(d).day for d in bmus_dates])
    
        # sort data
        for i, dpy in enumerate(list_pyear):
            _, s = np.where(
                [(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)]
            )
    
            b = bmus_values[s]
            b = b.flatten()
    
            for j in range(num_clusters):
                _, bb = np.where([(j+1 == b)])  # j+1 starts at 1 bmus value!
    
                m_plot[j,i] = float(len(bb)/float(num_sim))/len(s)
    
        return m_plot
    
    def GenOneYearDaily(self, yy=1981, month_ini=1):
        'returns one generic year in a list of datetimes. Daily resolution'
    
        dp1 = datetime(yy, month_ini, 1)
        dp2 = dp1+timedelta(days=365)
    
        return [dp1 + timedelta(days=i) for i in range((dp2-dp1).days)]


    def plot_perpetual_year(self):
    
        num_clusters = len(self.dwt_centroids.bmus)
        bmus_values = self.data.bmus.values
        bmus_dates = self.data.time.values
        month_ini = 1
        p_site = self.p_site
    
        'axes plot bmus perpetual year'
    
        # get cluster colors for stacked bar plot
        np_colors_int = get_cluster_colors(num_clusters)


        # generate dateticks
        x_val = self.GenOneYearDaily(month_ini = month_ini)
    
        # generate plot matrix
        m_plot = self.Generate_PerpYear_Matrix(
            num_clusters, bmus_values + 1, bmus_dates,
            month_ini = month_ini)
    
        fig, ax = plt.subplots(1, figsize = (15, 5))
    
        # plot stacked bars
        bottom_val = np.zeros(m_plot[1,:].shape)
        for r in range(num_clusters):
            row_val = m_plot[r,:]
            ax.bar(
                x_val, row_val, bottom = bottom_val,
                width = 1, color = np.array([np_colors_int[r]]),
                alpha = .7
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

        fig.savefig(op.join(p_site, 'plot_perpetual_year.png'), bbox_inches='tight')
        with open(op.join(p_site, 'plot_perpetual_year.pkl'), 'wb') as f:
            pickle.dump(fig, f)


class DWTError(Exception):
    """Custom exception for DWT class."""
    def __init__(self, message="DWT error occurred."):
        self.message = message
        super().__init__(self.message)