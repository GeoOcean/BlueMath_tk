import os
import os.path as op
import glob
from natsort import natsorted

import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature

from olas import estela

def preprocess_monthly_files(ds):
    filename = os.path.basename(ds.encoding['source'])
    year_month = filename.split('_')[1][:6]
    timenew = np.array(xr.DataArray(pd.to_datetime(year_month, format='%Y%m')))
    ds['time'] = (('time'), [timenew])
    return ds

def preprocess_monthly_files_sea(ds):
    filename = os.path.basename(ds.encoding['source'])
    year_month = filename.split('_')[2][:6]
    timenew = np.array(xr.DataArray(pd.to_datetime(year_month, format='%Y%m')))
    ds['time'] = (('time'), [timenew])
    return ds

plot_defaults = {
        'extent' : None,
        'figsize' : None, #[20, 8],
        'vmin': 0,
        'vmin_anomaly': None,
        'vmax': None,
        'vmax_anomaly': None,
        'central_longitude' : 180,
        'cbar': True,
        'marker': '.',
        'fontsize' : 12,
        'cmap' : 'rainbow',
        'cmap_anomaly' : 'RdBu_r',
        'c_color': 'plum',
        'alpha':1
    }

    
class ESTELA:

    '''
    This class implements the calculation of ESTELA using the python that can be found at: https://github.com/jorgeperezg/olas, which
    is based on the following publication [Pérez et al., 2014](https://doi.org/10.1007/s10236-014-0740-7)
    
    '''

    def __init__(self, p_waves, save_monthly = False, only_sea = False):
        
        self.p_waves = p_waves
        self.save_monthly = save_monthly
        self.only_sea = only_sea
        self.site = None
        self.lon0 = None
        self.lat0 = None
        self.PATHS = None

        #These wave parameter names are for CSIRO partitions. If other database is used, these should be changed.
        self.waves_parsm = vars = ['MAPSTA', 'hs0', 'hs1', 'hs2', 'hs3',
                                            'si0', 'si1', 'si2', 'si3',
                                            'th0', 'th1', 'th2', 'th3',
                                            'tp0', 'tp1', 'tp2', 'tp3']

        if save_monthly:
            self.groupers = ["time.month"]
        else:
            self.groupers = ["ALL", "time.season",  "time.month"] 

    def set_site(self, site, location):

        self.site = site
        self.location = location
        self.lon0 = location[0]
        self.lat0 = location[1]

        if location[0] < 0:
            raise ESTELAError('Location longitude must be within [0-360]. Location given: {0}'.format(location))

    def set_years(self, years):

        p_waves = self.p_waves
        y_start = years[0]
        y_end = years[1]
        
        self.years = years
        self.y_start = y_start
        self.y_end = y_end

        # database
        datafiles = natsorted(glob.glob(op.join(p_waves, "*.nc")))
        #Define range of years for the analysis
        years = np.arange(y_start, y_end+1)
        #select paths in those years
        PATHS = []
        for s in datafiles:
            for y in years:
                if y.astype('str') in s.split('.nc')[0].split('.')[-1][:-2]:
                    PATHS.append(s)
        
        self.PATHS = natsorted(PATHS)


    
    def get_estela(self, nblocks = 1, do_estela = True):

        if not (self.site is not None or self.lon0 is not None or self.lat0 is not None or self.PATHS is not None):
            raise ESTELAError('Site and Years need to be defined before using the function set_location and set_years')
            
    
        PATHS = self.PATHS
        lon0 = self.lon0
        lat0 = self.lat0
        save_monthly = self.save_monthly
        groupers = self.groupers
        site = self.site
        only_sea = self.only_sea
    
        # Generate paths for saving
    
        p_site = op.join(os.getcwd(), site)
        self.p_site = p_site
        # make output folder
        if not op.isdir(p_site):
            os.makedirs(p_site)

        if only_sea:
            p_estela_o = op.join(p_site, 'estela_sea.nc')
            p_estela_monthly = op.join(p_site, 'monthly_estela_sea.nc')
            name_estela = 'estela_sea'
        else:
            p_estela_o = op.join(p_site, 'estela.nc')
            p_estela_monthly = op.join(p_site, 'monthly_estela.nc')
            name_estela = 'estela'
            
            
        if do_estela: 
            # Calculate estela
            if save_monthly:
                for i in range(len(PATHS)):
                    p_estela = op.join(p_site, '{}_{}.nc'.format(name_estela, PATHS[i].split('/')[-1].split('.')[-2]))
                    
                    p_file_estela = PATHS[i].split('/')[-1].split('.')[-2]
                    y = np.double(p_file_estela[:-2])
                    m = np.double(p_file_estela[-2:])

                    if not op.isfile(p_estela):
                        if only_sea:
                            if ((y ==2013) and (m>=6)) or (y>2013):
                                est = estela.calc(PATHS[i], lat0, lon0, "phs0", "ptp0", "pdir0", "pspr0", groupers=groupers, nblocks=nblocks)
                            else:
                                est = estela.calc(PATHS[i], lat0, lon0, "hs0", "tp0", "th0", "si0", groupers=groupers, nblocks=nblocks)
                        else:
                            if ((y ==2013) and (m>=6)) or (y>2013):
                                est = estela.calc(PATHS[i], lat0, lon0, "phs.", "ptp.", "pdir.", "pspr.", groupers=groupers, nblocks=nblocks)
                            else:
                                est = estela.calc(PATHS[i], lat0, lon0, "hs.", "tp.", "th.", "si.", groupers=groupers, nblocks=nblocks)
                        est.to_netcdf(path = p_estela)

                p_estelas = op.join(p_site, '{}_*.nc'.format(name_estela))
                if only_sea:
                    est = xr.open_mfdataset(p_estelas, preprocess=preprocess_monthly_files_sea)
                else:
                    est = xr.open_mfdataset(p_estelas, preprocess=preprocess_monthly_files)
                est = est.assign({'estela_mask':(('latitude','longitude'), np.where(est.F.isel(time = 0).values > 0.001, 1, np.nan))})
                #est = est.mean(dim = 'time')
                est = est.sortby(est.latitude, ascending = False)
                est.to_netcdf(p_estela_monthly)

                self.p_estela_monthly = p_estela_monthly

                est = est.mean(dim = 'time')
                est.to_netcdf(p_estela_o)
                self.path_estela = p_estela_o
                
            else:
                if not op.isfile(p_estela_o):
                    #est = estela.calc(PATHS, lat0, lon0, "hs.", "tp.", "th.", 20, groupers=groupers, nblocks=nblocks)
                    est = estela.calc(PATHS, lat0, lon0, "hs.", "tp.", "th.", "si.", groupers=groupers, nblocks=nblocks)
                    
                    est = est.assign({'estela_mask':(('latitude','longitude'), np.where(est.F.isel(time = 0).values > 0.001, 1, np.nan))})
                    est = est.sortby(est.latitude, ascending = False)
                    est.to_netcdf(path = p_estela_o)
            
            est = xr.open_dataset(p_estela_o, decode_times = False)
            os.remove(p_estela_o)
            est['traveltime'] = est['traveltime']/3600/10**9/24
            est.to_netcdf(path = p_estela_o)

        else:

            if not op.isfile(p_estela_o):
                raise ESTELAError('Estela file not found at {0}. This could happen because do_estela parameter is set to False, and estela has not been run yet. Consider changing do_estela = True'.format(p_estela_o))
                
            # Load previously calculated estela
            est = xr.open_dataset(p_estela_o, decode_times = False)

        self.estela = est

    def load_estela_monthly(self):

        estela_monthly = xr.open_dataset(self.p_estela_monthly)
        estela_monthly['traveltime'] = estela_monthly['traveltime']/3600/10**9/24

        return estela_monthly

    def load_estela_monthly_mean(self):

        only_sea = self.only_sea

        if only_sea:
            p_estela_monthly_mean = op.join(self.p_site, 'monthly_mean_estela_sea.nc')
        else:
            p_estela_monthly_mean = op.join(self.p_site, 'monthly_mean_estela.nc')
            
        est = self.load_estela_monthly()
        est = est.groupby(est.time.dt.month).mean()
        est['month_name'] = (('month'), ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                            'August', 'September', 'October', 'November', 'December'])
        if os.path.exists(p_estela_monthly_mean): os.remove(p_estela_monthly_mean)
        est.to_netcdf(p_estela_monthly_mean)

        return xr.open_dataset(p_estela_monthly_mean, decode_times = False)

    def load_estela_seasonal(self):

        only_sea = self.only_sea

        if only_sea:
             p_estela_seasonal = op.join(self.p_site, 'seasonal_estela_sea.nc')
        else:
            p_estela_seasonal = op.join(self.p_site, 'seasonal_estela.nc')
        
        estela_monthly = xr.open_dataset(self.p_estela_monthly)
        # Define quarters. DJF MAM JJA SON
        estela_monthly['season'] = (('time'), np.digitize(estela_monthly['time.month'], [3, 6, 9, 12]) % 4)
        
        # Group data
        estela_seasonal = estela_monthly.groupby('season').mean()
        estela_seasonal['season_name'] = (('season'), np.array(['DJF', 'MAM', 'JJA', 'SON']))
        estela_seasonal['traveltime'] = estela_seasonal['traveltime']/3600/10**9/24

        if os.path.exists(p_estela_seasonal): os.remove(p_estela_seasonal)
        estela_seasonal.to_netcdf(p_estela_seasonal)

        return xr.open_dataset(p_estela_seasonal, decode_times = False)

    ##############################################
    ################## Plotting ##################
    ##############################################

    
    def plot_map_features(self, ax, land_color = cfeature.COLORS['land']):
        
        
        #ax.add_feature(cfeature.BORDERS, linestyle=':')
        #ax.add_feature(cfeature.LAND, edgecolor=None, color = land_color, zorder = 1)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha = .5)
        #ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black',  zorder = 2)
        land_10m = cartopy.feature.NaturalEarthFeature(
                            'physical', 'land', '10m',
                            edgecolor = 'black',
                            linewidth = .5,
                            facecolor = cfeature.COLORS['land'],
                            zorder = 1,
                                )
        ax.add_feature(land_10m)
        ax.gridlines()

    def get_extent(self, est, plot_params):
        
        if not plot_params['extent']:
            lon0, lon1 = np.nanmin(est.longitude[np.where(est.estela_mask == 1)[1]]), np.nanmax(est.longitude[np.where(est.estela_mask == 1)[1]])
            lat0, lat1 = np.nanmin(est.latitude[np.where(est.estela_mask == 1)[0]]), np.nanmax(est.latitude[np.where(est.estela_mask == 1)[0]])
            extent = (lon0, lon1, lat0, lat1)
            plot_params['extent'] = extent
        else:
            extent = plot_params['extent']

        return extent

    def get_figsize(self, plot_params, N = 15, ratio = 1):

        if not plot_params['figsize']:
            figsize = [N, N/ratio]
        else:
            figsize = plot_params['figsize']

        return figsize
    
    def plot_estela(self, est, plot_params, anomaly = False, fig = [], ax = []):

        if anomaly:
            mean_estela = self.estela
            energy_mean = mean_estela.F.values #* mean_estela.estela_mask.values

        extent = self.get_extent(est, plot_params)  
        ratio = ((extent[1]-extent[0])/(extent[3]-extent[2]))

     # generate figure and axes
        if not fig:

            figsize = self.get_figsize(plot_params, N = 15, ratio = ratio)
 
            fig = plt.figure(figsize = figsize)
            
            ax = plt.axes(projection = ccrs.PlateCarree(central_longitude = plot_params['central_longitude']))

            
        # cartopy land feature
        self.plot_map_features(ax)
    
        # plot estela
        energy = est.F.values #* est.estela_mask.values
        if anomaly:
            energy -= energy_mean

        if anomaly:
            vmin, vmax = plot_params['vmin_anomaly'], plot_params['vmax_anomaly']
            cmap_energy = plot_params['cmap_anomaly']
        else:
            vmin, vmax = plot_params['vmin'], plot_params['vmax']
            cmap_energy = plot_params['cmap']
            
        p1 = ax.contourf(
            est.longitude.values, est.latitude.values, energy, 100,
            transform = ccrs.PlateCarree(),
            zorder = 2, alpha = plot_params['alpha'],
            vmin = vmin, vmax = vmax,
            cmap = cmap_energy,
        )

        if not anomaly:
            cim = ax.contour(
                est.longitude.values, est.latitude.values, est.traveltime.values * est.estela_mask.values,
                levels = np.arange(1, 26, 1),
                transform = ccrs.PlateCarree(),
                zorder = 3,
                linestyles = ':',
                colors = plot_params['c_color'],
            )
            plt.clabel(cim, inline = True, fontsize = plot_defaults['fontsize'])

        ax.plot(self.lon0, self.lat0, transform = ccrs.PlateCarree(), marker = '*', color = 'black', markersize = 12, zorder = 5)
    
        if plot_params['cbar']:
            plt.colorbar(p1, shrink = .7).set_label('Energy', fontsize = plot_params['fontsize'])
    
        # set extent
        if plot_params['extent']:
            ax.set_extent(plot_params['extent'], crs = ccrs.PlateCarree())

        return p1

    def plot_mean_estela(self, fig = [], ax = [], custom_params = False):
        
        '''
        Plots estela map
    
        self.estela    - (xarray.Dataset) estela dataset [traveltime and F variables]
        '''

        est = self.estela
        
        plot_params = {**plot_defaults, **custom_params} if custom_params else plot_defaults  
        
        self.plot_estela(est, plot_params)
    
    def plot_season_estela(self, fig = [], ax = [], anomaly = False, custom_params = {'cbar' : False}):
        
        '''
        Plots estela map
    
        self.estela    - (xarray.Dataset) estela dataset [traveltime and F variables]
        '''

        est = self.load_estela_seasonal()

        en = est.F - self.estela.F
        lim = np.nanmax([np.abs(np.nanmin(en)), np.abs(np.nanmax(en))])
        plot_defaults['vmin_anomaly'] = -lim
        plot_defaults['vmax_anomaly'] = lim

        plot_defaults['vmax'] = est.F.max()
        
        plot_params = {**plot_defaults, **custom_params} if custom_params else plot_defaults

        extent = self.get_extent(est, plot_params)  
        ratio = ((extent[1]-extent[0])/(extent[3]-extent[2]))

        figsize = self.get_figsize(plot_params, N = 18, ratio = ratio)

        fig, axs = plt.subplots(2, 2, figsize=figsize, 
                                subplot_kw={'projection': ccrs.PlateCarree(central_longitude = plot_params['central_longitude'])})
        
        for i, ax in enumerate(axs.flat):

            ax.set_title('{}'.format(est.isel(season=i).season_name.values), fontsize = plot_params['fontsize'])  # Título del subplot
            im = self.plot_estela(est.isel(season = i), plot_params, fig = fig, ax = ax, anomaly = anomaly)

        if anomaly:
            norm = mcolors.Normalize(vmin=plot_params['vmin_anomaly'], vmax=plot_params['vmax_anomaly'])
            cmap = plot_params['cmap_anomaly']
        else:
            norm = mcolors.Normalize(vmin=plot_params['vmin'], vmax=plot_params['vmax'])
            cmap = plot_params['cmap']

        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax = axs, orientation='vertical', shrink = .7)
        cbar.set_label('Energy', fontsize = plot_params['fontsize'])


    def plot_monthly_estela(self, fig = [], ax = [], anomaly = False, custom_params = {'cbar' : False}):
        
        '''
        Plots estela map
    
        self.estela    - (xarray.Dataset) estela dataset [traveltime and F variables]
        '''

        est = self.load_estela_monthly_mean()
        
        en = est.F - self.estela.F
        lim = np.nanmax([np.abs(np.nanmin(en)), np.abs(np.nanmax(en))])
        plot_defaults['vmin_anomaly'] = -lim
        plot_defaults['vmax_anomaly'] = lim

        plot_defaults['vmax'] = est.F.max()
        
        plot_params = {**plot_defaults, **custom_params} if custom_params else plot_defaults

        extent = self.get_extent(est, plot_params)  
        ratio = ((extent[1]-extent[0])/(extent[3]-extent[2]))

        figsize = self.get_figsize(plot_params, N = 22, ratio = ratio*3/4)

        fig, axs = plt.subplots(4, 3, figsize=figsize, 
                                subplot_kw={'projection': ccrs.PlateCarree(central_longitude = plot_params['central_longitude'])})

        for i, ax in enumerate(axs.flat):

            ax.set_title('{}'.format(est.isel(month=i).month_name.values), fontsize = plot_params['fontsize'])
            im = self.plot_estela(est.isel(month = i), plot_params, fig = fig, ax = ax, anomaly = anomaly)


        if anomaly:
            norm = mcolors.Normalize(vmin=plot_params['vmin_anomaly'], vmax=plot_params['vmax_anomaly'])
            cmap = plot_params['cmap_anomaly']
        else:
            norm = mcolors.Normalize(vmin=plot_params['vmin'], vmax=plot_params['vmax'])
            cmap = plot_params['cmap']

        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax = axs, orientation='vertical', shrink = .7)
        cbar.set_label('Energy', fontsize = plot_params['fontsize'])



class ESTELAError(Exception):
    """Custom exception for ESTELA class."""
    def __init__(self, message="ESTELA error occurred."):
        self.message = message
        super().__init__(self.message)
        