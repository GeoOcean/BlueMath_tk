'''

- Project: Bluemath{toolkit}.datamining
- File: pca.py
- Description: PCA algorithm
- Author: GeoOcean Research Group, Universidad de Cantabria
- Created Date: 23 January 2024
- License: MIT
- Repository: https://gitlab.com/geoocean/bluemath/toolkit/

'''

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import itertools

# Principal component analysis
from sklearn.decomposition import PCA as PCAsk

import cartopy.crs as ccrs
import cartopy.feature as cfeature

class PCA:

    '''
    This class implements the Princial Component Algorithm (PCA)
    
    '''

    def __init__(self):
        self.data = []
        self

    def generate_data_matrix(self):
        '''
        Generate data matrix for PCA analysis
        '''

        ds = self.dataset

        coords_stack = list(ds.dims)
        coords_stack.remove(self.dim_pca)
    
        self.coords_extra = coords_stack
        
        self.vars_keys = list(ds.data_vars.keys()) #Variables involved in PCA analysis
        
        self.data_matrix = np.hstack([ds.stack(positions=(self.coords_extra))[var].values 
                            for var in self.vars_keys])

          
    def reshape_eofs(self, EOFs, n_components, num_splits):
        assert EOFs.shape[0] % num_splits == 0, "Total number of elements must be divisible by num_splits"
    
        EOFs_splits = np.array_split(EOFs, num_splits)

        coords_extra = self.coords_extra
        ds = self.dataset

        if len(coords_extra) == 2:
            c1, c2 = np.meshgrid(ds[coords_extra[1]].values, ds[coords_extra[0]].values)
            EOFs_reshaped_splits = [split.reshape((c1.shape[0], c1.shape[1], n_components)) for split in EOFs_splits]
        elif len(coords_extra) == 1:
            c1 = ds[coords_extra[0]].values
            EOFs_reshaped_splits = [split.reshape((c1.shape[0], n_components)) for split in EOFs_splits]
            
        return EOFs_reshaped_splits

    def reshape_vars(self, var, num_splits):
        
        assert var.shape[0] % num_splits == 0, "Total number of elements must be divisible by num_splits"
    
        var_splits = np.array_split(var, num_splits)

        coords_extra = self.coords_extra
        ds = self.dataset

        if len(coords_extra) == 2:
            c1, c2 = np.meshgrid(ds[coords_extra[1]].values, ds[coords_extra[0]].values)
            EOFs_reshaped_splits = [split.reshape((c1.shape[0], c1.shape[1])) for split in var_splits]
            
        elif len(coords_extra) == 1:
            c1 = ds[coords_extra[0]].values
            EOFs_reshaped_splits = [split.reshape((c1.shape[0])) for split in var_splits]
        
        return EOFs_reshaped_splits
            
    def pca(self):

        matrix = self.data_matrix
        dim_pca = self.dim_pca
        n_vars = len(self.vars_keys)

        coords_extra = self.coords_extra
        vars_keys = self.vars_keys
        ds = self.dataset

        # remove nans
        data_pos = ~np.isnan(matrix[0,:])
        clean_row = matrix[0, data_pos]
        dp_ur_nonan = np.nan * np.ones(
            (matrix.shape[0], len(clean_row))
        )
        matrix_nonan = matrix[:, data_pos]

        # standarize matrix
        self.pred_mean = np.nanmean(matrix_nonan, axis=0) + 0.000000001
        self.pred_std = np.nanstd(matrix_nonan, axis=0) + 0.000000001
        matrix_pca_norm = (matrix_nonan[:,:] - self.pred_mean) / self.pred_std
        
        pca = PCAsk().fit(matrix_pca_norm) 

        eofs = pca.components_.T
        
        eofs_total = np.full((matrix.shape[1], eofs.shape[1]), np.nan)
        eofs_total[np.where(data_pos == True)[0],:] = eofs
        
        EOFs_reshaped_splits = self.reshape_eofs(eofs_total, len(pca.explained_variance_), n_vars)
        
        #eofs reorder
        #EOFs_reshaped_splits = self.reshape_eofs(pca.components_.T, len(pca.explained_variance_), n_vars)

        means = np.full(matrix.shape[1], np.nan)
        means[np.where(data_pos == True)[0]] = self.pred_mean

        stds = np.full(matrix.shape[1], np.nan)
        stds[np.where(data_pos == True)[0]] = self.pred_std

        mean_splits = self.reshape_vars(means,  n_vars)
        std_splits = self.reshape_vars(stds,  n_vars)

        #APEV: the cummulative proportion of explained variance by ith PC
        APEV = np.cumsum(pca.explained_variance_) / np.sum(pca.explained_variance_)*100.0
       
        xds_PCA = xr.Dataset(
            {
                'PCs': ((dim_pca, 'n_components'), pca.transform(matrix_pca_norm)),
                'EOFs': (('var', *self.coords_extra, 'n_components'), EOFs_reshaped_splits),
                'variance': (('n_components',), pca.explained_variance_),
                'means': (('var', *self.coords_extra), mean_splits),
                'stds': (('var', *self.coords_extra), std_splits), 
                'var': (('var',), self.vars_keys), 
                'APEV': (('n_components',), APEV),
            }
        )
        xds_PCA.update({coord: ds[coord] for coord in coords_extra})
        
        self.pca = xds_PCA
    
    
     ### Plotting ###   
    default_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    def plot_data(self, N = 10, custom_params = None):

        '''

        N : Number of 2D maps to plot
        '''

        ds = self.dataset
        vars_keys = self.vars_keys
        coords_extra = self.coords_extra
        dim_pca = self.dim_pca

        plot_defaults = {
            'figsize': (len(vars_keys)*8, 15),
            'fontsize' : 12,
            'cmap':'rainbow'
        }

        plot_params = {**plot_defaults, **custom_params} if custom_params else plot_defaults

        
        
        if len(coords_extra) == 2: #2d maps
        
            fig , axs = plt.subplots(1, len(vars_keys), figsize=plot_params['figsize'], subplot_kw={'projection': '3d'})
            
            for iv, var in  enumerate(vars_keys):
        
                if len(vars_keys)>1:
                    ax = axs[iv]
                else:
                    ax=axs
        
                for iplot in range(np.nanmin([N, len(ds[dim_pca])])):
                    
                    c1, c2 = np.meshgrid(ds[coords_extra[1]].values, ds[coords_extra[0]].values)
                    
                    X_slice = ds.isel({dim_pca: iplot})[var].values

                    norm = Normalize(np.nanmin(X_slice), np.nanmax(X_slice))
                    cmap = plt.get_cmap(plot_params['cmap'])
                    cmap.set_bad('white')
                    colors = cmap(norm(X_slice))
                    im = ax.plot_surface(c1, iplot, c2, facecolors = colors, alpha = .9, ec = None,
                                         vmin = np.nanmin(ds[var].values), vmax = np.nanmax(ds[var].values))
                    
                ax.view_init(elev=20, azim=-40)
                ax.grid(False)
                ax.set_xlabel(coords_extra[0], fontsize = plot_params['fontsize'])
                ax.set_ylabel(f'{dim_pca} [PCA dim]', fontsize = 12, color = 'darkred')
                ax.set_zlabel(coords_extra[1], fontsize = plot_params['fontsize'])
                sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(plot_params['cmap']), norm=norm)
                cbar = fig.colorbar(sm, ax = ax, orientation = 'horizontal', shrink = .6, pad = .02)
        
                cbar.set_label(var, fontsize = plot_params['fontsize'])
                ax.tick_params(axis='both', colors='grey')
        
        elif len(coords_extra) == 1:
            fig , axs = plt.subplots(len(vars_keys), 1, figsize=plot_params['figsize'])

            for iv, var in  enumerate(vars_keys):
            
                if len(vars_keys)>1:
                    ax = axs[iv]
                else:
                    ax=axs
            
                for iplot in range(np.nanmin([N, len(ds[dim_pca])])):
                    
                    c1 = ds[coords_extra[0]].values
                    
                    X_slice = ds.isel({dim_pca: iplot})[var].values
            
                    norm = Normalize(0, np.nanmin([N, len(ds[dim_pca])]))
                    colors = plt.get_cmap(plot_params['cmap'])(norm(iplot))
                    if iplot == 0:
                        ax.plot(c1,  X_slice, color = colors, label = 'Dim PCA')
                    else:
                        ax.plot(c1,  X_slice, color = colors)
                
                ax.legend()
                ax.grid(False)
                ax.set_xlabel(coords_extra[0], fontsize = plot_params['fontsize'])
                ax.set_ylabel(var, fontsize = plot_params['fontsize'])
                ax.tick_params(axis='both', colors='grey')
                
                sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(plot_params['cmap']), norm=norm)
                cbar = fig.colorbar(sm, ax = ax, orientation = 'vertical', shrink = .6)
                cbar.set_label('Dim PCA')

    def plot_map_features(self, ax):
        
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black')
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha = .2)
        
    def plot_eofs(self, n_pca = 3, pc_sel = None, custom_params = None, plot_map = False, central_longitude = 180):
    
        coords_extra = self.coords_extra
        vars_keys = self.vars_keys
        dim_pca = self.dim_pca
        ds = self.dataset
        vars = self.vars_keys
        n_vars = len(vars)

        '''
        area = self.area
        if area[1]<area[0]:
            central_longitude = 0
        '''

        if pc_sel: 
            it=[pc_sel]
        elif pc_sel == 0:
            it = [pc_sel]
        else:
            it = range(n_pca)
            
        plot_defaults = {
                'figsize': (n_vars*8, len(it)*7),
                'fontsize' : 12,
                'cmap':'RdBu_r'
            }
    
        if len(coords_extra)>2:
                raise PCAError("Only up to 2D map plots are available for plottinh")
    
        plot_params = {**plot_defaults, **custom_params} if custom_params else plot_defaults

        if plot_map:
            fig, axs = plt.subplots(len(it), n_vars, figsize = plot_params['figsize'],
                                subplot_kw={'projection': ccrs.PlateCarree(central_longitude = central_longitude)})
        else:
            fig , axs = plt.subplots(len(it), n_vars, figsize= plot_params['figsize'])

        if len(coords_extra) == 2:
            c1, c2 = np.meshgrid(ds[coords_extra[1]].values, ds[coords_extra[0]].values)
        if len(coords_extra) == 1:
            c1 = ds[coords_extra[0]].values

        for pc in it:
            for ivar, var in enumerate(vars):
        
                if (n_vars == 1) & (len(it) ==1):
                    ax = axs
                elif n_vars == 1:
                    ax = axs[pc]
                elif len(it) == 1:
                    ax = axs[ivar]
                else:
                    ax = axs[pc, ivar]

                var_plot = self.pca.EOFs.isel(n_components = pc, var = ivar)

                lim = np.nanmax([np.abs(np.nanmin(var_plot)),np.abs(np.nanmax(var_plot))])

                if len(coords_extra) == 2:
                    if plot_map:
                        im = ax.pcolor(c1, c2, var_plot, cmap = plot_params['cmap'], 
                                  vmin = -lim, vmax = lim, transform=ccrs.PlateCarree())
                        ax.set_extent([ds.longitude.values.min()-10, ds.longitude.values.max()+10, 
                          ds.latitude.values.min()-10, ds.latitude.values.max()+10, ], crs=ccrs.PlateCarree())
                        self.plot_map_features(ax)
                    else:
                        im = ax.pcolor(c1, c2, var_plot, cmap = plot_params['cmap'], vmin = -lim, vmax = lim)

                    plt.colorbar(im, ax = ax, orientation = 'horizontal', shrink = .6).set_label(var)
                    
                elif len(coords_extra) == 1:
                    ax.plot(c1, var_plot, color = 'grey')
                    ax.scatter(c1, var_plot, c=var_plot, cmap = plot_params['cmap'], vmin = -lim, vmax = lim)
                    ax.grid(':', color = 'lightgrey')
                    
                ax.set_ylabel(f'EOF{pc}', fontsize = plot_params['fontsize'])
                ax.set_title(f'{var}', fontsize = plot_params['fontsize'])

        fig.suptitle(f'EOF{pc}', fontsize = plot_params['fontsize'])      
    
    def plot_pcs(self, n_pca = 3, pc_sel = None, custom_params = None):

        vars_keys = self.vars_keys
        dim_pca = self.dim_pca
        ds = self.dataset
        

        if pc_sel: 
            it=[pc_sel]
        else:
            it = range(n_pca)
            
        plot_defaults = {
                'figsize': (15, len(it)*4),
                'fontsize' : 12,
            }
        plot_params = {**plot_defaults, **custom_params} if custom_params else plot_defaults
    
        fig , axs = plt.subplots(len(it), 1, figsize=(15, len(it)*2), sharex = True)
        
        for pc in it:

            if len(it)==1:
                ax = axs
            else:
                ax = axs[pc]

            ax.grid(color = 'lightgrey')
            ax.plot(ds[dim_pca].values, self.pca.PCs.isel(n_components = pc), color = self.default_colors[2])
            ax.set_ylabel(f'PC{pc}', fontsize = 12)
            ax.set_xlim(ds[dim_pca].values[0], ds[dim_pca].values[-1])
    
    def plot_eofs_pcs(self, n_pca = 1, central_longitude = 180):

        ds = self.dataset
        spatial_dims = ['longitude', 'latitude']
        plot_map = all(dim in ds.dims for dim in spatial_dims)
    
        for n in range(n_pca):

            self.plot_eofs(pc_sel = n, plot_map = plot_map, central_longitude = central_longitude)
            self.plot_pcs(pc_sel = n)
            
class PCAError(Exception):
    """Custom exception for PCA class."""
    def __init__(self, message="PCA error occurred."):
        self.message = message
        super().__init__(self.message)


