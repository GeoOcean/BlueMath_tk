#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pip
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

# teslakit
from ..storms import GeoAzimuth

# import constants
from .config import _faspect, _fsize, _fdpi

def get_storm_linestyle(vel):
    if vel >= 60:
        return 'solid'
    elif vel >= 40:
        return 'dashed'
    elif vel >= 20:
        return 'dashdot'
    else:
        return 'dotted'

def get_storm_color(categ):

    dcs = {
        0 : 'green',
        1 : 'yellow',
        2 : 'orange',
        3 : 'red',
        4 : 'purple',
        5 : 'black',
    }

    return dcs[categ]


def axplot_Tracks_Circle(ax, lon_point, lat_point, r_point,
                         lon_storms, lat_storms, index_in, index_out,
                         categs, vel):
    'axes plot storm tracks inside radius circle'

    n_storms = len(categs)

    # plot point and  circle
    ax.plot(lon_point, lat_point, '.b', markersize=12, zorder=100)
    circle = plt.Circle(
        (lon_point, lat_point), r_point,
        edgecolor='k', fill=False)
    ax.add_patch(circle)

    # plot storms
    for lon_s, lat_s, i_i, i_o, v, c in zip(
        lon_storms, lat_storms, index_in, index_out, vel, categs):

        # get storm data
        lon_in = lon_s[i_i:i_o+1]
        lat_in = lat_s[i_i:i_o+1]

        # linestyle (velocity) and linecolor (category)
        ls = get_storm_linestyle(v)
        cs = get_storm_color(c)

        # plot storm and point at entry
        ax.plot(lon_in, lat_in, linestyle=ls, color=cs)
        ax.plot(lon_in[0], lat_in[0], '.', markersize=8, color=cs, zorder=99)

    # customize axes
    ax.axis('equal')
    ax.axis('off')
    ax.set_title(
        'Historical Tracks ({0} TCs inside R={1}º)'.format(n_storms, r_point),
        {'fontsize': 14, 'fontweight':'bold'}
    )

def axplot_Parametrized_Circle(ax, lon_point, lat_point, r_point,
                               delta_storms, gamma_storms, categs, vel):
    'axes plot storm parametrized tracks inside radius circle'

    n_storms = len(categs)

    # plot point and  circle
    ax.plot(lon_point, lat_point, '.b', markersize=12, zorder=100)
    circle = plt.Circle(
        (lon_point, lat_point), r_point,
        edgecolor='k', fill=False)
    ax.add_patch(circle)

    # TODO: def functions  track --> params / params --> track 

    # start-end points aux. opts.
    n = 1000
    ang = np.linspace(0, 2*np.pi, n)
    x_ang = r_point * np.cos(ang) + lat_point
    y_ang = r_point * np.sin(ang) + lon_point

    center_radius = np.asarray(
        [GeoAzimuth(lat_point, lon_point, x, y) for x, y in zip(x_ang, y_ang)])

    # plot storms
    for delta, gamma, v, c in zip(delta_storms, gamma_storms, vel, categs):

        # find most similar angle
        im = np.argmin(np.absolute(center_radius - delta))
        lon_entry = y_ang[im]
        lat_entry = x_ang[im]

        ent_radius = np.asarray(
            [GeoAzimuth(x, y, x_ang[im], y_ang[im]) for x, y in zip(
                x_ang, y_ang)])

        im2 = np.argmin(np.absolute(ent_radius - gamma))
        lon_exit = y_ang[im2]
        lat_exit = x_ang[im2]

        # linestyle (velocity) and linecolor (category)
        ls = get_storm_linestyle(v)
        cs = get_storm_color(c)

        # plot storm and point at entry
        ax.plot([lon_entry, lon_exit], [lat_entry, lat_exit], linestyle=ls, color=cs)
        ax.plot(lon_entry, lat_entry, '.', markersize=8, color=cs, zorder=99)

    # customize axes 
    ax.axis('equal')
    ax.axis('off')
    ax.set_title(
        'Parametrized Tracks',
        {'fontsize': 14, 'fontweight':'bold'}
    )

def axlegend_categ(ax):
    'add custom legend (storm category) to axes'

    # category legend
    lls_cat = [
        Line2D([0], [0], color = 'black'),
        Line2D([0], [0], color = 'purple'),
        Line2D([0], [0], color = 'red'),
        Line2D([0], [0], color = 'orange'),
        Line2D([0], [0], color = 'yellow'),
        Line2D([0], [0], color = 'green'),
    ]
    leg_cat = Legend(
        ax, lls_cat, ['5','4','3','2','1','0'],
        title = 'Category', bbox_to_anchor = (1.01, 1), loc='upper left',
    )
    ax.add_artist(leg_cat)

def axlegend_vel(ax):
    'add custom legend (storm velocity) to axes'

    # velocity legend
    lls_vel = [
        Line2D([0], [0], color = 'black', ls = 'solid'),
        Line2D([0], [0], color = 'black', ls = 'dashed'),
        Line2D([0], [0], color = 'black', ls = 'dashdot'),
        Line2D([0], [0], color = 'black', ls = 'dotted'),
    ]
    leg_vel = Legend(
        ax, lls_vel, ['> 60 km/h', '> 40 km/h', '> 20 km/h', '< 20 km/h'],
        title = 'Velocity', bbox_to_anchor = (1.01, 0.7), loc= 'upper left',
    )
    ax.add_artist(leg_vel)


def Plot_TCs_TracksParams(TCs_tracks, TCs_params, nm_lon='lon', nm_lat='lat', show=True):
    'Plots storms tracks and storms parametrized'

    # get data from historical tracks
    n_storms = TCs_tracks.storm.shape[0]
    lon_storms = TCs_tracks[nm_lon].values[:]
    lat_storms = TCs_tracks[nm_lat].values[:]

    # get data from parametrized tracks
    lon_point = TCs_params.point_lon
    lat_point = TCs_params.point_lat
    r_point = TCs_params.point_r

    index_in = TCs_params.index_in.values[:]
    index_out = TCs_params.index_out.values[:]
    categs_storms = TCs_params.category.values[:]
    vel_storms = TCs_params.velocity_mean.values[:]
    gamma_storms = TCs_params.gamma.values[:]
    delta_storms = TCs_params.delta.values[:]

    # figure
    fig, axs = plt.subplots(ncols=2, figsize=(_faspect*_fsize, _fsize))

    # storm tracks
    axplot_Tracks_Circle(
        axs[0], lon_point, lat_point, r_point,
        lon_storms, lat_storms, index_in, index_out,
        categs_storms, vel_storms
    )

    # storms  parametrized
    axplot_Parametrized_Circle(
        axs[1], lon_point, lat_point, r_point,
        delta_storms, gamma_storms, categs_storms, vel_storms
    )

    # small axis lims fix
    axs[0].set_xlim(axs[1].get_xlim())
    axs[0].set_ylim(axs[1].get_ylim())

    # add custom legends
    axlegend_categ(axs[1])
    axlegend_vel(axs[1])

    # show and return figure
    if show: plt.show()
    return fig

def Plot_TCs_HistoricalTracks(xds_TCs_r1, xds_TCs_r2,
                              lon1, lon2, lat1, lat2,
                              pnt_lon, pnt_lat, r1, r2,
                              nm_lon='lon', nm_lat='lat',
                              show=True):
    'Plot Historical TCs tracks map, requires basemap module'

    try:
        from mpl_toolkits.basemap import Basemap
    except:
        print('basemap module required.')
        return

    fig, ax = plt.subplots(1, figsize=(_faspect*_fsize, _fsize))

    # setup mercator map projection.
    m = Basemap(
        llcrnrlon = lon1, llcrnrlat = lat1,
        urcrnrlon = lon2, urcrnrlat = lat2,
        resolution = 'l', projection = 'cyl',
        lat_0 = lat1, lon_0 = lon1, area_thresh = 0.01,
    )
    m.drawcoastlines()
    m.fillcontinents(color = 'silver')
    m.drawmapboundary(fill_color = 'lightcyan')
    m.drawparallels(np.arange(lat1, lat2, 20), labels = [1,1,0,0])
    m.drawmeridians(np.arange(lon1, lon2, 20), labels = [0,0,0,1])

    # plot r1 storms
    for s in range(len(xds_TCs_r1.storm)):
        lon = xds_TCs_r1.isel(storm = s)[nm_lon].values[:]
        lon[np.where(lon<0)] = lon[np.where(lon<0)] + 360

        if s==0:
            ax.plot(
                lon, xds_TCs_r1.isel(storm = s)[nm_lat].values[:],
                '-', color = 'grey', alpha = 0.5,
                label = 'Enter {0}° radius'.format(r1)
            )
        else:
            ax.plot(
                lon, xds_TCs_r1.isel(storm = s)[nm_lat].values[:],
                '-', color = 'grey',alpha = 0.5
            )
            ax.plot(
                lon[0], xds_TCs_r1.isel(storm = s)[nm_lat].values[0],
                '.', color = 'grey', markersize = 10
            )

    # plot r2 storms
    for s in range(len(xds_TCs_r2.storm)):
        lon = xds_TCs_r2.isel(storm = s)[nm_lon].values[:]
        lon[np.where(lon<0)] = lon[np.where(lon<0)] + 360

        if s==0:
            ax.plot(
                lon, xds_TCs_r2.isel(storm = s)[nm_lat].values[:],
                color = 'indianred', alpha = 0.8,
                label = 'Enter {0}° radius'.format(r2)
            )
        else:
            ax.plot(
                lon, xds_TCs_r2.isel(storm = s)[nm_lat].values[:],
                color = 'indianred', alpha = 0.8
            )
            ax.plot(
                lon[0], xds_TCs_r2.isel(storm = s)[nm_lat].values[0],
                '.', color = 'indianred', markersize = 10
            )

    # plot point
    ax.plot(
        pnt_lon, pnt_lat, '.',
        markersize = 15, color = 'brown',
        label = 'STUDY SITE'
    )

    # plot r1 circle
    circle = Circle(
        m(pnt_lon, pnt_lat), r1,
        facecolor = 'grey', edgecolor = 'grey',
        linewidth = 3, alpha = 0.5,
        label='{0}° Radius'.format(r1)
    )
    ax.add_patch(circle)

    # plot r2 circle
    circle2 = Circle(
        m(pnt_lon, pnt_lat), r2,
        facecolor = 'indianred', edgecolor = 'indianred',
        linewidth = 3, alpha = 0.8,
        label='{0}° Radius'.format(r2))
    ax.add_patch(circle2)

    # customize axes
    ax.set_aspect(1.0)
    ax.set_ylim(lat1, lat2)
    ax.set_title('Historical TCs', fontsize=15)
    ax.legend(loc=0, fontsize=14)

    # show and return figure
    if show: plt.show()
    return fig

def Plot_TCs_HistoricalTracks_Category(xds_TCs_r1, cat,
                                      lon1, lon2, lat1, lat2,
                                      pnt_lon, pnt_lat, r1,
                                      nm_lon='lon', nm_lat='lat',
                                      show=True):
    'Plot Historical TCs category map, requires basemap module'

    try:
        from mpl_toolkits.basemap import Basemap
    except:
        print('basemap module required.')
        return

    fig, ax = plt.subplots(1, figsize=(_faspect*_fsize, _fsize))

    # setup mercator map projection.
    m = Basemap(
        llcrnrlon = lon1, llcrnrlat = lat1,
        urcrnrlon = lon2, urcrnrlat = lat2,
        resolution = 'l', projection = 'cyl',
        lat_0 = lat1, lon_0 = lon1, area_thresh=0.01
    )
    m.drawcoastlines()
    m.fillcontinents(color = 'silver')
    m.drawmapboundary(fill_color = 'lightcyan')
    m.drawparallels(np.arange(lat1, lat2, 20), labels = [1,0,0,0])
    m.drawmeridians(np.arange(lon1, lon2, 20), labels = [0,0,0,1])

    for s in range(len(xds_TCs_r1.storm)):
        lon = xds_TCs_r1.isel(storm = s)[nm_lon].values[:]
        lon[np.where(lon<0)] = lon[np.where(lon<0)] + 360

        if s==0:
            ax.plot(
                lon, xds_TCs_r1.isel(storm = s)[nm_lat].values[:],
                '-', color = get_storm_color(int(cat[s].values)),
                alpha = 0.5, label = 'Enter {0}° radius'.format(r1)
            )
        else:
            ax.plot(
                lon, xds_TCs_r1.isel(storm = s)[nm_lat].values[:],
                '-', color = get_storm_color(int(cat[s].values)),
                alpha = 0.5,
            )
            ax.plot(
                lon[0], xds_TCs_r1.isel(storm = s)[nm_lat].values[0],
                '.', color = get_storm_color(int(cat[s].values)),
                markersize = 10,
            )

    # plot point
    ax.plot(
        pnt_lon, pnt_lat, '.',
        markersize = 15, color = 'brown',
        label = 'STUDY SITE'
    )

    # plot circle
    circle = Circle(
        m(pnt_lon, pnt_lat), r1,
        facecolor = 'grey', edgecolor = 'grey',
        linewidth = 3, alpha = 0.5,
        label='Radius {0}º'.format(r1)
    )
    ax.add_patch(circle)

    # customize axes
    ax.set_aspect(1.0)
    ax.set_ylim(lat1,lat2)
    ax.set_title('Historical TCs', fontsize=15)
    ax.legend(loc=0, fontsize=14)
    axlegend_categ(ax)

    # show and return figure
    if show: plt.show()
    return fig

def Plot_TCs_SimTracks_Category(xds_TCs_r1, cat,
                                      lon1, lon2, lat1, lat2,
                                      pnt_lon, pnt_lat, r1,
                                      nm_lon='ylon_TC', nm_lat='ylat_TC',
                                      show=True):
    'Plot Historical TCs category map, requires basemap module'

    try:
        from mpl_toolkits.basemap import Basemap
    except:
        print('basemap module required.')
        return

    fig, ax = plt.subplots(1, figsize=[20,10])

    # setup mercator map projection.
    m = Basemap(
        llcrnrlon = lon1, llcrnrlat = lat1,
        urcrnrlon = lon2, urcrnrlat = lat2,
        resolution = 'l', projection = 'cyl',
        lat_0 = lat1, lon_0 = lon1, area_thresh=0.01
    )
    m.drawcoastlines()
    m.fillcontinents(color = 'silver')
    m.drawmapboundary(fill_color = 'lightcyan')
    m.drawparallels(np.arange(lat1, lat2, 20), labels = [1,0,0,0])
    m.drawmeridians(np.arange(lon1, lon2, 20), labels = [0,0,0,1])

    for s in range(len(xds_TCs_r1.storm)):
        lon = xds_TCs_r1.isel(storm = s)[nm_lon].values.tolist()
        lon[np.where(lon<0)] = lon[np.where(lon<0)] + 360

        if s==0:
            ax.plot(
                lon, xds_TCs_r1.isel(storm = s)[nm_lat].values.tolist(),
                '-', color = get_storm_color(int(cat[s].values)),
                alpha = 0.5, label = 'Enter {0}° radius'.format(r1)
            )
        else:
            ax.plot(
                lon, xds_TCs_r1.isel(storm = s)[nm_lat].values.tolist(),
                '-', color = get_storm_color(int(cat[s].values)),
                alpha = 0.5,
            )
            ax.plot(
                lon[0], xds_TCs_r1.isel(storm = s)[nm_lat].values.tolist()[0],
                '.', color = get_storm_color(int(cat[s].values)),
                markersize = 10,
            )

    # plot point
    ax.plot(
        pnt_lon, pnt_lat, '.',
        markersize = 15, color = 'brown',
        label = 'STUDY SITE'
    )

    # plot circle
    circle = Circle(
        m(pnt_lon, pnt_lat), r1,
        facecolor = 'grey', edgecolor = 'grey',
        linewidth = 3, alpha = 0.5,
        label='Radius {0}º'.format(r1)
    )
    ax.add_patch(circle)

    # customize axes
    ax.set_aspect(1.0)
    ax.set_ylim(lat1,lat2)
    ax.set_title('Historical TCs', fontsize=15)
    ax.legend(loc=0, fontsize=14)
    axlegend_categ(ax)

    # show and return figure
    if show: plt.show()
    return fig


def axplot_scatter_params(ax, x_hist, y_hist, x_sim, y_sim):
    'axes scatter plot variable1 vs variable2 historical and simulated'

    # simulated params 
    ax.scatter(
        x_sim, y_sim,
        c = 'silver',
        s = 3,
    )

    # historical params 
    ax.scatter(
        x_hist, y_hist,
        c = 'purple',
        s = 5,
    )

def axplot_scatter_params_MDA(ax, x_MDA, y_MDA, x_sim, y_sim):
    'axes scatter plot variable1 vs variable2 historical and simulated'

    # simulated params 
    ax.scatter(
        x_sim, y_sim,
        c = 'silver',
        s = 3,
    )

    # MDA params 
    batch_size = 100
    n = 4
    cs = ['black','red','orange','yellow','blue']

    for i in range(n):
        ix = range(i*100,(i+1)*100)

        ax.scatter(
            x_MDA[ix], y_MDA[ix],
            c = cs[i],
            s = 4,
            label = '{0}-{1}'.format(i*100,(i+1)*100-1)
        )

def axplot_histogram_params(ax, v_hist, v_sim, ttl):
    'axes histogram plot variable historical and simulated'

    # get bins
    j = np.concatenate((v_hist, v_sim))
    bins = np.linspace(np.min(j), np.max(j), 25)

    # historical
    ax.hist(
        v_hist, bins=bins,
        color = 'salmon',
        edgecolor='black',
        linewidth=1.2,
        alpha=0.5,
        label='Historical',
        density=True,
    )

    # simulated 
    ax.hist(
        v_sim, bins=bins,
        color = 'skyblue',
        edgecolor='black',
        linewidth=1.2,
        alpha=0.5,
        label='Simulated',
        density=True,
    )

    ax.set_title(ttl, fontweight='bold')
    ax.legend() #loc='upper right')


def Plot_TCs_Params_HISTvsSIM(TCs_params_hist, TCs_params_sim, show=True):
    '''
    Plot scatter with historical vs simulated parameters
    '''

    # figure conf.
    d_lab = {
        'pressure_min': 'Pmin (mbar)',
        'gamma': 'gamma (º)',
        'delta': 'delta (º)',
        'velocity_mean': 'Vmean (km/h)',
    }

    # variables to plot
    vns = ['pressure_min', 'gamma', 'delta', 'velocity_mean']
    n = len(vns)

    # figure
    fig = plt.figure(figsize=(_faspect*_fsize, _faspect*_fsize))
    gs = gridspec.GridSpec(n-1, n-1, wspace=0.2, hspace=0.2)

    for i in range(n):
        for j in range(i+1, n):

            # get variables to plot
            vn1 = vns[i]
            vn2 = vns[j]

            # historical and simulated
            vvh1 = TCs_params_hist[vn1].values[:]
            vvh2 = TCs_params_hist[vn2].values[:]

            vvs1 = TCs_params_sim[vn1].values[:]
            vvs2 = TCs_params_sim[vn2].values[:]

            # scatter plot 
            ax = plt.subplot(gs[i, j-1])
            axplot_scatter_params(ax, vvh2, vvh1, vvs2, vvs1)

            # custom labels
            if j==i+1:
                ax.set_xlabel(
                    d_lab[vn2],
                    {'fontsize':10, 'fontweight':'bold'}
                )
            if j==i+1:
                ax.set_ylabel(
                    d_lab[vn1],
                    {'fontsize':10, 'fontweight':'bold'}
                )

    # show and return figure
    if show: plt.show()
    return fig

def Plot_TCs_Params_HISTvsSIM_histogram(TCs_params_hist, TCs_params_sim,
                                        show=True):
    '''
    Plot scatter with historical vs simulated parameters
    '''

    # figure conf.
    d_lab = {
        'pressure_min': 'Pmin (mbar)',
        'gamma': 'gamma (º)',
        'delta': 'delta (º)',
        'velocity_mean': 'Vmean (km/h)',
    }

    # variables to plot
    vns = ['pressure_min', 'gamma', 'delta', 'velocity_mean']

    # figure
    fig = plt.figure(figsize=(_faspect*_fsize, _faspect*_fsize))
    gs = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)

    for c, vn in enumerate(vns):

        # historical and simulated
        vvh = TCs_params_hist[vn].values[:]
        vvs = TCs_params_sim[vn].values[:]

        # histograms plot 
        ax = plt.subplot(gs[c])
        axplot_histogram_params(ax, vvh, vvs, d_lab[vn])

    # show and return figure
    if show: plt.show()
    return fig

def Plot_TCs_Params_MDAvsSIM(TCs_params_MDA, TCs_params_sim, show=True):
    '''
    Plot scatter with MDA selection vs simulated parameters
    '''

    # figure conf.
    d_lab = {
        'pressure_min': 'Pmin (mbar)',
        'gamma': 'gamma (º)',
        'delta': 'delta (º)',
        'velocity_mean': 'Vmean (km/h)',
    }

    # variables to plot
    vns = ['pressure_min', 'gamma', 'delta', 'velocity_mean']
    n = len(vns)

    # figure
    fig = plt.figure(figsize=(_faspect*_fsize, _faspect*_fsize))
    gs = gridspec.GridSpec(n-1, n-1, wspace=0.2, hspace=0.2)

    for i in range(n):
        for j in range(i+1, n):

            # get variables to plot
            vn1 = vns[i]
            vn2 = vns[j]

            # historical and simulated
            vvh1 = TCs_params_MDA[vn1].values[:]
            vvh2 = TCs_params_MDA[vn2].values[:]

            vvs1 = TCs_params_sim[vn1].values[:]
            vvs2 = TCs_params_sim[vn2].values[:]

            # scatter plot 
            ax = plt.subplot(gs[i, j-1])
            axplot_scatter_params_MDA(ax, vvh2, vvh1, vvs2, vvs1)

            # custom labels
            if j==i+1:
                ax.set_xlabel(
                    d_lab[vn2],
                    {'fontsize':10, 'fontweight':'bold'}
                )
            if j==i+1:
                ax.set_ylabel(
                    d_lab[vn1],
                    {'fontsize':10, 'fontweight':'bold'}
                )

            if i==0 and j==n-1:
                ax.legend()

    # show and return figure
    if show: plt.show()
    return fig

def Plot_Category_Change(xds_categ_changeprobs, cmap='Blues', show=True):
    '''
    Plot category change betwen r1 and r2
    '''

    cp = xds_categ_changeprobs.category_change_probs.values[:]
    cs = xds_categ_changeprobs.category.values[:]

    # figure
    fig, ax = plt.subplots(1, figsize=(_faspect*_fsize/2, _faspect*_fsize/3))

    pc = ax.pcolor(cp, cmap=cmap)
    fig.colorbar(pc)

    # customize axes
    ttl = 'Category change probabilities'
    ax.set_title(ttl, {'fontsize':10, 'fontweight':'bold'})
    ax.set_xlabel('category')
    ax.set_ylabel('category')

    ticks = np.arange(cp.shape[0]) + 0.5
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ticks_labs = np.arange(cp.shape[0])
    ax.set_xticklabels(ticks_labs)
    ax.set_yticklabels(ticks_labs)

    # show and return figure
    if show: plt.show()
    return fig

