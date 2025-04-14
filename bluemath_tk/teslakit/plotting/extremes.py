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
# from matplotlib.colors import DivergingNorm
from scipy.interpolate import interp1d
from scipy.stats import  gumbel_l, genextreme

# teslakit
from .outputs import axplot_compare_histograms

# import constants
from .config import _faspect, _fsize, _fdpi

# TODO: REFACTOR CHROM Y SIGMA


# Climate Emulator: Fit Report

def Plot_GEVParams(xda_gev_var, c_shape='bwr', c_other='hot_r', show=True):
    'Plot GEV params for a GEV parameter variable (sea_Hs, swell_1_Hs, ...)'

    name = xda_gev_var.name
    params = xda_gev_var.parameter.values[:]
    ss = int(np.sqrt(len(xda_gev_var.n_cluster)))  # this will fail if cant sqrt

    # plot figure
    fig, axs = plt.subplots(2,2, figsize=(_faspect*_fsize, _fsize))
    axs = [i for sl in axs for i in sl]

    # empty last axis
    axs[3].axis('off')

    for c, par in enumerate(params):
        ax = axs[c]

        par_values = xda_gev_var.sel(parameter=par).values[:]
        par_values[par_values==1.0e-10]=0

        rr_pv = np.flipud(np.reshape(par_values,(ss,ss)).T)

        if par == 'shape':
            max_value = max(abs(np.min(par_values)), abs(np.max(par_values)))
            cl = [-max_value, max_value]
            if cl[0]>=0: cl[0]=-0.000000001
            if cl[1]<=0: cl[1]=+0.000000001
            # norm = DivergingNorm(vmin=cl[0], vcenter=0, vmax=cl[1])
            norm = None
            cma = c_shape

        else:
            cl = [np.min(par_values), np.max(par_values)]
            norm = None
            cma = c_other

        cc = ax.pcolor(rr_pv, cmap=cma, vmin=cl[0], vmax=cl[1], norm=norm, edgecolor='k')
        fig.colorbar(cc, ax=ax)

        # add grid and title
        ax.set_title(par)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    ttl = 'GEV - {0}'.format(name)
    fig.suptitle(ttl, fontweight='bold', fontsize=14)

    # show
    if show: plt.show()
    return fig

def Plot_ChromosomesProbs(xds_chrom, show=True):
    'Plot chromosomes probabilities for each WT using chrom triangles'

    # TODO: colores poco fuertes
    # TODO: orden al colorcar los WTs ? mirar otras zonas de codigo
    # mirar tambien el orden de los WTs al generar chrom, si usamos sorted_bmus
    # desde un principio los WTs deberian respetar el orden de todo
    # TODO: lib/plotting/KMA.py Plot_KMArg_clusters_datamean para copiar estilo

    probs = xds_chrom.probs.values[:]
    chrom = xds_chrom.chrom.values[:]
    WTs = xds_chrom.WT
    ss = int(np.sqrt(len(WTs)))  # works for 36

    # triangle
    tsd = 10.0
    tsq = np.sqrt(tsd**2-(tsd/2)**2)
    tri_x = np.array([0, 0, tsq, 0])
    tri_y = np.array([0, tsd, tsd/2, 0])

    # chrom triangle positions (sea - swell_1 - swell_2)
    d_chrom_cntrs = {
        '100': (tsq, tsq/2),
        '010': (0, tsd),
        '001': (0, 0),
        '110': (tsq/2, tsd*3/4),
        '011': (0, tsd/2),
        '101': (tsq/2, tsd/4),
        '111': (tsq/3, tsd/2),
    }

    # max prob and size
    maxprob = np.nanmax(probs[:])
    maxsize = 1.5/maxprob

    # figure
    fig, axs = plt.subplots(ss, ss, figsize=(_faspect*_fsize*1.2, _fsize*1.2))
    axs = [i for sl in axs for i in sl]

    # plot each WT
    for c_wt, wt in enumerate(WTs):
        ax = axs[c_wt]

        # plot triangle
        ax.plot(tri_x, tri_y, 'k:', linewidth=1)

        # plot chromosomes prob circles
        for c_c, ch in enumerate(chrom):
            cntr = d_chrom_cntrs[''.join([str(int(x)) for x in ch])]
            if np.sum(ch) == 3:
                cl = [0,0,0]
            else:
                cl = ch
            ax.add_patch(
                plt.Circle(
                    cntr,
                    radius = maxsize * probs[c_wt, c_c],
                    edgecolor = cl,
                    facecolor = cl,
                    alpha = probs[c_wt, c_c] / maxprob,
                )
            )

        # waves family location text
        ax.text(tsq+0.75, tsq/2+0.5, 'SEA', va='center', ha='center', size=7)
        ax.text(0, tsd+0.5, 'SWELL_1', va='center', ha='center', size=7)
        ax.text(0, 0-0.5, 'SWELL_2', va='center', ha='center', size=7)

        # axis parameters
        ax.axis('equal')
        ax.axis('off')
        #ax.set_title('WT: {0}'.format(c_wt+1))

        ttl = 'Chromosomes Probabilities - WTs'
        fig.suptitle(ttl, fontweight='bold', fontsize=14)

    # show
    if show: plt.show()
    return fig

def Plot_SigmaCorrelation(xds_chrom, d_sigma, show=True):
    'Plot sigma correlation (Hs1-Hs2, Hs1-Tp1) for each WT using chrom triangles'

    # Get sigma correlation values for plot
    n_wts = len(d_sigma.keys())
    n_chs = len(d_sigma[list(d_sigma.keys())[0]].keys())
    ss = int(np.sqrt(n_wts))  # works for 36

    chrom = xds_chrom.chrom.values[:]

    sigma_plot = np.zeros((n_wts, n_chs)) * np.nan
    sigma_wtcrom = np.zeros((n_wts, n_chs)) * np.nan

    for c_wt, k_wt in enumerate(sorted(d_sigma.keys())):
        for c_ch, k_ch in enumerate(sorted(d_sigma[k_wt].keys())):

            corr = d_sigma[k_wt][k_ch]['corr']
            if corr.shape[0]==3:
                sigma_plot[c_wt, c_ch] = corr[0,1]  # one family. Hs - Tp corr
            elif corr.shape[0]==6:
                sigma_plot[c_wt, c_ch] = corr[0,3]  # two families. Hs1 - Hs2 corr

            sigma_wtcrom[c_wt, c_ch] = d_sigma[k_wt][k_ch]['wt_crom']

    # triangle
    tsd = 10.0
    tsq = np.sqrt(tsd**2-(tsd/2)**2)
    tri_x = np.array([0, 0, tsq, 0])
    tri_y = np.array([0, tsd, tsd/2, 0])

    # chrom triangle positions (sea - swell_1 - swell_2)
    d_chrom_cntrs = {
        '100': (tsq, tsq/2),
        '010': (0, tsd),
        '001': (0, 0),
        '110': (tsq/2, tsd*3/4),
        '011': (0, tsd/2),
        '101': (tsq/2, tsd/4),
        '111': (tsq/3, tsd/2),
    }
    # marker colors
    mk_pos = np.array([255, 0, 127]) / 255
    mk_neg = np.array([0, 127, 255]) / 255

    # max corr and size
    maxcorr = np.nanmax(sigma_plot[:])
    maxsize = 1.5/maxcorr

    # figure
    fig, axs = plt.subplots(ss, ss, figsize=(_faspect*_fsize*1.2, _fsize*1.2))
    axs = [i for sl in axs for i in sl]

    # plot each WT
    for c_wt, k_wt in enumerate(sorted(d_sigma.keys())):
        ax = axs[c_wt]

        # plot triangle
        ax.plot(tri_x, tri_y, 'k:', linewidth=1)

        # plot chromosomes prob circles
        for c_c, ch in enumerate(chrom):
            cntr = d_chrom_cntrs[''.join([str(int(x)) for x in ch])]

            # colors by sigma and wtcrom values
            if sigma_plot[c_wt, c_c] >= 0:
                mk_cl = mk_pos
            else:
                mk_cl = mk_neg

            if sigma_wtcrom[c_wt, c_c] == 0:
                e_cl = [0,0,0]
            else:
               e_cl = mk_cl

            # add circle    
            al = np.min([1.0, np.absolute(sigma_plot[c_wt, c_c]) / maxcorr])
            ax.add_patch(
                plt.Circle(
                    cntr,
                    radius = maxsize * np.absolute(sigma_plot[c_wt, c_c]),
                    edgecolor = e_cl,
                    facecolor = mk_cl,
                    alpha = al
                )
            )

        # waves family location text
        ax.text(tsq+0.75, tsq/2+0.5, 'SEA', va='center', ha='center', size=7)
        ax.text(0, tsd+0.5, 'SWELL_1', va='center', ha='center', size=7)
        ax.text(0, 0-0.5, 'SWELL_2', va='center', ha='center', size=7)
        # TODO: textos HH HT etc

        # axis parameters
        ax.axis('equal')
        ax.axis('off')
        #ax.set_title('WT: {0}'.format(c_wt+1))

        ttl = 'Sigma Correlation - WTs'
        fig.suptitle(ttl, fontweight='bold', fontsize=14)

    # show
    if show: plt.show()
    return fig

def Plot_Schemaball(xds_data):
    # TODO
    return None


# Climate Emulator: Simulaton Report

# TODO: refactor, upgrade, add to CE.Report_...

def axplot_compare_annualmax(ax, var_1, trp_1, var_2, trp_2, vn,
                             label_1='Historical', label_2='Simulation',
                             color_1='white', color_2='skyblue'):
    'axes plot histogram comparison between fit-sim variables'

    ax.semilogx(trp_1, var_1, 'ok', color = color_1, label = label_1,
                markersize = 4, zorder = 9)
    ax.semilogx(trp_2, var_2, '.-', color = color_2, label = label_2,
                linewidth = 2, zorder = 8)

    # customize axes
    ax.set_ylabel(vn)
    ax.grid(True)

def Plot_FitSim_AnnualMax(data_fit, data_sim, vns, vn_max=None,
                           color_1='white', color_2='skyblue',
                           label_1='Historical', label_2 = 'Simulation',
                           supt=False, show=True):
    'Plots fit vs sim annual maxima comparison for variables "vns"'

    # aux func for calculating rp time
    def t_rp(time_y):
        ny = len(time_y)
        return np.array([1/(1-(n/(ny+1))) for n in np.arange(1,ny+1)])

    # def. some auxiliar function to select all dataset variables at vn max by groups
    def grouped_max(ds, vn=None, dim=None):
        return ds.isel(**{dim: ds[vn].argmax(dim)})

    # grid spec number of rows
    gs_1 = len(vns)

    # plot figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize*gs_1/2.3))

    # grid spec
    gs = gridspec.GridSpec(gs_1, 1)  #, wspace=0.0, hspace=0.0)

    # handle optional max variable and marginals
    if vn_max:
        amax_fit_marg = data_fit.groupby('time.year').apply(
            grouped_max, vn=vn_max, dim='time')
        amax_sim_marg = data_sim.groupby('time.year').apply(
            grouped_max, vn=vn_max, dim='time')

    # variables
    for c, vn in enumerate(vns):

        if vn_max:
            # get marginal variables at max
            amax_fit = amax_fit_marg[vn]
            amax_sim = amax_sim_marg[vn]

        else:
            # calculate Annual Maxima values 
            amax_fit = data_fit[vn].groupby('time.year').max(dim='time')
            amax_sim = data_sim[vn].groupby('time.year').max(dim='time')

        # get values and time, remove nans
        dh = amax_fit.values[:]; th = amax_fit['year'].values[:]
        th = th[~np.isnan(dh)]; dh = dh[~np.isnan(dh)]

        # remove last simulation year (1 time instant only to calculate max)
        ds = amax_sim.values[:-1]; ts = amax_sim['year'].values[:-1]
        ts = ts[~np.isnan(ds)]; ds = ds[~np.isnan(ds)]

        # prepare values for return period plot
        dh = np.sort(dh); th = t_rp(th)
        ds = np.sort(ds); ts = t_rp(ts)

        # axes
        ax = plt.subplot(gs[c, 0])
        axplot_compare_annualmax(
            ax, dh, th, ds, ts, vn,
            color_1=color_1, color_2=color_2,
            label_1=label_1, label_2=label_2,
        )

        # customize axes
        ax.legend(prop={'size':8})

    # last xaxis
    ax.set_xlabel('Return Period (Years)', fontsize=14)

    # fig suptitle
    if supt:
        fig.suptitle(
            '{0} - {1} Annual Max. Comparison: {2}'.format(label_1, label_2, ', '.join(vns)),
            fontsize=13, fontweight = 'bold',
        )

    # show and return figure
    if show: plt.show()
    return fig

def Plot_FitSim_GevFit(data_fit, data_sim, vn, xds_GEV_Par, kma_fit,
                       n_bins=30,
                       color_1='white', color_2='skyblue',
                       alpha_1=0.7, alpha_2=0.4,
                       label_1='Historical', label_2 = 'Simulation',
                       gs_1 = 1, gs_2 = 1, n_clusters = 1, vlim=1,
                       show=True):
    'Plots fit vs sim histograms and gev fit by clusters for variable "vn"'

    # plot figure
    fig = plt.figure(figsize=(_fsize*gs_2/2, _fsize*gs_1/2.3))

    # grid spec
    gs = gridspec.GridSpec(gs_1, gs_2)  #, wspace=0.0, hspace=0.0)

    # clusters
    for c in range(n_clusters):

        # select wt data
        wt = c+1

        ph_wt = np.where(kma_fit.bmus==wt)[0]
        ps_wt = np.where(data_sim.DWT==wt)[0]

        dh = data_fit[vn].values[:][ph_wt]  #; dh = dh[~np.isnan(dh)]
        ds = data_sim[vn].values[:][ps_wt] #; ds = ds[~np.isnan(ds)]

        # TODO: problem if gumbell?
        # select wt GEV parameters
        pars_GEV = xds_GEV_Par[vn]
        sha = pars_GEV.sel(parameter='shape').sel(n_cluster=wt).values
        sca = pars_GEV.sel(parameter='scale').sel(n_cluster=wt).values
        loc = pars_GEV.sel(parameter='location').sel(n_cluster=wt).values


        # compare histograms
        ax = fig.add_subplot(gs[c])
        axplot_compare_histograms(
            ax, dh, ds, ttl='WT: {0}'.format(wt), density=True, n_bins=n_bins,
            color_1=color_1, color_2=color_2,
            alpha_1=alpha_1, alpha_2=alpha_2,
            label_1=label_1, label_2=label_2,
        )

        # add gev fit 
        x = np.linspace(genextreme.ppf(0.001, -1*sha, loc, sca), vlim, 100)
        ax.plot(x, genextreme.pdf(x, -1*sha, loc, sca), label='GEV fit')

        # customize axis
        ax.legend(prop={'size':8})

    # fig suptitle
    #fig.suptitle('{0}'.format(vn), fontsize=14, fontweight = 'bold')

    # show and return figure
    if show: plt.show()
    return fig


def Plot_Fit_QQ(data_fit, vn, xds_GEV_Par, kma_fit, color='black',
                gs_1 = 1, gs_2 = 1, n_clusters = 1,
                show=True):
    'Plots QQ (empirical-gev) for variable vn and each kma cluster'

    # plot figure
    fig = plt.figure(figsize=(_fsize*gs_2/2, _fsize*gs_1/2.3))

    # grid spec
    gs = gridspec.GridSpec(gs_1, gs_2)  #, wspace=0.0, hspace=0.0)

    # clusters
    for c in range(n_clusters):

        # select wt data
        wt = c+1
        ph_wt = np.where(kma_fit.bmus==wt)[0]
        dh = data_fit[vn].values[:][ph_wt]; dh = dh[~np.isnan(dh)]

        # prepare data
        Q_emp = np.sort(dh)
        bs = np.linspace(1, len(dh), len(dh))
        pp = bs / (len(dh)+1)

        # TODO: problem if gumbell?
        # select wt GEV parameters
        pars_GEV = xds_GEV_Par[vn]
        sha = pars_GEV.sel(parameter='shape').sel(n_cluster=wt).values
        sca = pars_GEV.sel(parameter='scale').sel(n_cluster=wt).values
        loc = pars_GEV.sel(parameter='location').sel(n_cluster=wt).values

        # calc GEV pdf
        Q_gev = genextreme.ppf(pp, -1*sha, loc, sca)

        # scatter plot
        ax = fig.add_subplot(gs[c])
        ax.plot(Q_emp, Q_gev, 'ok', color = color, label='N = {0}'.format(len(dh)))
        ax.plot([0, 1], [0, 1], '--b', transform=ax.transAxes)

        # customize axis
        ax.set_title('WT: {0}'.format(wt))
        ax.axis('equal')
        #ax.set_xlabel('Empirical')
        ax.set_ylabel('GEV')
        ax.legend(prop={'size':8})

    # fig suptitle
    #fig.suptitle('{0}'.format(vn), fontsize=14, fontweight = 'bold')

    # show and return figure
    if show: plt.show()
    return fig


# Extremes Return Period for all simulations

def axplot_RP(ax, t_h, v_h, tg_h, vg_h, t_s, v_s, var_name, sim_percentile=95):
    'axes plot return period historical vs simulation'

    # historical maxima
    ax.semilogx(
        t_h, v_h, 'ok',
        markersize = 3, label = 'Historical',
        zorder=9,
    )

    # TODO: fit historical to gev
    # historical GEV fit
    #ax.semilogx(
    #    tg_h, vg_h, '-b',
    #    label = 'Historical - GEV Fit',
    #)

    # simulation maxima - mean
    mn = np.mean(v_s, axis=0)
    ax.semilogx(
        t_s, mn, '-r',
        linewidth = 2, label = 'Simulation (mean)',
        zorder=8,
    )

    # simulation maxima percentiles
    out = 100 - sim_percentile
    p95 = np.percentile(v_s, 100-out/2.0, axis=0,)
    p05 = np.percentile(v_s, out/2.0, axis=0,)

    ax.semilogx(
        t_s, p95, linestyle='-', color='grey',
        linewidth = 2, #label = 'Simulation (95% percentile)',
    )

    ax.semilogx(
        t_s, p05, linestyle='-', color='grey',
        linewidth = 2, # label = 'Simulation (05% percentile)',
    )
    ax.fill_between(
        t_s, p05, p95, color='lightgray',
        label = 'Simulation ({0}% C.I)'.format(sim_percentile)
    )

    # customize axs
    ax.legend(loc='lower right')
    ax.set_title('Annual Maxima', fontweight='bold')
    ax.set_xlabel('Return Period (years)')
    ax.set_ylabel('{0}'.format(var_name))
    ax.set_xlim(left=10**0, right=np.max(np.concatenate([t_h,t_s])))
    ax.tick_params(axis='both', which='both', top=True, right=True)
    ax.grid(which='both')

def Plot_ReturnPeriodValidation(xds_hist, xds_sim, sim_percentile=95, show=True):
    'Plot Return Period historical - simulation validation'

    # aux func for calculating rp time
    def t_rp(time_y):
        ny = len(time_y)
        return np.array([1/(1-(n/(ny+1))) for n in np.arange(1,ny+1)])

    # aux func for gev fit
    # TODO: fix it
    def gev_fit(var_fit):
        c = -0.1
        vv = np.linspace(0,10,200)

        sha_g, loc_g, sca_g =  genextreme.fit(var_fit, c)
        pg = genextreme.cdf(vv, sha_g, loc_g, sca_g)

        ix = pg > 0.1
        vv = vv[ix]
        ts = 1/(1 - pg[ix])

        # TODO gev params 95% confidence intervals

        return ts, vv

    # clean nans
    t_r = xds_hist.year.values[:]
    v_r = xds_hist.values[:]

    ix_nan = np.isnan(v_r)
    t_r = t_r[~ix_nan]
    v_r = v_r[~ix_nan]

    # RP calculation, var sorting historical
    t_h = t_rp(t_r)
    v_h = np.sort(v_r)

    # GEV fit historical
    #tg_h, vg_h = gev_fit(v_h)
    tg_h, vg_h = [],[]

    # RP calculation, var sorting simulation
    t_s = t_rp(xds_sim.year.values[:-1])  # remove last year*
    v_s = np.sort(xds_sim.values[:,:-1])  # remove last year*

    # figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize))

    axplot_RP(
        axs,
        t_h, v_h, tg_h, vg_h,
        t_s, v_s,
        xds_sim.name,
        sim_percentile=sim_percentile,
    )

    # show and return figure
    if show: plt.show()
    return fig








# TODO: revisar funciones _simulations


def axplot_RP_Sims(ax, tups_time_val, var_name):
    'axes plot return period historical vs simulation'


    lbls_cls = [
        #('Simulation', 'black'),
        ('Simulation - El Niño', 'red'),
        ('Simulation - La Niña' , 'blue'),
    ]

    for (t_s, v_s), (lb, cl) in zip(tups_time_val, lbls_cls):


        # simulation maxima - mean
        mn = np.mean(v_s, axis=0)
        ax.semilogx(
            t_s, mn, linestyle= '-', color=cl,
            linewidth = 2, label = lb,
            zorder=8,
        )

        # simulation maxima percentile 95% and 05%
        p95 = np.percentile(v_s, 95, axis=0,)
        p05 = np.percentile(v_s, 5, axis=0,)

        ax.semilogx(
            t_s, p95, linestyle='--', color=cl, alpha=0.05,
            linewidth = 2, #label = 'Simulation (95% percentile)',
        )

        ax.semilogx(
            t_s, p05, linestyle='--', color=cl, alpha=0.05,
            linewidth = 2, # label = 'Simulation (05% percentile)',
        )
        ax.fill_between(
            t_s, p05, p95, color=cl,alpha=0.05,
            #label = 'Simulation (05 - 95 percentile)'
        )

    # customize axs
    ax.legend(loc='lower right')
    ax.set_title('Annual Maxima', fontweight='bold')
    ax.set_xlabel('Return Period (years)')
    ax.set_ylabel('{0}'.format(var_name))
    ax.set_xlim(left=10**0, right=np.max(t_s))
    ax.tick_params(axis='both', which='both', top=True, right=True)
    ax.grid(which='both')

def Plot_ReturnPeriod_Simulations(l_xds_sim, show=True):
    'Plot Return Period historical - simulation validation'

    # aux func for calculating rp time
    def t_rp(time_y):
        ny = len(time_y)
        return np.array([1/(1-(n/(ny+1))) for n in np.arange(1,ny+1)])

    # RP calculation, var sorting simulation
    tups_time_val = []
    for xds_sim in l_xds_sim:
        t_s = t_rp(xds_sim.year.values[:-1])  # remove last year*
        v_s = np.sort(xds_sim.values[:,:-1])  # remove last year*

        tups_time_val.append((t_s, v_s))

    # figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize))

    axplot_RP_Sims(
        axs,
        tups_time_val,
        xds_sim.name,
    )

    # show and return figure
    if show: plt.show()
    return fig



# TODO: revisar funciones _cambio climatico


def axplot_RP_CC(ax, t_h, v_h, tg_h, vg_h, t_s, v_s, t_s2, v_s2, var_name, sim_percentile=95,label_1='Simulation', label_2 = 'Simulation Climate Change', ):
    'axes plot return period historical vs simulation'

    # historical maxima
    ax.semilogx(
        t_h, v_h, 'ok',
        markersize = 3, label = 'Historical',
        zorder=9,
    )

    # TODO: fit historical to gev
    # historical GEV fit
    #ax.semilogx(
    #    tg_h, vg_h, '-b',
    #    label = 'Historical - GEV Fit',
    #)

    # simulation maxima - mean
    mn = np.mean(v_s, axis=0)
    ax.semilogx(
        t_s, mn, '-r',
        linewidth = 2, label = '{} (mean)'.format(label_1),
        zorder=8,
    )

    # simulation climate change maxima - mean
    mn2 = np.mean(v_s2, axis=0)
    ax.semilogx(
        t_s2, mn2, '-b',
        linewidth = 2, label = '{} (mean)'.format(label_2),
        zorder=8,
    )

    # simulation maxima percentiles
    out = 100 - sim_percentile
    p95 = np.percentile(v_s, 100-out/2.0, axis=0,)
    p05 = np.percentile(v_s, out/2.0, axis=0,)

    ax.semilogx(
        t_s, p95, linestyle='-', color='grey',
        linewidth = 2, #label = 'Simulation (95% percentile)',
    )

    ax.semilogx(
        t_s, p05, linestyle='-', color='grey',
        linewidth = 2, # label = 'Simulation (05% percentile)',
    )
    ax.fill_between(
        t_s, p05, p95, color='lightgray',
        #label = 'Simulation ({0}% C.I)'.format(sim_percentile)
        label = '{} ({} C.I)'.format(label_1, sim_percentile)
    )

    # simulation climate change maxima percentiles
    out = 100 - sim_percentile
    p95 = np.percentile(v_s2, 100-out/2.0, axis=0,)
    p05 = np.percentile(v_s2, out/2.0, axis=0,)

    ax.semilogx(
        t_s2, p95, linestyle='-', color='skyblue',
        linewidth = 2, #label = 'Simulation (95% percentile)',
    )

    ax.semilogx(
        t_s2, p05, linestyle='-', color='skyblue',
        linewidth = 2, # label = 'Simulation (05% percentile)',
    )
    ax.fill_between(
        t_s2, p05, p95, color='skyblue', alpha=0.5,
        #label = 'Simulation Climate Change ({0}% C.I)'.format(sim_percentile)
        label = '{} ({} C.I)'.format(label_2, sim_percentile)
    )

    # customize axs
    ax.legend(loc='lower right')
    ax.set_title('Annual Maxima', fontweight='bold')
    ax.set_xlabel('Return Period (years)')
    ax.set_ylabel('{0}'.format(var_name))
    ax.set_xlim(left=10**0, right=np.max(np.concatenate([t_h,t_s])))
    ax.tick_params(axis='both', which='both', top=True, right=True)
    ax.grid(which='both')


def Plot_ReturnPeriodValidation_CC(xds_hist, xds_sim, xds_sim2, sim_percentile=95, label_1='Simulation', label_2 = 'Simulation Climate Change', show=True):
    'Plot Return Period historical - simulation validation - simulation CLIMATE CHANGE'

    # aux func for calculating rp time
    def t_rp(time_y):
        ny = len(time_y)
        return np.array([1/(1-(n/(ny+1))) for n in np.arange(1,ny+1)])

    # aux func for gev fit
    # TODO: fix it
    def gev_fit(var_fit):
        c = -0.1
        vv = np.linspace(0,10,200)

        sha_g, loc_g, sca_g = genextreme.fit(var_fit, c)
        pg = genextreme.cdf(vv, sha_g, loc_g, sca_g)

        ix = pg > 0.1
        vv = vv[ix]
        ts = 1/(1 - pg[ix])

        # TODO gev params 95% confidence intervals

        return ts, vv

    # clean nans
    t_r = xds_hist.year.values[:]
    v_r = xds_hist.values[:]

    ix_nan = np.isnan(v_r)
    t_r = t_r[~ix_nan]
    v_r = v_r[~ix_nan]

    # RP calculation, var sorting historical
    t_h = t_rp(t_r)
    v_h = np.sort(v_r)

    # GEV fit historical
    #tg_h, vg_h = gev_fit(v_h)
    tg_h, vg_h = [],[]

    # RP calculation, var sorting simulation
    t_s = t_rp(xds_sim.year.values[:-1])  # remove last year*
    v_s = np.sort(xds_sim.values[:,:-1])  # remove last year*

    t_s2 = t_rp(xds_sim2.year.values[:-1])  # remove last year*
    v_s2 = np.sort(xds_sim2.values[:,:-1])  # remove last year*

    # figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize))

    axplot_RP_CC(
        axs,
        t_h, v_h, tg_h, vg_h,
        t_s, v_s,
        t_s2, v_s2,
        xds_sim.name,
        sim_percentile=sim_percentile,
        label_1 = label_1,
        label_2 = label_2,
    )

    # show and return figure
    if show: plt.show()
    return fig
