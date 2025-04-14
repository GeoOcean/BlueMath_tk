#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
from datetime import datetime, timedelta

# tk
from .util.time_operations import get_years_months_days, npdt64todatetime, \
fast_reindex_hourly

# hide numpy warnings
#np.warnings.filterwarnings('ignore')


# TODO: combine GetDistribution functions (parse gow input and call GetDistribution?)

def GetDistribution_gow(xds_wps, swell_sectors, n_partitions=5):
    '''
    Separates wave partitions (0-n_partitions) into families.
    Default: sea, swl1, swl2

    compatible with GOW.mat file

    xds_wps (waves partitionss):
        xarray.Dataset (time,), phs, pspr, pwfrac... {0-5 partitions}

    sectors: list of degrees to cut wave energy [(a1, a2), (a2, a3), (a3, a1)]

    returns
        xarray.Dataset (time,), fam_V, {fam: sea,swell_1,swell2. V: Hs,Tp,Dir}
    '''

    # fix data
    hs_fix_data = 50
    for i in range(6):
        phs = xds_wps['phs{0}'.format(i)].values
        p_fix = np.where(phs >= hs_fix_data)[0]

        # fix data
        xds_wps['phs{0}'.format(i)][p_fix] = np.nan
        xds_wps['ptp{0}'.format(i)][p_fix] = np.nan
        xds_wps['pdir{0}'.format(i)][p_fix] = np.nan

    # sea (partition 0)
    sea_Hs = xds_wps['phs0'].values
    sea_Tp = xds_wps['ptp0'].values
    sea_Dir = xds_wps['pdir0'].values
    time = xds_wps['time'].values

    # concatenate energy groups 
    cat_hs = np.column_stack(
        [xds_wps['phs{0}'.format(i)] for i in range(1, n_partitions+1)])
    cat_tp = np.column_stack(
        [xds_wps['ptp{0}'.format(i)] for i in range(1, n_partitions+1)])
    cat_dir = np.column_stack(
        [xds_wps['pdir{0}'.format(i)] for i in range(1, n_partitions+1)])

    # prepare output array
    xds_fams = xr.Dataset(
        {
            'sea_Hs': ('time', sea_Hs),
            'sea_Tp': ('time', sea_Tp),
            'sea_Dir': ('time', sea_Dir),
        },
        coords = {'time': time}
    )

    # solve sectors
    c = 1
    for s_ini, s_end in swell_sectors:
        if s_ini < s_end:
            p_sw = np.where((cat_dir <= s_end) & (cat_dir > s_ini))
        else:
            p_sw = np.where((cat_dir <= s_end) | (cat_dir > s_ini))

        # get data inside sector
        sect_dir = np.zeros(cat_dir.shape)*np.nan
        sect_hs = np.zeros(cat_dir.shape)*np.nan
        sect_tp = np.zeros(cat_dir.shape)*np.nan

        sect_dir[p_sw] = cat_dir[p_sw]
        sect_hs[p_sw] = cat_hs[p_sw]
        sect_tp[p_sw] = cat_tp[p_sw]

        # calculate swell Hs, Tp, Dir
        swell_Hs = np.sqrt(np.nansum(np.power(sect_hs,2), axis=1))

        swell_Tp = np.sqrt(
            np.nansum(np.power(sect_hs,2), axis=1) /
            np.nansum(np.power(sect_hs,2)/np.power(sect_tp,2), axis=1)
        )
        swell_Dir = np.arctan2(
            np.nansum(np.power(sect_hs,2) * sect_tp * np.sin(sect_dir*np.pi/180), axis=1),
            np.nansum(np.power(sect_hs,2) * sect_tp * np.cos(sect_dir*np.pi/180), axis=1)
        )

        # dir correction and denormalization 
        swell_Dir[np.where((swell_Dir<0))] = swell_Dir[np.where((swell_Dir<0))]+2*np.pi
        swell_Dir = swell_Dir*180/np.pi

        # dont do arctan2 if there is only one dir
        i_onedir = np.where(
            (np.count_nonzero(~np.isnan(sect_dir),axis=1)==1)
        )[0]
        swell_Dir[i_onedir] = np.nanmin(sect_dir[i_onedir], axis=1)

        # out of bound dir correction
        swell_Dir[np.where((swell_Dir>360))] = swell_Dir[np.where((swell_Dir>360))]-360
        swell_Dir[np.where((swell_Dir<0))] = swell_Dir[np.where((swell_Dir<0))]+360


        # fix swell all-nans to 0s nansum behaviour
        p_fix = np.where(swell_Hs==0)
        swell_Hs[p_fix] = np.nan
        swell_Dir[p_fix] = np.nan

        # append data to partitons dataset
        xds_fams['swell_{0}_Hs'.format(c)] = ('time', swell_Hs)
        xds_fams['swell_{0}_Tp'.format(c)] = ('time', swell_Tp)
        xds_fams['swell_{0}_Dir'.format(c)] = ('time', swell_Dir)
        c+=1

    return xds_fams

def GetDistribution_ws(xds_wps, swell_sectors, n_partitions=5):
    '''
    Separates wave partitions (0-5) into families.
    Default: sea, swl1, swl2

    Compatible with wavespectra partitions

    xds_wps (waves partitionss):
        xarray.Dataset (time,), phs, pspr, pwfrac... {0-5 partitions}

    sectors: list of degrees to cut wave energy [(a1, a2), (a2, a3), (a3, a1)]

    returns
        xarray.Dataset (time,), fam_V, {fam: sea,swell_1,swell2. V: Hs,Tp,Dir}
    '''

    # sea (partition 0)
    sea_Hs = xds_wps.isel(part=0).hs.values
    sea_Tp = xds_wps.isel(part=0).tp.values
    sea_Dir = xds_wps.isel(part=0).dpm.values
    time = xds_wps.time.values

    # fix sea all-nans to 0s nansum behaviour
    p_fix = np.where(sea_Hs==0)
    sea_Hs[p_fix] = np.nan
    sea_Tp[p_fix] = np.nan
    sea_Dir[p_fix] = np.nan

    # concatenate energy groups 
    cat_hs = np.column_stack(
        [xds_wps.isel(part=i).hs.values for i in range(1, n_partitions+1)])
    cat_tp = np.column_stack(
        [xds_wps.isel(part=i).tp.values for i in range(1, n_partitions+1)])
    cat_dir = np.column_stack(
        [xds_wps.isel(part=i).dpm.values for i in range(1, n_partitions+1)])

    # prepare output array
    xds_fams = xr.Dataset(
        {
            'sea_Hs': ('time', sea_Hs),
            'sea_Tp': ('time', sea_Tp),
            'sea_Dir': ('time', sea_Dir),
        },
        coords = {'time': time}
    )

    # solve sectors
    c = 1
    for s_ini, s_end in swell_sectors:
        if s_ini < s_end:
            p_sw = np.where((cat_dir <= s_end) & (cat_dir > s_ini))
        else:
            p_sw = np.where((cat_dir <= s_end) | (cat_dir > s_ini))

        # get data inside sector
        sect_dir = np.zeros(cat_dir.shape)*np.nan
        sect_hs = np.zeros(cat_dir.shape)*np.nan
        sect_tp = np.zeros(cat_dir.shape)*np.nan

        sect_dir[p_sw] = cat_dir[p_sw]
        sect_hs[p_sw] = cat_hs[p_sw]
        sect_tp[p_sw] = cat_tp[p_sw]

        # calculate swell Hs, Tp, Dir
        swell_Hs = np.sqrt(np.nansum(np.power(sect_hs,2), axis=1))

        swell_Tp = np.sqrt(
            np.nansum(np.power(sect_hs,2), axis=1) /
            np.nansum(np.power(sect_hs,2)/np.power(sect_tp,2), axis=1)
        )
        swell_Dir = np.arctan2(
            np.nansum(np.power(sect_hs,2) * sect_tp * np.sin(sect_dir*np.pi/180), axis=1),
            np.nansum(np.power(sect_hs,2) * sect_tp * np.cos(sect_dir*np.pi/180), axis=1)
        )

        # dir correction and denormalization 
        swell_Dir[np.where((swell_Dir<0))] = swell_Dir[np.where((swell_Dir<0))]+2*np.pi
        swell_Dir = swell_Dir*180/np.pi

        # dont do arctan2 if there is only one dir
        i_onedir = np.where(
            (np.count_nonzero(~np.isnan(sect_dir),axis=1)==1)
        )[0]
        swell_Dir[i_onedir] = np.nanmin(sect_dir[i_onedir], axis=1)

        # out of bound dir correction
        swell_Dir[np.where((swell_Dir>360))] = swell_Dir[np.where((swell_Dir>360))]-360
        swell_Dir[np.where((swell_Dir<0))] = swell_Dir[np.where((swell_Dir<0))]+360


        # fix swell all-nans to 0s nansum behaviour
        p_fix = np.where(swell_Hs==0)
        swell_Hs[p_fix] = np.nan
        swell_Tp[p_fix] = np.nan
        swell_Dir[p_fix] = np.nan

        # append data to partitons dataset
        xds_fams['swell_{0}_Hs'.format(c)] = ('time', swell_Hs)
        xds_fams['swell_{0}_Tp'.format(c)] = ('time', swell_Tp)
        xds_fams['swell_{0}_Dir'.format(c)] = ('time', swell_Dir)
        c+=1

    return xds_fams

def AWL(hs, tp):
    'Returns Atmospheric Water Level'

    return 0.043*(hs*1.56*(tp/1.25)**2)**(0.5)

def TWL(awl, ss, at, mmsl):
    'Returns Total Water Level'

    return awl + ss + at + mmsl

def Aggregate_WavesFamilies(wvs_fams, a_tp='quadratic'):
    '''
    Aggregate Hs, Tp and Dir from waves families data

    wvs_fams (waves families):
        xarray.Dataset (time,), fam1_Hs, fam1_Tp, fam1_Dir, ...
        {any number of families}

    a_tp = 'quadratic' / 'max_energy', Tp aggregation formulae

    returns Hs, Tp, Dir (numpy.array)
    '''

    # get variable names
    vs = [str(x) for x in wvs_fams.keys()]
    vs_Hs = [x for x in vs if x.endswith('_Hs')]
    vs_Tp = [x for x in vs if x.endswith('_Tp')]
    vs_Dir = [x for x in vs if x.endswith('_Dir')]
    times = wvs_fams.time.values[:]

    # join variable values
    vv_Hs = np.column_stack([wvs_fams[v].values[:] for v in vs_Hs])
    vv_Tp = np.column_stack([wvs_fams[v].values[:] for v in vs_Tp])
    vv_Dir = np.column_stack([wvs_fams[v].values[:] for v in vs_Dir])

    # TODO: entire row nan?
    #p_rn = np.where([x.all() for x in np.isnan(vv_Hs)])[0]
    #vv_Hs = vv_Hs[~p_rn]
    #vv_Tp = vv_Tp[~p_rn]
    #vv_Dir = vv_Dir[~p_rn]
    #times = wvs_fams.time.values[~p_rn]

    # Hs from families
    HS = np.sqrt(np.nansum(np.power(vv_Hs,2), axis=1))

    # nan positions
    ix_nan_data = np.where(HS==0)

    # Tp
    if a_tp == 'quadratic':

        # TP from families 
        tmp1 = np.power(vv_Hs,2)
        tmp2 = np.divide(np.power(vv_Hs,2), np.power(vv_Tp,2))
        TP = np.sqrt(np.nansum(tmp1, axis=1) / np.nansum(tmp2, axis=1))

    elif a_tp == 'max_energy':

        # Hs maximun position 
        vv_Hs_nanzero = vv_Hs.copy()
        vv_Hs_nanzero[np.isnan(vv_Hs)] = 0
        p_max_hs = np.nanargmax(vv_Hs_nanzero, axis=1)

        # Tp from families (Hs max pos)
        TP = np.array([r[i] for r,i in zip(vv_Tp, p_max_hs)])

    else:
        # TODO: make it fail
        pass


    # Dir from families
    tmp3 = np.arctan2(
        np.nansum(np.power(vv_Hs,2) * vv_Tp * np.sin(vv_Dir * np.pi/180), axis=1),
        np.nansum(np.power(vv_Hs,2) * vv_Tp * np.cos(vv_Dir * np.pi/180), axis=1)
    )
    tmp3[tmp3<0] = tmp3[tmp3<0] + 2*np.pi
    DIR = tmp3 * 180/np.pi

    # clear nans
    HS[ix_nan_data] = np.nan
    TP[ix_nan_data] = np.nan
    DIR[ix_nan_data] = np.nan

    # TODO: se usa?
    # Dir from families (Hs max pos)
    #DIR = np.array([r[i] for r,i in zip(vv_Dir, p_max_hs)])

    # return xarray.Dataset
    xds_AGGR = xr.Dataset(
        {
            'Hs': (('time',), HS),
            'Tp': (('time',), TP),
            'Dir': (('time',), DIR),
        },
        coords = {
            'time': times,  # get time from input
        }
    )

    return xds_AGGR

def Intradaily_Hydrograph(xds_wvs, xds_tcs):
    '''
    Calculates intradaily hydrograph (hourly) from a time series of storms.
    storms waves data (hs, tp, dir) and TCs data (mu, tau, ss) is needed.

    xds_wvs (waves aggregated):
        xarray.Dataset (time,), Hs, Tp, Dir

    xds_tcs (TCs):
        xarray.Dataset (time,), mu, tau, ss

    returns xarray.Dataset (time,), Hs, Tp, Dir, SS  (hourly)
    '''

    # input data (storms aggregated waves)
    Hs = xds_wvs.Hs.values[:]
    Tp = xds_wvs.Tp.values[:]
    Dir = xds_wvs.Dir.values[:]
    ts = xds_wvs.time.values[:]

    # fix times
    # TODO: this should not be needed
    if isinstance(ts[0], np.datetime64):
        ts = [npdt64todatetime(x) for x in ts]

    # input data (storms TCs)
    try:
        tau = xds_tcs.tau.values[:]  # storm max. instant (0-1)
        mu = xds_tcs.mu.values[:]
        ss = xds_tcs.ss.values[:]
    except:
        tau = xds_wvs.tau.values[:]  # storm max. instant (0-1)
        mu = xds_wvs.mu.values[:]
        ss = xds_wvs.ss.values[:]

    # storm durations
    s_dur_d = np.array([x.days for x in np.diff(ts)])  # days
    s_dur_h = s_dur_d * 24  # hours
    s_cs_h = np.cumsum(s_dur_h)  # hours since time start
    s_cs_h = np.insert(s_cs_h,0,0)

    # storm tau max (hourly)
    tau_h = np.floor(s_cs_h[:-1] + s_dur_h * tau[:-1])

    # aux function
    def CalcHydro(vv, vt, tt, mt):
        '''
        Calculate variable hourly hydrograph.
        vv - var value at max.
        vt - var time (hours since start, at hydrograph extremes)
        tt - tau max time (hours since start).
        mt - mu value
        '''

        # var value at hydrographs extremes
        vv_extr = vv * np.power(2*mt-1, 2)

        # make it continuous
        vv_extr_cont = (np.roll(vv_extr,1) + vv_extr) / 2
        vv_extr_cont[0] = vv_extr_cont[1]
        vv_extr_cont[-1] = vv_extr_cont[-2]

        # join hydrograph max. and extremes variable data
        vt_full = np.concatenate([vt, tt])  # concatenate times (used for sorting)
        vv_full = np.concatenate([vv_extr_cont, vv])

        # sort data
        ix = np.argsort(vt_full)
        vt_sf = vt_full[ix]
        vv_sf = vv_full[ix]

        # interpolate to fill all hours
        h_times = np.arange(vt_sf[0], vt_sf[-1] + 1, 1)
        h_values = np.interp(h_times, vt_sf, vv_sf)

        # fix times
        h_times = h_times.astype(int)

        return h_values, h_times

    # hydrograph variables: hs and ss
    hourly_Hs, hourly_times = CalcHydro(Hs, s_cs_h, tau_h, mu)
    hourly_ss, _ = CalcHydro(ss, s_cs_h, tau_h, mu)

    # resample waves data to hourly (pad Tp and Dir)
    xds_wvs_h = fast_reindex_hourly(xds_wvs)

    # add Hs and SS 
    xds_wvs_h['Hs'] =(('time',), hourly_Hs)
    xds_wvs_h['SS'] =(('time',), hourly_ss)

    return xds_wvs_h


# --------------------------------------

# TODO check / refactor 
import math
def dispersionLonda(T, h):
    L1 = 1
    L2 = ((9.81*T**2)/(2*np.pi))*math.tanh(h*2*np.pi/L1)
    umbral = 2

    while(umbral>0.1):
        L2 = ((9.81*T**2)/(2*np.pi))*math.tanh(h*2*np.pi/L1)
        umbral = abs(L2-L1)
        L1 = L2

    L = L2
    k = (2*np.pi)/L
    c = np.sqrt(9.8*np.tanh(k*h)/k)
    return L, k, c

def Snell_Propagation(T, H_I, dir_I, Prof_I, Prof_E, OrientBati):
    '''
    [H_E,dir_E]=PropagacionSNELL(T,H_I,dir_I,Prof_I,Prof_E,OrientBati)
    [H_E,dir_E,L_E,L_I]=PropagacionSNELL(T,H_I,dir_I,Prof_I,Prof_E,OrientBati)
    [H_E,dir_E,L_E,L_I,Ks,Kr]=PropagacionSNELL(T,H_I,dir_I,Prof_I,Prof_E,OrientBati)

    Descripci?n: Funci?n que propaga el oleaje por SNELL con batimetr?a recta
    y paralela.

    Entradas:
      T: Periodo.                                        Segundos.
      H_I: Altura de ola en el punto inicial.            Metros.
      dir_I: Direcci?n del oleaje inicial.               Rumbo (0 en el N)
      prof_I: Profundidad inicial.                       Metros.
      prof_E: Profundidad final.                         Metros.
      OrientBati: Orientaci?n de la perpendicular a la batimetria. Rumbo (0 en el N)

    Salidas:
      H_E: Altura de ola en el punto final.              Metros.
      dir_E: Direccion del oleaje final.                 Rumbo (0 en el N)
      L_E: Longitud de onda final.                       Metros
      L_I: Longitud de onda inicial.                     Metros
      Ks: Coeficiente de asomeramiento (shoaling).
      Kr: Coeficiente de refraccion (Snell batimetria recta y paralela).


     Autor:


       Versi?n:Ene/19

     Basada en el script de:
     Soledad Requejo Landeira &   Jos? Antonio ?lvarez Antol?nez
    '''

    # Establece el angulo relativo entre el oleaje y la batimetria
    Teta_I = dir_I - OrientBati

    # Fija el angulo relativo entre -90 y 90 grados
    posd1 = np.where(Teta_I < -90)
    Teta_I[posd1[0]] = Teta_I[posd1[0]] + 360

    posd2 = np.where(Teta_I > 90)
    Teta_I[posd2[0]] = Teta_I[posd2[0]] - 360

    # obligamos que el angulo este en este sector
    Teta_I[np.where(Teta_I > 90)[0]] = 90
    Teta_I[np.where(Teta_I < -90)[0]] = -90

    # Resolucion de la ec. de dispersion en la profundidad de partida y 
    # en la objetivo y calculo de las celeridades de grupo correspondientes
    L_I = []
    k_I = []
    c_I = []
    Cg_I = []

    L_E = []
    k_E = []
    c_E = []
    Cg_E = []

    for i in range(len(T)):
        [a,b,c] = dispersionLonda(T[i],Prof_I)
        L_I.append(a)
        k_I.append(b)
        c_I.append(c)
        Cg_I.append((c/2)*(1+((2*b*Prof_I)/(np.sinh(2*b*Prof_I)))))

        [d,e,f] = dispersionLonda(T[i],Prof_E)
        L_E.append(d)
        k_E.append(e)
        c_E.append(f)
        Cg_E.append((f/2)*(1+((2*e*Prof_E)/(np.sinh(2*e*Prof_E)))))


    dir_E = []
    Teta_E = []
    Ks = []
    Kr = []
    for i in range(len(Cg_E)):
        H = math.asin((k_I[i]*np.sin(np.deg2rad(Teta_I[i])))/k_E[i])
        Teta_E.append(np.rad2deg(H))
        dir_E.append(np.rad2deg(H) + OrientBati)
        Ks.append(np.sqrt(Cg_I[i]/Cg_E[i]))
        Kr.append(np.sqrt(np.cos(np.deg2rad(Teta_I[i]))/np.cos(H)))

    for i in range(len(dir_E)):
        if dir_E[i] < 0:
            dir_E[i] = dir_E[i] + 360
        if dir_E[i] >=360:
            dir_E[i] = dir_E[i] -360

    # Altura de ola final
    H_E = H_I*Kr*Ks

    return H_E, dir_E, Ks, Kr

