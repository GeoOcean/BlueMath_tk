
#%%
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op

# pip
import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import pickle
from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
# DEV: override installed teslakit
import sys
sys.path.insert(0, op.join(os.path.abspath(''), '..', '..'))

# teslakit
from teslakit.database import Database
from teslakit.rbf import RBF_Interpolation
from teslakit.waves import Intradaily_Hydrograph
from teslakit.io.matlab import ReadMatfile
from teslakit.util.time_operations import date2datenum as d2d


## Database and Site parameters
# %%
# --------------------------------------
# Teslakit database

p_data = r'/Users/anacrueda/Documents/Proyectos/TESLA/Teslakit/data'
db = Database(p_data)

# set site
db.SetSite('KWAJALEIN')

# %%
# --------------------------------------
# load data and set parameters
#  wave data reconstructed a 30m water depth

p_sim = r'/Users/anacrueda/Documents/Proyectos/SERDP/nearshores_waves_albaR/'

p_dataset = op.join(p_sim, 'sim10_propagg21.pkl')
fileObject = open(p_dataset,'rb')
subset = pickle.load(fileObject)

p_datasetO = op.join(p_sim, 'complete_h_offshore.nc')
dataO = xr.open_dataset(p_datasetO)

p_dataO_storm = op.join(p_sim, 'SIM_WAVES_TCs_offshore.nc')
data_storm = xr.open_dataset(p_dataO_storm)

#p_tcs = op.join(p_sim, 'SIM_TCs.nc')
#data_tcs = xr.open_dataset(p_tcs)

# %% load slr data

p_slr = r'/Users/anacrueda/Documents/Proyectos/SERDP/niveles_ref_proyecc/slr_kwa.pkl'
file_slr = open(p_slr,'rb')
data_slr = pickle.load(file_slr)

data_slr = data_slr.sel(scenario='0.5')

print(data_slr)

# %% modify time in simulations to be able to compare them

fechas_nuevas = [datetime(1700, 1, 1, 00) + timedelta(hours=x) for x in range(int(len(dataO.time)))]

dataO['time'] = fechas_nuevas


# %% varying AT, SS and MMSL for each simulation

Level_xds = xr.Dataset({})
for i_sim in dataO.n_sim:
    dataO_storm = dataO.sel(n_sim=i_sim, time=data_storm.time, drop=True)
    data_storm_time_h_rand = [

        d2d(d) + timedelta(hours=np.random.randint(23)) for d in data_storm.time.values[:]

    ]
    # el ultimo valor no puede ser random porque la serie sintetica acaba ese dia a las 00
    # OJO chapuza! nico help!!
    data_storm_time_h_rand[-1] = datetime(2700, 1, 1, 00)
    AT = dataO.sel(n_sim=i_sim, time=data_storm_time_h_rand[:]).AT

    Level_s = dataO_storm.SS + dataO_storm.MMSL + AT

    Level = xr.Dataset({
        'level': ('time', xr.DataArray(Level_s)),
    },
        coords={'time': data_storm.time}
    )
    #print(Level)
    #print(i_sim)
    Level_xds = xr.concat([Level_xds, Level], dim='n_sim')

# %% get same time in simulations and slr scenarios

HS = np.reshape([subset['Hs']], (10,len(data_storm.time)),'C')
TP = np.reshape([subset['Tp']], (10,len(data_storm.time)),'C')
DIR = np.reshape([subset['Dir']], (10,len(data_storm.time)),'C')
#LEVEL = np.reshape([Level_xds['level']], (10,len(data_storm.time)),'C')

wvs_agr = xr.Dataset(
    {'Hs': (('n_sim', 'time'), HS),
     'Tp': (('n_sim', 'time'), TP),
     'Dir': (('n_sim', 'time'), DIR),
     'Level':(('n_sim', 'time'), Level_xds['level'])
},
    coords={'n_sim': dataO.n_sim, 'time': data_storm.time}
)


# %% For each simulation modify the dates to fit the slr projections

wvs_agr_slr = wvs_agr.sel(n_sim=0)
fechas_eventos = wvs_agr_slr.time.sel(time=slice('2600', '2699'))
data_storm_time_slr = [

        d2d(d)- relativedelta(years=601) for d in fechas_eventos.time.values[:]

    ]

# %%
wvs_cent =  xr.Dataset({})
for i_sim in range(0,10):
    if i_sim == 0:
        wvs = wvs_agr_slr.sel(time=slice('1700', '1799'))
        # data_storm_time_slr = [
        #
        #     d2d(d) + relativedelta(years=301) for d in wvs.time.values[:]
        #
        # ]
    elif i_sim == 1:
        wvs = wvs_agr_slr.sel(time=slice('1800', '1899'))
        # data_storm_time_slr = [
        #
        #     d2d(d) + relativedelta(years=201) for d in wvs.time.values[:]
        #
        # ]
    elif i_sim == 2:
        wvs = wvs_agr_slr.sel(time=slice('1900', '1999'))
        # data_storm_time_slr = [
        #
        #     d2d(d) + relativedelta(years=101) for d in wvs.time.values[:]
        #
        # ]
    elif i_sim == 3:
        wvs = wvs_agr_slr.sel(time=slice('2000', '2099'))
        # data_storm_time_slr = [
        #
        #     d2d(d) + relativedelta(years=1) for d in wvs.time.values[:]
        #
        # ]
    elif i_sim == 4:
        wvs = wvs_agr_slr.sel(time=slice('2100', '2199'))
        # data_storm_time_slr = [
        #
        #     d2d(d) - relativedelta(years=101) for d in wvs.time.values[:]
        #
        # ]
    elif i_sim == 5:
        wvs = wvs_agr_slr.sel(time=slice('2200', '2299'))
        # data_storm_time_slr = [
        #
        #     d2d(d) - relativedelta(years=201) for d in wvs.time.values[:]
        #
        # ]
    elif i_sim == 6:
        wvs = wvs_agr_slr.sel(time=slice('2300', '2399'))
        # data_storm_time_slr = [
        #
        #     d2d(d) - relativedelta(years=301) for d in wvs.time.values[:]
        #
        # ]
    elif i_sim == 7:
        wvs = wvs_agr_slr.sel(time=slice('2400', '2499'))
        # data_storm_time_slr = [
        #
        #     d2d(d) - relativedelta(years=401) for d in wvs.time.values[:]
        #
        # ]
    elif i_sim == 8:
        wvs = wvs_agr_slr.sel(time=slice('2500', '2599'))
        # data_storm_time_slr = [
        #
        #     d2d(d) - relativedelta(years=501) for d in wvs.time.values[:]
        #
        # ]
    elif i_sim == 9:
        wvs = wvs_agr_slr.sel(time=slice('2600', '2699'))
        # data_storm_time_slr = [
        #
        #     d2d(d) - relativedelta(years=601) for d in wvs.time.values[:]
        #
        # ]


    wvs_temp = xr.Dataset(
    {'Hs': ('time', wvs['Hs'][0:len(data_storm_time_slr)]),
     'Tp': ('time', wvs['Tp'][0:len(data_storm_time_slr)]),
     'Dir': ('time', wvs['Dir'][0:len(data_storm_time_slr)]),
     'Level': ('time', wvs['Level'][0:len(data_storm_time_slr)]),
     #'time': ('time', data_storm_time_slr)
    },
    coords={'time': data_storm_time_slr}
    )

    wvs_cent = xr.concat([wvs_cent, wvs_temp], dim='n_sim')


# %%

_, index = np.unique(wvs_cent['time'], return_index=True)

print(index)
print(len(index))
wvs_cent = wvs_cent.isel(time=index)


# %%


merged_xds = xr.merge([wvs_cent, data_slr], compat='no_conflicts', join='inner')

print(merged_xds)


# %%
#define reef characteristics
Rslope = 0.0505*np.ones(len(merged_xds.time))
Bslope = 0.1667*np.ones(len(merged_xds.time))
Rw = 250*np.ones(len(merged_xds.time))
Cf = 0.0105*np.ones(len(merged_xds.time))

print(len(merged_xds.time))

# %%
merged_xds['Level_SLR'] = merged_xds.Level + merged_xds.slr

hs_lo2 = merged_xds['Hs']/(1.5613*merged_xds['Tp']**2)

xds_subset = xr.Dataset(
    {'hs':(('n_sim', 'time'), merged_xds['Hs']),
     'tp':(('n_sim', 'time'), merged_xds['Tp']),
     'dir':(('n_sim', 'time'), merged_xds['Dir']),
     'level':(('n_sim', 'time'), merged_xds['Level_SLR']),
     'hs_lo2': hs_lo2,
     'rslope': Rslope,
     'bslope': Bslope,
     'rwidth': Rw,
     'cf': Cf,
},
    coords={'n_sim': merged_xds.n_sim, 'time': merged_xds.time}
)

print(xds_subset)

# %%

# load min, max, of simulated reef and wave conditions and rbfs coefficients

p_rbf = r'/Users/anacrueda/Documents/Proyectos/SERDP/HyCreWWcode/RBF_coefficients/'

p_min = op.join(p_rbf, 'Min_from_simulations.mat')
p_max = op.join(p_rbf, 'Max_from_simulations.mat')

smin = ReadMatfile(p_min)
smax = ReadMatfile(p_max)

# data max and min
dl = np.column_stack([smin['minimum'], smax['maximum']])

print(dl)

d = {
    'hs': dl[1],
    'tp': dl[2],
    'level':dl[0],
    'rslope': dl[3],
    'bslope': dl[4],
    'rwidth': dl[5],
    'cf': dl[6],
    'hs_lo2': [0.005, 0.05],
}

for k in d.keys():
    print(k, d[k])

# %%
# input data quality check
#['level','hs','tp','rslope','bslope','rwidth','cf']:

runup_all_xds = xr.Dataset({})

for i_sim in xds_subset.n_sim:
    xds_f = xds_subset.sel(n_sim=i_sim)

    for vn in xds_f.variables:
        if vn in d.keys():
            # get limits from dictionary
            l, h = tuple(d[vn])

            # find positions inside limits
            xds_f[vn] = xds_f[vn].where((xds_f[vn] >= l) & (xds_f[vn] <= h))

    print('xds_f')
    print(xds_f)
    print()
    # Normalize reef and water level parameters data
    xds_reef = xr.Dataset({
       'level': (xds_f['level']),
       'rslope': (xds_f['rslope']),
       'bslope': (xds_f['bslope']),
       'rwidth': (xds_f['rwidth']),
       'cf': (xds_f['cf']),
        },
         coords={'time': xds_f['time']}
        )

    print('xds_reef')
    print(xds_reef)
    print()
    xds_n = xds_reef.copy()
    for k in xds_reef.variables:
        if k in d.keys():
           v = xds_n[k].values[:]
           l, h = tuple(d[k])
           xds_n[k] =(('time',), (v-l)/(h-l))
    print('xds_n')
    print(xds_n)

    # Hycreeww
    # 15 different wave conditions were considered for each water level and reef characteristics
    # the runup is calculated and then it is linearly interpolated for the actual wave conditions.
    ncases = 15
    runup = []
    # Build RBF target numpy array
    target = np.column_stack(
         (xds_n['level'], xds_n['rslope'], xds_n['bslope'], xds_n['rwidth'], xds_n['cf'])
    )
    target_wavecon = np.column_stack(
      (xds_f['hs'],xds_f['tp'],xds_f['hs_lo2'])
    )

    f_tot = []
    for j in range(0, ncases):
        p_coeff= op.join(p_rbf, 'Coeffs_Runup_Xbeach_test' + str(j+1) + '.mat')
        coeff = ReadMatfile(p_coeff)
        rbf_constant = coeff['coeffsave']['RBFConstant']
        rbf_coeff = coeff['coeffsave']['rbfcoeff']
        nodes = coeff['coeffsave']['x']
        x = target
        f_temp = RBF_Interpolation(rbf_constant, rbf_coeff, nodes,x.T)
        f_tot.append(f_temp)
    print(f_tot)

    #linear interpolation
    hs=[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5] #Wave heigth conditions used in RBF
    hs_lo=[0.005, 0.025, 0.05, 0.005, 0.025, 0.05, 0.005, 0.025, 0.05, 0.005, 0.025, 0.05, 0.005, 0.025, 0.05 ] #Wave conditions used in RBF

    #print(np.shape(f_tot))
    #print(target_wavecon.shape)
    to_z = np.array(f_tot)
    #print(to_z.shape)
    RU = []
    for j in range (0,len(target)):
        x=hs; y=hs_lo; z=to_z[:,j]
        hs_e=target_wavecon[j,0]
        #tp_e=target_wavecon[j,1]
        hs_lo_e=target_wavecon[j,2]
        vq = griddata((x,y),z,(hs_e,hs_lo_e),method='linear')
        RU.append(vq)
    print(RU)
    print()
    #print(i_sim)

    runup_x = np.array(RU)

    runup_xds = xr.Dataset({
        'runup': ('time', xr.DataArray(runup_x)),
    },
        coords={'time': xds_subset.time}
    )
    runup_all_xds = xr.concat([runup_all_xds, runup_xds], dim='n_sim')


# %%

true_runup = runup_all_xds['runup'] + xds_subset['level']

# we need to add the level to the hycreww estimation of Ru to get the proper runup
xds_results = xr.Dataset({
        'hs': (('n_sim','time'), xds_subset['hs']),
        'tp': (('n_sim','time'), xds_subset['tp']),
        'dir': (('n_sim','time'), xds_subset['dir']),
        'level': (('n_sim','time'), xds_subset['level']),
        'hs_lo2': (('n_sim','time'), xds_subset['hs_lo2']),
        'rslope': (xds_subset['rslope']),
        'bslope': (xds_subset['bslope']),
        'rwidth': (xds_subset['rwidth']),
        'cf': (xds_subset['cf']),
        'runup': (('n_sim','time'), true_runup),
        },

         coords={'n_sim': xds_subset.n_sim, 'time': xds_subset.time},

    )

print(xds_results)

# %%



fig, ax = plt.subplots(1,1, figsize=(12, 9))
ax.plot(xds_results.time, xds_results.runup[9,:], '.')
#ax.plot(time_axis_H, y_fit_H, label=scenario)

ax.set_xlim([time_axis[0], time_axis[-3]])

ax.set_ylim([0, 4])
ax.set_title(site + ', runup with SL rise', fontweight='bold')
ax.set_ylabel('m')
ax.grid()
plt.show()
#fig.savefig(os.path.join(rutin, 'SLR_' + site + '.png'))


# %%



fig, ax = plt.subplots(1,1, figsize=(12, 9))
ax.plot(merged_xds.time, merged_xds.slr, '-')
ax.plot(merged_xds.time, merged_xds.Level_SLR[5,:],'.r')
plt.show()

# %%


# ANNUAL MAXIMA

sim_A = xds_results['runup'].groupby('time.year').max(dim='time')

def axplot_AM_SLR(ax, t_s, v_s, var_name):
    'axes plot runup annual maxima for a SLR scenario'

    # simulation maxima - mean
    mn = np.mean(v_s, axis=0)
    ax.plot(
        t_s, mn, '-r',
        linewidth = 2, label = 'Simulation (mean)',
        zorder=8,
    )

    # simulation maxima percentile 95% and 05%
    p95 = np.percentile(v_s, 95, axis=0,)
    p05 = np.percentile(v_s, 5, axis=0,)

    ax.plot(
        t_s, p95, linestyle='-', color='grey',
        linewidth = 2, #label = 'Simulation (95% percentile)',
    )

    ax.plot(
        t_s, p05, linestyle='-', color='grey',
        linewidth = 2, # label = 'Simulation (05% percentile)',
    )
    ax.fill_between(
        t_s, p05, p95, color='lightgray',
        label = 'Simulation (05 - 95 percentile)'
    )

    # customize axs
    ax.legend(loc='lower right')
    ax.set_title('Annual Maxima', fontweight='bold')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel(' runup (m)')
    #ax.set_xlim(left=t_s[0], right=t_s[-1])))
    ax.tick_params(axis='both', which='both', top=True, right=True)
    ax.grid(which='both')


t_s = sim_A['year']
v_s = sim_A


# figure

# common fig parameters
_faspect = 1.618
_fsize = 9.8
_fdpi = 128
name = 'runup'

fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize))

axplot_AM_SLR(
        axs,
        t_s, v_s,
        name,
)
plt.show()