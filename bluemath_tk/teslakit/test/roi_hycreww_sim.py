
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
import numpy.matlib as nm
from cftime._cftime import DatetimeGregorian


# DEV: override installed teslakit
import sys
sys.path.insert(0, op.join(os.path.abspath(''), '..', '..'))

# teslakit
from teslakit.database import Database
from teslakit.io.matlab import ReadMatfile
from teslakit.rbf import RBF_Interpolation
from teslakit.util.time_operations import date2datenum as d2d
from teslakit.plotting.extremes import Plot_ReturnPeriodValidation
from teslakit.util.time_operations import DateConverter_Mat2Py


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
#  wave data reconstructed at 30m water depth

p_sim = r'/Users/anacrueda/Documents/Proyectos/SERDP/nearshores_waves_albaR/'

p_dataset = op.join(p_sim, 'sim10_propagg21.pkl')
fileObject = open(p_dataset,'rb')
subset = pickle.load(fileObject)

p_datasetO = op.join(p_sim, 'complete_h_offshore.nc')
dataO = xr.open_dataset(p_datasetO)

p_dataO_storm = op.join(p_sim, 'SIM_WAVES_TCs_offshore.nc')
data_storm = xr.open_dataset(p_dataO_storm)


# %%
#define reef characteristics
Rslope = 0.0505*np.ones(len(data_storm.time))
Bslope = 0.1667*np.ones(len(data_storm.time))
Rw = 250*np.ones(len(data_storm.time))
Cf = 0.0105*np.ones(len(data_storm.time))


# %% time vector for the hourly data

fechas_nuevas = [datetime(1700, 1, 1, 00) + timedelta(hours=x) for x in range(int(len(dataO.time)))]
#dataO_mod = dataO.loc[~dataO.time.duplicated(keep='first')]

#print(len(fechas_nuevas))

dataO['time'] = fechas_nuevas
print(dataO.time)

#print(dataO.time[-1].values)
#ttt = dataO.time.values[:]
#print(len(data_storm_time_h_rand))
#print(data_storm_time_h_rand[-1] in ttt)


# %%
# NICO: DE ESTA SELECCION COGEMOS MMSL Y SS
#dataO_storm = dataO.sel(n_sim=0, time=data_storm.time, drop=True)

# NICO: NOS CREAMOS UN VECTOR DE TIEMPO IGUAL QUE DATA_STORM.TIME PERO CON HORA
# ALEATORIA

#data_storm_time_h_rand = [

 #   d2d(d) + timedelta(hours=np.random.randint(23)) for d in data_storm.time.values[:]

#]
#print(data_storm_time_h_rand)

# NICO: COGEMOS AT DE AQUI
#el ultimo valor no puede ser random porque la serie sintetica acaba ese dia a las 00
# OJO chapuza! nico help!!
#data_storm_time_h_rand[-1] = datetime(2700,1,1,00)

#AT = dataO.sel(n_sim=0,time=data_storm_time_h_rand[:]).AT
#dataO_storm = dataO.sel(n_sim=0, time=data_storm.time)

#Level = dataO_storm.SS + dataO_storm.MMSL + AT

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

# %%
# dataset with wave and reef characteristics
print(np.shape(subset['Hs']))

Hs = np.reshape([subset['Hs']], (10,len(data_storm.time)),'C')
Tp = np.reshape([subset['Tp']], (10,len(data_storm.time)),'C')
Dir = np.reshape([subset['Dir']], (10,len(data_storm.time)),'C')

# Level_r = nm.repmat(Level, 10, 1)
#Level_r = np.reshape([Level_xds['level']], (10,len(data_storm.time)),'C')


sim_wvs = xr.Dataset(
    {'Hs': (('n_sim', 'time'), Hs),
     'Tp': (('n_sim', 'time'), Tp),
     'Dir': (('n_sim', 'time'), Dir),
     'Level': (('n_sim', 'time'), Level_xds['level'])
},
    coords={'n_sim': dataO.n_sim, 'time': data_storm.time}
)

print(sim_wvs)


# %%

hs_lo2 = sim_wvs['Hs']/(1.5613*sim_wvs['Tp']**2)

print(hs_lo2)

xds_subset = xr.Dataset(
    {
        'hs': (('n_sim','time'), sim_wvs['Hs']), #subset['Hs']
        'tp': (('n_sim','time'),sim_wvs['Tp']),
        'dir': (('n_sim','time'),sim_wvs['Dir']),
        'level': (('n_sim','time'),sim_wvs['Level']),
        'hs_lo2': (('n_sim','time'),hs_lo2),
        'rslope': Rslope,
        'bslope': Bslope,
        'rwidth': Rw,
        'cf': Cf,
    },

    coords={'n_sim': dataO.n_sim, 'time': data_storm.time},
    # dims = {'n_sim':10,'time':len(time)}

)

print(xds_subset)

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

for i_sim in dataO.n_sim:
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
        coords={'time': data_storm.time}
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

         coords={'n_sim': dataO.n_sim, 'time': data_storm.time},

    )

print(xds_results)

#%%
file_Name = r'/Users/anacrueda/Documents/Proyectos/SERDP/nearshores_waves_albaR/runup_all_sim_test_hycreww_py.pkl'
# open the file for writing
fileO = open(file_Name,'wb')

# this writes the object a to the
# file named 'testfile'
pickle.dump(xds_results,fileO)

# here we close the fileObject
fileO.close()


#print(xds_results)

#runup_nan = xds_results.where(np.isnan(xds_results.runup) == 'True', drop='True')

#print(runup_nan)

# %% figure annual maxima

# load historical
file_hist =r'/Users/anacrueda/Documents/Proyectos/SERDP/nearshores_waves_albaR/runup_hist_hycreww_py.pkl'

df = open(file_hist, 'rb')
data_hist = pickle.load(df)

print(data_hist)

# load simulated
file_sim= r'/Users/anacrueda/Documents/Proyectos/SERDP/nearshores_waves_albaR/runup_all_sim_test_hycreww_py.pkl'

df_sim = open(file_sim, 'rb')
data_sim= pickle.load(df_sim)

hist_A = data_hist['runup'].groupby('time.year').max(dim='time')
sim_A = data_sim['runup'].groupby('time.year').max(dim='time')

# Return Period historical vs. simulations
Plot_ReturnPeriodValidation(hist_A, sim_A)
