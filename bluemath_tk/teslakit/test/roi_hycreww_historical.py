
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

# DEV: override installed teslakit
import sys
sys.path.insert(0, op.join(os.path.abspath(''), '..', '..'))

# teslakit
from teslakit.database import Database
from teslakit.io.matlab import ReadMatfile
from teslakit.rbf import RBF_Interpolation


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
#  wave data reconstructed a 30m water depth

p_hist = r'/Users/anacrueda/Documents/Proyectos/SERDP/nearshores_waves_albaR/'

p_dataset = op.join(p_hist, 'dataset_propagg21.pkl')
fileObject = open(p_dataset,'rb')
subset = pickle.load(fileObject) # oleaje reconstruido

p_histO = r'/Users/anacrueda/Documents/Proyectos/SERDP/results_files/Historicos/'
p_dataset = op.join(p_histO, 'Reconstr_Hs_Tp_Dir_Level_NOtransectRoi_Historical_2016_p6_in30m_snell_sep.mat') # datos de nivel
p_gow = op.join(p_histO, 'KWA_historical_parameters_2016_sep.mat') # tiempo

subsetO = ReadMatfile(p_dataset)
timeo = ReadMatfile(p_gow)['time']
# parse matlab datenum to datetime
time = DateConverter_Mat2Py(timeo)

# parse matlab datenum to datetime
#time = DateConverter_Mat2Py(timeo)

#define reef characteristics
Rslope = 0.0505*np.ones(len(time))
Bslope = 0.1667*np.ones(len(time))
Rw = 250*np.ones(len(time))
Cf = 0.0105*np.ones(len(time))

# %%

print(subsetO['Level'])

plt.figure()
plt.plot(subsetO['Level'])
plt.show()


#print(subsetO.time.min(),subsetO.time.max())
#print(time.min(),time.max())

# %%
# dataset with wave and reef characteristics

hs_lo2 = subset['Hs']/(1.5613*subset['Tp']**2)

xds_subset = xr.Dataset(
    {
        'hs': ('time', subset['Hs'][0:111033]),
        'tp': ('time', subset['Tp'][0:111033]),
        'dir': ('time', subset['Dir'][0:111033]),
        'level': ('time', subsetO['Level']),
        'hs_lo2': ('time', hs_lo2[0:111033]),
        'rslope': ('time', Rslope),
        'bslope': ('time', Bslope),
        'rwidth': ('time', Rw),
        'cf': ('time', Cf),


    },

    coords={'time': time}
)

#print(xds_subset)

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

# NICO
xds_f = xds_subset.copy()

for vn in xds_subset.variables:
    if vn in d.keys():

        # get limits from dictionary
        l, h = tuple(d[vn])

        # find positions inside limits
        xds_f[vn] = xds_subset[vn].where((xds_subset[vn] >= l) & (xds_subset[vn] <= h))

print(xds_f)

#%%

# Normalize reef and water level parameters data

xds_reef = xr.Dataset({

    'level': ('time', xds_f['level']),
    'rslope': ('time', xds_f['rslope']),
    'bslope': ('time', xds_f['bslope']),
    'rwidth': ('time', xds_f['rwidth']),
    'cf': ('time', xds_f['cf']),
},
    coords={'time': xds_f['time']}
)

print(xds_reef)

xds_n = xds_reef.copy()
for k in xds_reef.variables:
    if k in d.keys():
        v = xds_n[k].values[:]
        l, h = tuple(d[k])
        xds_n[k] =(('time',), (v-l)/(h-l))


print()
print(xds_n)

#%%

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

#%%

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

#%%

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

#%%
print(RU)

# we need to add the level to the hycreww estimation of Ru to get the proper runup
xds_results = xr.Dataset(
    {
        'hs': ('time', xds_f['hs']),
        'tp': ('time', xds_f['tp']),
        'dir': ('time', xds_f['dir']),
        'level': ('time', xds_f['level']),
        'hs_lo2': ('time', xds_f['hs_lo2']),
        'rslope': ('time', xds_f['rslope']),
        'bslope': ('time', xds_f['bslope']),
        'rwidth': ('time', xds_f['rwidth']),
        'cf': ('time', xds_f['cf']),
        'runup': ('time', RU + xds_f['level']),


    },

    coords={'time': time}
)

print(xds_results)

plt.figure(figsize=(13, 3.5))
ax = plt.gca()
xds_results.runup.plot(ax=ax)
plt.show()


#%%
file_Name =r'/Users/anacrueda/Documents/Proyectos/SERDP/nearshores_waves_albaR/runup_hist_hycreww_py.pkl'
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