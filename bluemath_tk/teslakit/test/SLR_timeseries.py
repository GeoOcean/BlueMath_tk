
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import pickle
import sys

#----------------------------
Scenarios = ['0.3', '0.5', '1.0', '1.5']#, '2.0', '2.5']
Subscenario = 'MED'
# Subscenario = 'HIGH'
# Subscenario = 'LOW'

#----------------------------
# # Kwajalein
lon = 167.5
lat = 9.75

# Guam
#lon = 144.5
#lat = 13.5

# # San Diego
# lon = -117.17
# lat = 32.71

# MAJURO
#lon = 171.3725
#lat = 7.1060

#----------------------------
# location = 'GMSL'
location = 'TGs'
#location = 'RegionalGrid'


#------------------------------------------------------------------------
# 1) read excel data and select site
# %%
rutin = '/Users/anacrueda/Documents/Proyectos/SERDP/niveles_ref_proyecc/'
file = 'techrpt083.csv'

data = pd.read_csv(os.path.join(rutin, file), skiprows=15)


if location == 'GMSL': # Global SL

    data = data.iloc[:18]

else: # Regional SL

    if location == 'TGs':
        data = data.iloc[18:4302]

    elif location == 'RegionalGrid':
        data = data.iloc[4302:]
# %%
#print(data)

# find closest point:
dif_lon = data['Longitude'] - lon
dif_lat = data['Latitude'] - lat
dif = np.sqrt(dif_lon**2 + dif_lat**2)

min_dist = dif.min(skipna=True)
min_ind = dif.idxmin(skipna=True)


site = data['Site'].loc[min_ind]
data_site = data.loc[(data['Site'] == site)]

# %%
#print(data_site.T)
#print()


#------------------------------------------------------------------------
# 2) Fit RSL to a 2º order polynomial
# based on matlab code that Peter's group used to interpolate the Sweet et al. (2017) SLR decadal projection to hourly data


time_axis = np.array(['2000-01-01', '2010-01-01', '2020-01-01', '2030-01-01', '2040-01-01', '2050-01-01', '2060-01-01',
                    '2070-01-01', '2080-01-01', '2090-01-01', '2100-01-01', '2120-01-01', '2150-01-01', '2200-01-01'], dtype='datetime64[h]')
time_axis_rel = (time_axis - time_axis[0]).astype('int')


time_axis_H = np.arange('2015-01-01', '2101-01-01', dtype='datetime64[h]')
time_axis_H_rel = (time_axis_H - time_axis[0]).astype('int')

y_slr = []
fig, ax = plt.subplots(1,1, figsize=(12, 9))
for scenario in Scenarios:

    data_subset = data_site.loc[data_site['Scenario'] == (scenario + ' - ' + Subscenario)]

    y = data_subset.to_numpy()[0][6:]
    y = y.astype('int')
    y = y/100.0 # to meters

    # Fit data, from 2000 to 2100, to a 2º polynomial function
    coef = np.polyfit(time_axis_rel[:-3], y[:-3], 2)

    # Obtain hourly data from the adjusted polynomial function
    y_fit_H = coef[0]*time_axis_H_rel**2 + coef[1]*time_axis_H_rel + coef[2]

    y_slr.append(y_fit_H)
    # plot
    ax.plot(time_axis, y, '.', label='')
    ax.plot(time_axis_H, y_fit_H, label=scenario)

ax.set_xlim([time_axis[0], time_axis[-3]])

ax.set_ylim([0, 2])
ax.set_title(site + ', SL rise', fontweight='bold')
ax.set_ylabel('m')
ax.grid()
plt.legend(title='GMSL rise scenarios')
# plt.show()
fig.savefig(os.path.join(rutin, 'SLR_' + site + '.png'))

# %%

print(y_slr)
print(np.shape(y_slr))
print()
print(Scenarios)

# %%
xds_slr = xr.Dataset(
    {'slr': (('scenario', 'time'), y_slr),
},
    coords={'scenario': Scenarios, 'time': time_axis_H}
)

print(xds_slr)

# %%
file_Name =r'/Users/anacrueda/Documents/Proyectos/SERDP/niveles_ref_proyecc/slr_kwa.pkl'
# open the file for writing
fileO = open(file_Name,'wb')

# this writes the object a to the
# file named 'testfile'
pickle.dump(xds_slr,fileO)

# here we close the fileObject
fileO.close()

# %%
#file_csv = r'/Users/anacrueda/Documents/Proyectos/SERDP/niveles_ref_proyecc/slr_majuro_05.csv'

#slr_a = xds_slr.to_array()
#slr_c = slr_a.sel(scenario='0.5').values[:].flatten()
#.to_array()

#print(np.shape(slr_c))


#slr_d = xr.Dataset(
#    {'slr': ('time', slr_c),
#},
#    coords={'time': slr_a.time.values[:]}
#)

#print(slr_d)

#slr_x = slr_d.to_array()

# %%
#slr_b = slr_x.to_pandas().T
#print(slr_b)
# %%
#slr_b.to_csv(file_csv)
#print(slr_b)

#xds_slr.to_csv(file_csv)
