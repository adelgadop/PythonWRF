import wrf
import numpy as np
import pandas as pd
from netCDF4 import Dataset
#import matplotlib.pyplot as plt
#%matplotlib inline
import glob

print("Reading each wrfout...")
wrfout = [Dataset(i) for i in sorted(glob.glob('../wrfout_diss/wrfout_d02*'))]

print("Extracting meteorological variables...")
t2 = wrf.getvar(wrfout, 'T2', timeidx=wrf.ALL_TIMES, method='cat')
rh2 = wrf.getvar(wrfout, 'rh2', timeidx=wrf.ALL_TIMES, method='cat')
wind = wrf.getvar(wrfout, 'uvmet10_wspd_wdir', timeidx=wrf.ALL_TIMES, method='cat')
ws = wind.sel(wspd_wdir='wspd')
wd = wind.sel(wspd_wdir='wdir')
psfc = wrf.getvar(wrfout, 'PSFC', timeidx=wrf.ALL_TIMES, method='cat')

print("Extracting polutants variables...")
o3 = wrf.getvar(wrfout, 'o3', timeidx=wrf.ALL_TIMES, method='cat')
no = wrf.getvar(wrfout, 'no', timeidx=wrf.ALL_TIMES, method='cat')
no2 = wrf.getvar(wrfout, 'no2', timeidx=wrf.ALL_TIMES, method='cat')
co = wrf.getvar(wrfout, 'co', timeidx=wrf.ALL_TIMES, method='cat')

# Retrieving values from surface
o3_sfc  = o3.isel(bottom_top=0)
co_sfc  = co.isel(bottom_top=0)
no_sfc  = no.isel(bottom_top=0)
no2_sfc = no2.isel(bottom_top=0)

print("From ppm to ug/m3...o3, no, no2")
# [ug/m3] = [ppm] * P * M_i / (R * T)
# R = 8.3143 J/K mol
# P in Pa
# T in K
# WRF-Chem gas units in ppmv
R = 8.3144598 # J/K mol
o3_u = o3_sfc * psfc * (16 * 3) / (R * t2)
no_u = no_sfc * psfc * (14 + 16) / (R * t2)
no2_u = no2_sfc * psfc * (14 + 2*16) / (R * t2)

print("Reading file with station location points")
cetesb_stations = pd.read_csv('./cetesb2017_latlon.dat')
print(cetesb_stations)

# Locating stations in west_east (x) and north_south (y) coordinates
stations_xy = wrf.ll_to_xy(wrfout,
                           latitude=cetesb_stations.lat,
                           longitude=cetesb_stations.lon)
cetesb_stations['x'] = stations_xy[0]
cetesb_stations['y'] = stations_xy[1]

# Filter stations inside WRF domain
filter_dom = (cetesb_stations.x > 0) & (cetesb_stations.x < t2.shape[2]) & (cetesb_stations.y > 0) & \
 (cetesb_stations.y < t2.shape[1])
cetesb_dom = cetesb_stations[filter_dom]

# Function to retrieve variables from WRF-Chem
def cetesb_from_wrf(i, to_local=True):
    wrf_est = pd.DataFrame({
    'date': t2.Time.values,
    'tc': t2.sel(south_north=cetesb_dom.y.values[i],
       west_east=cetesb_dom.x.values[i]).values - 273.15,
    'rh': rh2.sel(south_north=cetesb_dom.y.values[i],
       west_east=cetesb_dom.x.values[i]).values,
    'ws': ws.sel(south_north=cetesb_dom.y.values[i],
       west_east=cetesb_dom.x.values[i]).values,
    'wd': wd.sel(south_north=cetesb_dom.y.values[i],
       west_east=cetesb_dom.x.values[i]).values,
    'o3': o3_u.sel(south_north=cetesb_dom.y.values[i],
       west_east=cetesb_dom.x.values[i]).values,
    'no': no_u.sel(south_north=cetesb_dom.y.values[i],
       west_east=cetesb_dom.x.values[i]).values,
    'no2': no2_u.sel(south_north=cetesb_dom.y.values[i],
       west_east=cetesb_dom.x.values[i]).values,
    'co': co_sfc.sel(south_north=cetesb_dom.y.values[i],
       west_east=cetesb_dom.x.values[i]).values,
    'code': cetesb_dom.code.values[i],
    'name': cetesb_dom.name.values[i]})
    if to_local:
        wrf_est['local_date'] = wrf_est['date'].dt.tz_localize('UTC').dt.tz_convert('America/Sao_Paulo')
    return(wrf_est)

print("Extracting data and saving it in a dictionary")
wrf_cetesb = {}

for i in range(0, len(cetesb_dom)):
    wrf_cetesb[cetesb_dom.name.iloc[i]] = cetesb_from_wrf(i)

print('Exporting to csv... ')
name = input('_name.csv: ')
def cetesb_write_wrf(df):
    file_name = str(df.code[0]) + name
    df.to_csv(file_name, index=False)


for k, v in wrf_cetesb.items():
    cetesb_write_wrf(v)

print('''
!!!!!!!!!!!!!!!!!
!! Succesfully !!
!!!!!!!!!!!!!!!!!
''')


