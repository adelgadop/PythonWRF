import wrf
import numpy as np
import pandas as pd
from netCDF4 import Dataset
#import matplotlib.pyplot as plt
#%matplotlib inline
import glob
import pickle as pkl

print("Reading each wrfout...")
month = input('month (e.g., 09): ')
year = input('year: ')
scenario = input('scenario: ')
wrfout = [Dataset(i) for i in sorted(glob.glob('../wrfout_diss/wrfout_d02_'+year+'-'+month+'-*'))]

print("Extracting RAINC and RAINNC variables, named as rainc and rainnc")
rainc = wrf.getvar(wrfout,'RAINC',timeidx=wrf.ALL_TIMES, method='cat')
rainnc = wrf.getvar(wrfout,'RAINNC',timeidx=wrf.ALL_TIMES, method='cat')

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
filter_dom = (cetesb_stations.x > 0) & (cetesb_stations.x < rainc.shape[2]) & (cetesb_stations.y > 0) & \
 (cetesb_stations.y < rainc.shape[1])
cetesb_dom = cetesb_stations[filter_dom]

# Function to retrieve variables from WRF-Chem
def cetesb_from_wrf(i, to_local=True):
    wrf_est = pd.DataFrame({
        'date': rainc.Time.values,
        'rainc': rainc.sel(south_north=cetesb_dom.y.values[i],
                           west_east=cetesb_dom.x.values[i]).values,
        'rainnc': rainnc.sel(south_north=cetesb_dom.y.values[i],
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

print('Exporting to *.pickle')
pkl.dump(wrf_cetesb, open('rain_'+year+'-'+month+'_'+scenario+'.pickle','wb'))
print('''
!!!!!!!!!!!!!!!!!
!! Succesfully !!
!!!!!!!!!!!!!!!!!
''')



