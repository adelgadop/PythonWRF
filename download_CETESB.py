# Downloading meteorological and air quality parameters from CETESB stations
import numpy as np
import pandas as pd
import qualar_py as qr

cetesb_stations = pd.read_csv(
    '../3_Validation/cetesb_station_2017_codes_qualr.csv', encoding = "ISO-8859-1")
cetesb_stations[1:]

obs_dates = pd.DataFrame({'date': pd.date_range(input('date (YYYY-MM-DD):'), 
            periods=int(input('days:'))*24, freq='H')})
obs_dates['date_qualar']=obs_dates['date'].dt.strftime('%d/%m/%Y')

cetesb_login =  input('login:') 
cetesb_password = input('password:')
start_date = obs_dates.date_qualar.values[0]
end_date = obs_dates.date_qualar.values[-1]

print(obs_dates)

 for i in cetesb_stations.code:
     print('Downloading poll ' + cetesb_stations.name[cetesb_stations.code == i].values + ' Station')
     qr.all_photo(cetesb_login, cetesb_password,
                         start_date, end_date, i, csv_photo=True)
     print('Downloading meteo ' + cetesb_stations.name[cetesb_stations.code == i].values + ' Station')
     qr.all_met(cetesb_login, cetesb_password,
                         start_date, end_date, i, csv_met=True)

