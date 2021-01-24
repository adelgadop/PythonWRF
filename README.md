# PythonWRF
Process outputs from WRF-Chem.

## WRF-Chem model results processing
WRF-Chem model outputs are files format (`wrfout_<domain>_{year}-{m}-{d}_{h}:00:0`) that content matrix values for many parameters meteorological and chemical species.

There are scripts and Jupyter Notebooks used to process WRF-Chem outputs:

* `wrf_extract.py`: Script developed by M. Gavidia [quishqa](https://github.com/quishqa). This script extract meteorological and polutants variables such as temperature (t2), relative humidity (rh2), wind, atmospheric pressure (psfc), ozone (o3), nitrogen monoxide (no), nitrogen dioxide (no2), and carbon monoxide (co). This script requires location points as latitude and longitude. In this case, we used CETESB locations provided in its website (QUALAR). Pollutants (with the exception of carbon monoxide) were converted from ppm to $\mu$ gm$^{-3}$. After that, variables extracted were saving as dictionary and exporting to csv format.
* `wrf_rain.py`: Script based on `wrf_extract.py` that extracts `rain` and `rainnc`. These variables are accumulated rain and require to be processed. Finally, the script export values based on location as dictionary saving as pickle (very useful in Python).
* `mod_stats.py`: Script developed by M. Gavidia [quishqa](https://github.com/quishqa) and modified by me. This script analyze model outputs and observed values and calculate values for each statistical benchmarks.
* Jupyter Notebooks: `3.1 Evaluation of simulation Results for September 2018.ipynb` for instance contents Python scripts used to obtain charts and statistical results. `3.2 Meteorological results evaluation` also is similar to before one, but it process meteorological outputs. `Maps Locations.ipynb` shows scripts to make maps locations.

## ACKNOWLEDGMENTS
The author thank the CAPES (Coordenação de Aperfeiçonamento de Pessoal de Nível Superior) for the financial support of this work. I also thank the State Company for the Environment (CETESB) for the hourly data from the station networks available as open acces in its website (QUALAR).
