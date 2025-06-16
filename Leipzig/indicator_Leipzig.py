import xarray as xr
import numpy as np
from glob import glob
import os
import xclim
from xclim.indices.stats import frequency_analysis
from xclim.core.calendar import percentile_doy
from xclim import units
from xclim.indices.stats import frequency_analysis

def wsdi(tx, baseline = ('1991','2021'), compute_percentile = 'yes', per_path =''):

    tx.attrs['units'] = 'degK'
    if compute_percentile =='no':
        dspercentile = xr.open_dataset(per_path)
    else:
        dspercentile = percentile_doy(tx.sel(time = slice(baseline[0],baseline[1])), window=5, per=90).sel(percentiles=90)
        print('computed percentile')
        dspercentile.to_dataset(name = 'percentile').to_netcdf(per_path)
        
def utci(tas, hurs, sfcWind):
    tas.attrs['units'] = 'degK'
    hurs.attrs['units'] = '%'
    sfcWind.attrs['units'] = 'm/s'

    indicator = xclim.indices.universal_thermal_climate_index(tas, hurs, sfcWind)
    return indicator


def air_t(tas):
    
    return tas.resample(time = '1D').mean(dim = 'time').resample(time = '1Y').mean(dim = 'time').mean(dim = 'time')

def rx5day(pr, thresh = 5, freq = 'YS'):
    pr = xr.where(pr < 1, 0, pr)
    pr.attrs['units'] = 'mm/day'
    
    indicatore_tot = xclim.indices.max_n_day_precipitation_amount(pr, window=thresh, freq=freq)
    return indicatore_tot

def pr95prctile(pr, freq = 'YS'):
    
    pr_m = xr.where(pr < 1, np.nan, pr)
    quant = pr_m.quantile(0.95, dim = 'time')
    quant.attrs['units'] = 'mm'
    return quant.to_dataset(name = 'pr95prctile')