import xarray as xr
import numpy as np
from glob import glob
import os
import xclim
from xclim.indices.stats import frequency_analysis
from xclim.core.calendar import percentile_doy
from xclim import units
from xclim.indices.stats import frequency_analysis


def utci(tas, hurs, sfcWind):
    tas.attrs['units'] = 'degK'
    hurs.attrs['units'] = '%'
    sfcWind.attrs['units'] = 'm/s'

    indicator = xclim.indices.universal_thermal_climate_index(tas, hurs, sfcWind)
    return indicator


def air_t(tas):
    
    return tas.resample(time = '1D').mean(dim = 'time').resample(time = '1Y').mean(dim = 'time').mean(dim = 'time')

def relhum(rh):
    
    return rh.resample(time = '1D').mean(dim = 'time').resample(time = '1Y').mean(dim = 'time').mean(dim = 'time')

def su(tx, thresh = 25, freq = 'YE'):
    tx = tx - 273.15
    tx.attrs['units'] = 'degC'

    indicator = xr.where(tx > thresh, 1, 0)
    indicator = indicator.resample(time = freq).sum(dim='time')
    indicator.attrs['units'] = 'days'
    return indicator


def hot_days(tx, thresh = 35, freq = 'YE'):
    return su(tx, thresh = thresh)