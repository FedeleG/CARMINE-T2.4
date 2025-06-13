import xarray as xr
import numpy as np
from glob import glob
import os
import xclim
from xclim.indices.stats import frequency_analysis
from xclim.core.calendar import percentile_doy
from xclim import units
from xclim.indices.stats import frequency_analysis

def seas_txx(tasmax):
    txx_summer = xclim.indices.tx_max(tasmax, freq='QS-DEC').groupby('time.season').mean(dim = 'time').sel(season = 'JJA')
    return txx_summer
    
def cdds(ds_tas, thresh='21 degC', freq='YS'):

    ds_tas = ds_tas - 273.15
    ds_tas.attrs['units'] = 'degC'
    # filtro il dataset mantenendo solo le temperature sopra 24°C (condizione definizione PNACC)
    mask = ds_tas >= 24
    ds_tas = ds_tas.where(mask, 0)
    # applico funzione xclim con threshold a 21°C. Quindi ccds_raw contiene la differenza fra tas e threshold (21°C) dove tas sono unicamente i valori > 24°C
    cdds_raw = xclim.indices.cooling_degree_days(tas=ds_tas, thresh=thresh, freq=freq)
    
    return cdds_raw 
    
def cdd(pr, thresh='1 mm/day', freq='YS'):
   
    pr = xr.where(pr < 1, 0, pr)

    # combined_ds['TOT_PREC'] = combined_ds['TOT_PREC'].assign_attrs(units='mm/day')
    pr.attrs['units'] = 'mm/day'
 
    indicator = xclim.indices.maximum_consecutive_dry_days(pr, thresh=thresh, freq=freq,
                                                                resample_before_rl=True)
    return indicator

def cwd(pr, thresh='1 mm/day', freq='YS'):
   
    pr = xr.where(pr < 1, 0, pr)

    # combined_ds['TOT_PREC'] = combined_ds['TOT_PREC'].assign_attrs(units='mm/day')
    pr.attrs['units'] = 'mm/day'
 
    indicator = xclim.indices.maximum_consecutive_wet_days(pr, thresh=thresh, freq=freq,
                                                                resample_before_rl=True)
    return indicator

def rr(pr, freq = 'YS'):
   
    rr = xr.where(pr > 1, 1, 0)
    indicator = rr.resample(time = freq).sum(dim = 'time')

    return indicator


def rx5day(pr, thresh = 5, freq = 'YS'):
    pr = xr.where(pr < 1, 0, pr)
    pr.attrs['units'] = 'mm/day'
    
    indicatore_tot = xclim.indices.max_n_day_precipitation_amount(pr, window=thresh, freq=freq)
    return indicatore_tot


def su(tx, thresh = 25, freq = 'YE'):
    tx = tx - 273.15
    tx.attrs['units'] = 'degC'

    indicator = xr.where(tx > thresh, 1, 0)
    indicator = indicator.resample(time = freq).sum(dim='time')
    indicator.attrs['units'] = 'days'
    return indicator


def hot_days(tx, thresh = 35, freq = 'YE'):
    return su(tx, thresh = thresh)

def wsdi(tx, baseline = ('1991','2021'), compute_percentile = 'yes', per_path =''):

    tx.attrs['units'] = 'degK'
    if compute_percentile =='no':
        dspercentile = xr.open_dataset(per_path)
    else:
        dspercentile = percentile_doy(tx.sel(time = slice(baseline[0],baseline[1])), window=5, per=90).sel(percentiles=90)
        print('computed percentile')
        dspercentile.to_dataset(name = 'percentile').to_netcdf(per_path)

    indicator = xclim.indices.warm_spell_duration_index(tx, dspercentile, window=6, freq='YS')
    del dspercentile
    indicator.attrs['units'] = 'days'
    return indicator

