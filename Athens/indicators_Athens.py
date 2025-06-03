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