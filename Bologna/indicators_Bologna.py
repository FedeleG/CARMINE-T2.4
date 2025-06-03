import xarray as xr
import numpy as np
from glob import glob
import os
import xclim
from xclim.indices.stats import frequency_analysis
from xclim.core.calendar import percentile_doy
from xclim import units
from xclim.indices.stats import frequency_analysis

def wsdi(tasmax, window = 6, percentiles = 90):
    
    tasmax.attrs['units'] = 'K'
    dspercentile = percentile_doy(tasmax, window=5, per=percentiles).sel(percentiles=percentiles)
    
    indicatore_tot = xclim.indices.warm_spell_duration_index(tasmax, dspercentile, window=window, freq='YS')
    return indicatore_tot


def tr(tasmin):
    # setto a 0 il rumore del modello
    tr = xr.where(tasmin > 293.15, 1, 0)

    indicatore_tot = tr.resample(time = '1Y').sum(dim='time')
    return indicatore_tot

def cdds(ds_tas, thresh='21 degC', freq='YS'):

    ds_tas = ds_tas - 273.15
    ds_tas.attrs['units'] = 'degC'
    # filtro il dataset mantenendo solo le temperature sopra 24°C (condizione definizione PNACC)
    mask = ds_tas >= 24
    ds_tas = ds_tas.where(mask, 0)
    # applico funzione xclim con threshold a 21°C. Quindi ccds_raw contiene la differenza fra tas e threshold (21°C) dove tas sono unicamente i valori > 24°C
    cdds_raw = xclim.indices.cooling_degree_days(tas=ds_tas, thresh=thresh, freq=freq)
    
    return cdds_raw 