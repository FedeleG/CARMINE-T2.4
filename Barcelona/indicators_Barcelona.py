import xarray as xr
import numpy as np
from glob import glob
import os
import xclim
from xclim.indices.stats import frequency_analysis
from xclim.core.calendar import percentile_doy
from xclim import units
from xclim.indices.stats import frequency_analysis

def cdd(pr, thresh='1 mm/day', freq='YS'):
   
    pr = xr.where(pr < 1, 0, pr)

    # combined_ds['TOT_PREC'] = combined_ds['TOT_PREC'].assign_attrs(units='mm/day')
    pr.attrs['units'] = 'mm/day'
 
    indicator = xclim.indices.maximum_consecutive_dry_days(pr, thresh=thresh, freq=freq,
                                                                resample_before_rl=True)
    return indicator
    
def drought_spi(tp_daily, baseline = (1981, 2011), scale=3):
        from climate_indices import indices as cindx
        """ Using Standard Precipitation index """

        ## total monthly precipitation
        #tp_monthly = tp_daily.resample(time='MS').sum(dim='time')
        tp_monthly = tp_daily.resample(time='MS').sum(dim = 'time')  # Riga corretta
        
        ## Wrap daily precipitation function
        DISTRIBUTION = cindx.Distribution['gamma']
        DATA_START_YEAR = int(tp_daily.isel(time=0)['time.year'])
        CALIBRATION_YEAR_INITIAL = baseline[0]
        CALIBRATION_YEAR_FINAL = baseline[1] - 1
        PERIODICITY = cindx.compute.Periodicity['monthly']
        compute_spi =   lambda x : cindx.spi(x,
                                            scale,
                                            DISTRIBUTION,
                                            DATA_START_YEAR,
                                            CALIBRATION_YEAR_INITIAL,
                                            CALIBRATION_YEAR_FINAL,
                                            PERIODICITY)

        ## compute SPI/gamma at 3-month scale
        tp_monthly = tp_monthly.chunk({'time' : -1}) # re-chunk along time axis
        time_dim = tp_monthly.sizes['time']

        stand_prec_index = xr.apply_ufunc(compute_spi,
                                tp_monthly,
                                input_core_dims=[['time']],
                                exclude_dims=set(('time',)),
                                output_core_dims=[['time']],
                                output_sizes={'time': time_dim},
                                dask = 'parallelized',
                                output_dtypes  = [float],
                                vectorize = True
                                )

        ## setting up the original time axis
        old_time_ax = tp_monthly['time'].values
        stand_prec_index = stand_prec_index.assign_coords(time=old_time_ax)
        #print(list(stand_prec_index.variables.keys())[0] )
        prec_label = list(stand_prec_index.variables.keys())[0]
        #print(stand_prec_index)
        stand_prec_index = stand_prec_index.rename({ prec_label : 'drought'})
        stand_prec_index['drought'].attrs['long_name'] = 'drought index '
        stand_prec_index['time'].attrs['long_name'] = 'time'

        return  stand_prec_index
    
def spi3(pr, scale = 3):
    baseline = (1989, 2018)
    spi = drought_spi(pr, scale=scale, baseline = baseline)
    return spi

def drought_spei(daily_water_budget, baseline = (1981, 2011), scale=3):
        from climate_indices import indices as cindx
        """ Using Standard Precipitation index """

        ## Wrap daily precipitation function
        DISTRIBUTION = cindx.Distribution['pearson']
        DATA_START_YEAR = int(daily_water_budget.isel(time=0)['time.year'])
        CALIBRATION_YEAR_INITIAL = baseline[0]
        CALIBRATION_YEAR_FINAL = baseline[1] - 1
        PERIODICITY = cindx.compute.Periodicity['monthly']
        compute_spei =   lambda x : cindx.spi(x,
                                            scale,
                                            DISTRIBUTION,
                                            DATA_START_YEAR,
                                            CALIBRATION_YEAR_INITIAL,
                                            CALIBRATION_YEAR_FINAL,
                                            PERIODICITY)
        print(compute_spei)

        ## compute SPEI/pearson at scale-month scale
        tp_monthly = daily_water_budget.chunk({'time' : -1}) # re-chunk along time axis
        time_dim = tp_monthly.sizes['time']

        stand_prec_index = xr.apply_ufunc(compute_spei,
                                tp_monthly,
                                input_core_dims=[['time']],
                                exclude_dims=set(('time',)),
                                output_core_dims=[['time']],
                                output_sizes={'time': time_dim},
                                dask = 'parallelized',
                                output_dtypes  = [float],
                                vectorize = True
                                )

        ## setting up the original time axis
        old_time_ax = tp_monthly['time'].values
        stand_prec_index = stand_prec_index.assign_coords(time=old_time_ax)
        #print(list(stand_prec_index.variables.keys())[0] )
        prec_label = list(stand_prec_index.variables.keys())[0]
        #print(stand_prec_index)
        stand_prec_index = stand_prec_index.rename({ prec_label : 'drought'})
        stand_prec_index['drought'].attrs['long_name'] = 'drought index '
        stand_prec_index['time'].attrs['long_name'] = 'time'

        return  stand_prec_index
def spei3(pr, tas, scale = 3):
    tas.attrs['units'] = 'degK'
    pet = xclim.indices.potential_evapotranspiration(method='TW48', tas = tas)
    pr_mon = pr.resample(time='MS').sum(dim = 'time') 
    dwb = pr - pet
    dwb = dwb.to_dataset(name = 'DWB')
    baseline = (1989, 2018)
    
    spei = drought_spei(dwb, scale=scale, baseline = baseline)
    return spei