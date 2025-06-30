from src.utils.classes import latlon_box
from dotenv import load_dotenv

import xarray as xr
import numpy as np
import os

load_dotenv()

def read_daily_data(variable: str, year: int, month: int, reduced_domain: latlon_box, domain_clip: bool): 
    INPUT_DATA_PATH=os.environ.get("INPUT_DATA_PATH")
    data_emo1 = xr.open_dataset(f'{INPUT_DATA_PATH}/EMO-1arcmin-{variable}_{year}.nc', chunks={"time": 10})[variable]
    if domain_clip:
        data_emo1 = data_emo1.sel(lat=slice(reduced_domain.ur_lat,reduced_domain.ll_lat),lon=slice(reduced_domain.ll_lon,reduced_domain.ur_lon))
    data_emo1 = homogenize_start_day(data_emo1)
    month_selection = (data_emo1.time.dt.month == month)
    this_month_data = data_emo1.isel(time=month_selection)
    this_month_data = homogenize_hour(this_month_data)
    return this_month_data

def read_6h_data(variable: str, year: int, month: int, reduced_domain: latlon_box, domain_clip: bool):
    INPUT_DATA_PATH=os.environ.get("INPUT_DATA_PATH")
    data_emo1 = xr.open_dataset(f'{INPUT_DATA_PATH}/EMO-1arcmin-{variable}6_{year}.nc', chunks={"time": 10})[f'{variable}6']
    if domain_clip:
        data_emo1 = data_emo1.sel(lat=slice(reduced_domain.ur_lat,reduced_domain.ll_lat),lon=slice(reduced_domain.ll_lon,reduced_domain.ur_lon))
    month_and_time_selection_at_12h = ((data_emo1.time.dt.hour == 12) & (data_emo1.time.dt.month == month))
    this_month_data_12h = data_emo1.isel(time=month_and_time_selection_at_12h)
    this_month_data_12h = homogenize_hour(this_month_data_12h)
    return this_month_data_12h

def homogenize_hour(array):
    """
    Set hour to 12:00h
    
    Parameters:
    - array: any array with time dimension
    
    Returns:
    - same array with time dimension values at 12:00h
    """
    array["time"] = array.time.astype('datetime64[D]') + np.timedelta64(12, 'h')
    return array

def homogenize_start_day(array):
    """
    For those variables whose time reference is +1 day, remove that shift
    
    Parameters:
    - array: any array with time dimension
    
    Returns:
    - same array with time reference at +0 day
    """
    if array.time.dt.day[0]==2:
        array["time"] = array.time - np.timedelta64(1, 'D')
    return array
