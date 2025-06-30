import xarray as xr
import numpy as np
from src.utils.read_emo1 import read_daily_data

def preprocess_wind(year,month,reduced_domain,domain_clip,wind_limit):

    wind_speed = read_daily_data(variable='ws', year=year, month=month, reduced_domain=reduced_domain, domain_clip=domain_clip)

    if wind_limit:
        wind_speed = upper_wind_limit(wind_speed=wind_speed, wind_limit=wind_limit)
    
    return wind_speed

def upper_wind_limit(wind_speed,wind_limit):
    wind_speed = xr.where(wind_speed>wind_limit,wind_limit,wind_speed)
    return wind_speed