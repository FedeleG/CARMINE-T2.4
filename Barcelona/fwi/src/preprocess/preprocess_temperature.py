import xarray as xr
import numpy as np
from src.utils.read_emo1 import read_daily_data


def preprocess_temperature(year,month,reduced_domain,domain_clip):

    tmax = read_daily_data(variable='tx', year=year, month=month, reduced_domain=reduced_domain, domain_clip=domain_clip)
    tmin = read_daily_data(variable='tn', year=year, month=month, reduced_domain=reduced_domain, domain_clip=domain_clip)
      
    temperature = convert_to_temperature_at_1200(tmax=tmax, tmin=tmin)

    temperature = convert_to_kelvin(temperature_in_C=temperature)
    return temperature

def convert_to_kelvin(temperature_in_C):
    temperature_in_K = temperature_in_C + 273.15
    return temperature_in_K

def convert_to_temperature_at_1200(tmax, tmin):
    """
    Calculate temperature at noon from maximum and minimum temperature from Sin (14R-1) method from Chow et al., (2007)

    Chow, D. H. C., & Levermore, G. J. (2007). New algorithm for generating hourly temperature values using daily maximum, minimum and average values from climate models. Building Services Engineering Research and Technology, 28(3), 237-248.
    
    Parameters:
    - tmax: daily maximum temperature
    - tmin: daily minimum temperature
    
    Returns:
    - temperature at local noon
    """

    hour = 12

    hour_tmax = get_hour_tmax()
    hour_tmin = get_hour_tmin(tmin)

    f1 = ( np.cos( np.pi * ( hour - hour_tmin ) / ( hour_tmax - hour_tmin ) ) + 1. ) / 2.
    f2 = 1.-f1

    t12 = tmin*f1+tmax*f2
    return t12

def get_hour_tmax():
    hour_tmax = 14.
    return hour_tmax

def get_hour_tmin(tmin):
    """
    Calculate hour of minimum temperature as the hour before sunrise
    
    Parameters:
    - tmin: daily minimum temperature
    
    Returns:
    - hour of minimum temperature
    """

    dayofyear = get_dayofyear(array=tmin)
    lats = get_lats(array=tmin)
    daylength = daylength_calculation(lats=lats,dayofyear=dayofyear)
    sunrise = 12.-daylength/2.
    hour_tmin = sunrise-1.
    hour_tmin[hour_tmin>9.]=9.
    hour_tmin[hour_tmin<3.]=3.
    return hour_tmin

def get_dayofyear(array):
    dayofyear = array.time.dt.dayofyear.values
    return dayofyear

def get_lats(array):
    lats = array.lat.values
    return lats

def daylength_calculation(lats, dayofyear):
    """
    Calculate day length in hours from latitude and day of year
    
    Parameters:
    - lats: array of latitudes
    - dayofyear: array of day of year
    
    Returns:
    - daylength in hours
    """
    
    dayofyear = dayofyear.reshape(-1, 1, 1)  # shape (D, 1, 1)
    lats = lats.reshape(1, -1, 1)            # shape (1, L, 1)
    
    declination = -23.44 * np.cos(np.radians((360./365.)*(dayofyear + 10)))
    
    tan_lats = np.tan(np.radians(lats))
    tan_decl = np.tan(np.radians(declination))
    
    product = tan_lats * tan_decl
    
    # Initialize daylength array with NaN
    daylength = np.empty(product.shape)
    
    # Cases where product is within valid range
    valid = (product >= -1) & (product <= 1)
    sunrise_hourangle = np.arccos(-product[valid])
    daylength[valid] = 2 * (24./(2*np.pi)) * sunrise_hourangle
    
    # Polar night (24-hour darkness)
    polar_night = (product < -1)
    # Polar day (24-hour daylight)
    polar_day = (product > 1)
    
    # Northern hemisphere cases
    north = (lats > 0)
    daylength[polar_night & north] = 0
    daylength[polar_day & north] = 24
    
    # Southern hemisphere cases
    south = (lats < 0)
    daylength[polar_night & south] = 24
    daylength[polar_day & south] = 0

    
    return daylength