import xarray as xr
import numpy as np
from src.utils.read_emo1 import read_daily_data


def preprocess_relative_humidity(year,month,temperature,reduced_domain,domain_clip):

    vapor_pressure = read_daily_data(variable='pd',year=year, month=month, reduced_domain=reduced_domain, domain_clip=domain_clip)

    temperature_in_C = convert_to_celsius(temperature_in_K=temperature)
      
    relative_humidity = calculate_relative_humidity(vapor_pressure=vapor_pressure, temperature=temperature_in_C)

    return relative_humidity

def calculate_relative_humidity(vapor_pressure, temperature):
    """
    Calculate relative humidity from vapor pressure and temperature using the August-Roche-Magnus formula
    
    Parameters:
    - vapor_pressure: in hPa
    - temperature: in degrees Celsius
    
    Returns:
    - relative humidity as a percentage (0-100)
    """
    
    a = 6.1094  # hPa
    b = 17.625
    c = 243.04  # Celsius
    
    saturation_vapor_pressure = a * np.exp((b * temperature) / (c + temperature))
    
    relative_humidity = (vapor_pressure / saturation_vapor_pressure) * 100

    relative_humidity = xr.where(relative_humidity>100.,100.,relative_humidity)
    relative_humidity = xr.where(relative_humidity<0.,0.,relative_humidity)
  
    return relative_humidity

def convert_to_celsius(temperature_in_K):
    temperature_in_C = temperature_in_K - 273.15
    return temperature_in_C
