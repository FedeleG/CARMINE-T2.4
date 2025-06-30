import xarray as xr
import numpy as np
from src.utils.read_emo1 import read_daily_data, read_6h_data

def preprocess_precipitation(year,month,which_precipitation,reduced_domain,domain_clip):

    if which_precipitation=='6h':
        precipitation = read_6h_data(variable='pr', year=year, month=month, reduced_domain=reduced_domain, domain_clip=domain_clip)
    elif which_precipitation=='daily':
        precipitation = read_daily_data(variable='pr', year=year, month=month, reduced_domain=reduced_domain, domain_clip=domain_clip)
    else:
        raise TypeError("Only daily or 6h precipitation types are allowed")    
    
    return precipitation
