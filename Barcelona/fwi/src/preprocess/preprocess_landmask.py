import xarray as xr
import numpy as np

def preprocess_landmask(data):

    land_mask = create_land_mask(data=data)
  
    return land_mask

def create_land_mask(data):
    land_mask = xr.full_like(data,1.)
    land_mask = xr.where(np.isnan(data),0.,land_mask)
    return land_mask