import sys
sys.path.insert('/Athens/')
import indicators_Athens as iA 
import xarray as xr
'''
This example is to compute an indicator for Athens CSA with EOBS dataset. The name of the variable could be different using other dataset
'''
path_data = ''
outpath = ''

ds = xr.open_dataset(f'{path_data}')

CWD = iA.cwd(ds.rr)

'''
Last thing to do is to mean over the reference period choice and save in netcdf
'''
CWD.mean(dim='time').to_dataset(name = 'CWD').to_netcdf(f'{outpath}/eobs_cwd.nc')