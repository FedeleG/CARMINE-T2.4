import eccodes as ec
import numpy as np
import xarray as xr
from dotenv import load_dotenv
import os 

load_dotenv()

def save_to_grib(variable_name, data_values, number):

    data_values = xr.where(np.isnan(data_values),-9999.0,data_values)

    grib_keys = define_default_keys()
    grib_keys = set_variable_keys(variable_name=variable_name,keys=grib_keys)

    grib_keys = set_grid_keys(data_values=data_values,keys=grib_keys)
    grib_keys = set_step_keys(data_values=data_values,keys=grib_keys)
    write_grib_daily_file(variable_name=variable_name,number=number,data_values=data_values.transpose('time','lat','lon'),keys=grib_keys)
    

def write_grib_static_file(variable_name,number,data_values,keys):

    WORKDIR_PATH=os.environ.get("WORKDIR_PATH")

    sample_id = ec.codes_grib_new_from_samples("regular_ll_sfc_grib1")

    output_file = f'{WORKDIR_PATH}/{variable_name}_{number}_EMO1.grib'
    with open(output_file, 'wb') as file_out:
        clone_id = ec.codes_clone(sample_id)

        for key in keys:
            ec.codes_set(clone_id, key, keys[key])

        curr_vals = data_values

        ec.codes_set_values(clone_id, curr_vals.values)

        ec.codes_write(clone_id, file_out)

def write_grib_daily_file(variable_name,number,data_values,keys):

    WORKDIR_PATH=os.environ.get("WORKDIR_PATH")

    sample_id = ec.codes_grib_new_from_samples("regular_ll_sfc_grib1")

    output_file = f'{WORKDIR_PATH}/{variable_name}_{number}_EMO1.grib'
    with open(output_file, 'wb') as file_out:
        for day in range(data_values.time.size):

            clone_id = ec.codes_clone(sample_id)

            for key in keys:
                ec.codes_set(clone_id, key, keys[key])

            keys['startStep'] += 24
            keys['endStep'] += 24
            keys['stepRange'] += 24

            curr_vals = data_values.isel(time=day)

            ec.codes_set_values(clone_id, curr_vals.values)

            ec.codes_write(clone_id, file_out)


def define_default_keys():
    keys = {
        'missingValue': 9999,
        'typeOfLevel': 'surface',
        'NV': 0,
        'stepUnits': 1,
        'stepType': 'instant',
        'gridType': 'regular_ll',
        'bitsPerValue': 16,
    }
    return keys

def set_variable_keys(variable_name,keys):
    paramIds = {
        'temperature': [167,61,'2t'],
        'relative_humidity': [168,168,'2d'],
        'wind_speed': [165,165,'10u'],
        'precipitation': [228,228,'tp'],
        'land_mask': [142,142,'lsm'],
    }

    keys['paramId'] = paramIds[variable_name][0]
    keys['indicatorOfParameter'] = paramIds[variable_name][1]
    keys['shortName'] = paramIds[variable_name][2]

    return keys

def set_grid_keys(data_values,keys):

    nx = int(data_values.lon.size)
    ny = int(data_values.lat.size)
    dx = float(np.abs(data_values.lon[1].values-data_values.lon[0].values))
    dy = float(np.abs(data_values.lat[1].values-data_values.lat[0].values))
    first_lon = float(data_values.lon[0].values)
    last_lon = float(data_values.lon[-1].values)
    first_lat = float(data_values.lat[0].values)
    last_lat = float(data_values.lat[-1].values)+dy

    keys['Nx']=nx
    keys['iDirectionIncrementInDegrees']=dx
    keys['iScansNegatively']=0
    keys['longitudeOfFirstGridPointInDegrees']=first_lon
    keys['longitudeOfLastGridPointInDegrees']=last_lon
    keys['Ny']=ny
    keys['jDirectionIncrementInDegrees']=dy
    keys['jPointsAreConsecutive']=0
    keys['jScansPositively']=0
    keys['latitudeOfFirstGridPointInDegrees']=first_lat
    keys['latitudeOfLastGridPointInDegrees']=last_lat

    return keys

def set_step_keys(data_values,keys):

    keys['startStep'] = 0
    keys['endStep'] = 0
    keys['stepRange'] = 0
    keys['dataDate'] = int(data_values.time[0].dt.strftime('%Y%m%d').values)
    keys['dataTime'] = 1200

    return keys

