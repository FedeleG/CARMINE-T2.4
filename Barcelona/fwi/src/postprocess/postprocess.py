
import calendar
import xarray as xr
import argparse
import os 
import shutil
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

import src.utils.read_emo1 as read_emo1

load_dotenv()

def main():
    args = parse_input_arguments()

    year: int = args.year
    month: int = args.month
    domain_clip: bool = args.domain_clip    
    ll_lat: float = args.ll_lat
    ll_lon: float = args.ll_lon
    ur_lat: float = args.ur_lat
    ur_lon: float = args.ur_lon
    clear_inputs: bool = args.clear_inputs
    
    reduced_domain = read_emo1.latlon_box(
        ur_lat=ur_lat,
        ur_lon=ur_lon,
        ll_lat=ll_lat,
        ll_lon=ll_lon,
    )

    WORKDIR_PATH=os.environ.get("WORKDIR_PATH")
    OUTPUTS_PATH=os.environ.get("OUTPUTS_PATH")
    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    
    emo1_variable = read_emo1.read_daily_data(variable='tx', year=year, month=month, reduced_domain=reduced_domain, domain_clip=domain_clip)
    real_lats = emo1_variable['lat']
    real_lons = emo1_variable['lon']

    for this_var in ['FWI']:#,'DC','DMC','FFMC','ISI','BUI']:
        print(this_var)
        variable_data = read_output_vars(year=year, month=month, var=this_var)
        variable_data = correct_coords(variable_data=variable_data, real_lats=real_lats, real_lons=real_lons)
        save_output(year=year, month=month, var=this_var, variable_data=variable_data)

    if clear_inputs:
        for filename in os.listdir(WORKDIR_PATH):
            file_path = os.path.join(WORKDIR_PATH, filename)
            try:
                if filename=="output_restart.grib":
                    print(f"Saved restart file in {file_path}.")
                    shutil.copyfile(file_path, os.path.join(OUTPUTS_PATH, f"output_restart_{year}-{month:02d}.grib"))
                elif os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

def parse_input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year for FWI calculation",
    )
    parser.add_argument(
        "--month",
        type=int,
        required=True,
        help="Month for FWI calculation",
    )
    parser.add_argument(
        "--domain_clip",
        type=bool,
        default=False,
        help="Reduce domain size (True) or not (False)"
    )
    parser.add_argument(
        "--ll_lat",
        type=float,
        default=-90.,
        help="Lower left latitude of reduced domain (if domain_clip=True)",
    )
    parser.add_argument(
        "--ll_lon",
        type=float,
        default=-180.,
        help="Lower left longitude of reduced domain (if domain_clip=True)",
    )
    parser.add_argument(
        "--ur_lat",
        type=float,
        default=90.,
        help="Upper right latitude of reduced domain (if domain_clip=True)",
    )
    parser.add_argument(
        "--ur_lon",
        type=float,
        default=180.,
        help="Upper right longitude of reduced domain (if domain_clip=True)",
    )
    parser.add_argument(
        "--clear_inputs",
        type=bool,
        default=True,
        help="Remove input files (True) or not (False)"
    )
    args = parser.parse_args()

    return args


def save_output(year, month, var, variable_data):

    OUTPUTS_PATH=os.environ.get("OUTPUTS_PATH")
    
    long_name = variable_data[var].long_name
    units = variable_data[var].units
    variable_data.attrs = {}
    variable_data[var].attrs = {}
    variable_data = variable_data.assign_attrs(
        long_name = long_name,
        units = units,
    )
    variable_data.to_netcdf(f"{OUTPUTS_PATH}/{var}_{year}-{month:02d}_init-test.nc")

def correct_coords(variable_data, real_lats, real_lons):
    variable_data['latitude'] = real_lats.values
    variable_data['longitude'] = real_lons.values
    return variable_data

def read_output_vars(year, month, var):

    WORKDIR_PATH=os.environ.get("WORKDIR_PATH")
    
    variable_names = {
        'FWI': 'fwinx',
        'DC': 'drtcode',
        'DMC': 'dufmcode',
        'BUI': 'fbupinx',
        'FFMC': 'ffmcode',
        'ISI': 'infsinx',
    }
    variable_grib_name = variable_names[var]
    variable_data = xr.open_mfdataset(f"{WORKDIR_PATH}/{year}{month:02d}*_{variable_grib_name}.grib",engine="cfgrib",concat_dim="time",combine="nested",decode_timedelta=False)
    variable_data = variable_data.rename_vars({variable_grib_name: var})
    return variable_data

if __name__ == "__main__":
    main()
