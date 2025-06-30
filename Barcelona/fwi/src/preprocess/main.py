import calendar
import argparse
import os 
from dotenv import load_dotenv

from src.preprocess.preprocess_temperature import preprocess_temperature
from src.preprocess.preprocess_relative_humidity import preprocess_relative_humidity
from src.preprocess.preprocess_precipitation import preprocess_precipitation
from src.preprocess.preprocess_wind import preprocess_wind
from src.preprocess.preprocess_snowcover import preprocess_snowcover
from src.preprocess.preprocess_landmask import preprocess_landmask
from src.preprocess.save_to_grib import save_to_grib
from src.preprocess.create_namelist import create_namelist
from src.utils.classes import latlon_box

load_dotenv()

def main():

    args = parse_input_arguments()

    year: int = args.year
    month: int = args.month
    which_precipitation: str = args.which_precipitation
    wind_limit: float = args.wind_limit
    domain_clip: bool = args.domain_clip    
    ll_lat: float = args.ll_lat
    ll_lon: float = args.ll_lon
    ur_lat: float = args.ur_lat
    ur_lon: float = args.ur_lon

    reduced_domain = latlon_box(
        ur_lat=ur_lat,
        ur_lon=ur_lon,
        ll_lat=ll_lat,
        ll_lon=ll_lon,
    )

    WORKDIR_PATH=os.environ.get("WORKDIR_PATH")
    os.makedirs(WORKDIR_PATH, exist_ok=True)

    this_month_length = calendar.monthrange(year,month)[1]

    print("Temperature")
    temperature = preprocess_temperature(year=year,month=month,reduced_domain=reduced_domain,domain_clip=domain_clip)
    create_inputs(variable_name="temperature", variable_data=temperature, this_month_length=this_month_length)

    print("Snow cover")
    snow_cover = preprocess_snowcover(temperature=temperature)
    create_inputs(variable_name="snow_cover", variable_data=snow_cover, this_month_length=this_month_length)
    del snow_cover

    print("Relative humidity")
    relative_humidity = preprocess_relative_humidity(year=year,month=month,temperature=temperature,reduced_domain=reduced_domain,domain_clip=domain_clip)
    create_inputs(variable_name="relative_humidity", variable_data=relative_humidity, this_month_length=this_month_length)
    del relative_humidity, temperature 

    print("Precipitation")
    precipitation = preprocess_precipitation(year=year,month=month,which_precipitation=which_precipitation,reduced_domain=reduced_domain,domain_clip=domain_clip)
    create_inputs(variable_name="precipitation", variable_data=precipitation, this_month_length=this_month_length)
    del precipitation 
    
    print("Wind speed")
    wind_speed = preprocess_wind(year=year,month=month,reduced_domain=reduced_domain,domain_clip=domain_clip,wind_limit=wind_limit)
    create_inputs(variable_name="wind_speed", variable_data=wind_speed, this_month_length=this_month_length)
    
    print("Land mask")
    land_mask = preprocess_landmask(data=wind_speed)
    create_inputs(variable_name="land_mask", variable_data=land_mask, this_month_length=this_month_length)    
    del wind_speed, land_mask
    
    print("Create namelists")
    create_all_namelists(year=year, month=month, this_month_length=this_month_length)

def create_inputs(variable_name, variable_data, this_month_length):
    for this_day in range(this_month_length):
        this_day_str = f"{this_day:02d}"
        start_day = this_day
        end_day = this_day+1
        days_selection = slice(start_day,end_day)

        variable_data_this_step = variable_data.isel(time=days_selection)
        
        save_to_grib(variable_name=variable_name, data_values=variable_data_this_step, number=this_day_str)

def create_all_namelists(year, month, this_month_length):
    for this_day in range(this_month_length):
        this_day_str = f"{this_day:02d}"
        start_day = this_day
        end_day = this_day+1
    
        if (month==1) & (start_day==0):
            initialization="artificial"
        else:
            initialization="from_file"

        create_namelist(
            year=year,
            month=month,
            calc_num=this_day_str,
            start_day=start_day,
            end_day=end_day,
            initialization=initialization,
            )

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
        "--which_precipitation",
        type=str,
        required=True,
        choices=['daily', '6h'],
        help="Type of input precipitation data (forecast, historical)",
    )
    parser.add_argument(
        "--wind_limit",
        type=float,
        default=None,
        help="Wind upper limit in m/s",
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
    args = parser.parse_args()

    return args
    
if __name__ == "__main__":
    main()
