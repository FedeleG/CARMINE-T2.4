import pathlib 
from dotenv import load_dotenv
import os 

load_dotenv()

def create_namelist(year: int, month: int, calc_num: int, start_day: int, end_day: int, initialization: str):

    WORKDIR_PATH=os.environ.get("WORKDIR_PATH")
    
    old_words, new_words = namelist_inputs(calc_num,year,month,start_day,end_day,initialization)

    file_path = pathlib.Path(__file__).parent.resolve()
    input_file = open(f"{file_path}/gefftemplate", "r")
    filedata = input_file.read()
    input_file.close()

    for n_word in range(len(old_words)):
        old_word = old_words[n_word]
        new_word = new_words[n_word]
        filedata = filedata.replace(f"{old_word}", f"{new_word}")
    
    file_out = open(f"{WORKDIR_PATH}/geff_{calc_num}.namelist", "w")
    file_out.write(filedata)
    file_out.close()

def namelist_inputs(calc_num,year,month,start_day,end_day,initialization):

    OUTPUT_FILE_NAME=f"'fwi_output_{calc_num}.grib'"
    INITIAL_DATE=f"{year}{month:02d}{(start_day+1):02d}"
    if initialization=="artificial":
        INPUT_RESTART_NAME="''"
    elif initialization=="from_file":
        INPUT_RESTART_NAME="'input_restart.grib'"
    else:
        raise ValueError(f"No valid initialization: {initialization}. Valid options: artificial, from_file")
    RESTART_DAY=end_day-start_day
    TIME_NOW=INITIAL_DATE

    TEMPERATURE_FILE_NAME=f"'./temperature_{calc_num}_EMO1.grib'"
    RELATIVE_HUMIDITY_FILE_NAME=f"'./relative_humidity_{calc_num}_EMO1.grib'"
    PRECIPITATION_FILE_NAME=f"'./precipitation_{calc_num}_EMO1.grib'"
    WIND_SPEED_FILE_NAME=f"'./wind_speed_{calc_num}_EMO1.grib'"
    SNOW_COVER_FILE_NAME=f"'./snow_cover_{calc_num}_EMO1.grib'"
    LAND_MASK_FILE_NAME=f"'./land_mask_{calc_num}_EMO1.grib'"
   
    new_words = [
        OUTPUT_FILE_NAME,
        INITIAL_DATE,
        INPUT_RESTART_NAME,
        RESTART_DAY,
        TIME_NOW,
        TEMPERATURE_FILE_NAME,
        RELATIVE_HUMIDITY_FILE_NAME,
        PRECIPITATION_FILE_NAME,
        WIND_SPEED_FILE_NAME,
        SNOW_COVER_FILE_NAME,
        LAND_MASK_FILE_NAME,
    ]
    old_words = [
        "OUTPUT_FILE_NAME",
        "INITIAL_DATE",
        "INPUT_RESTART_NAME",
        "RESTART_DAY",
        "TIME_NOW",
        "TEMPERATURE_FILE_NAME",
        "RELATIVE_HUMIDITY_FILE_NAME",
        "PRECIPITATION_FILE_NAME",
        "WIND_SPEED_FILE_NAME",
        "SNOW_COVER_FILE_NAME",
        "LAND_MASK_FILE_NAME",
    ]
    return old_words, new_words
