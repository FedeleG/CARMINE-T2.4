#!/bin/bash
set -e
source .env

firstyear=$1
lastyear=$2

mkdir -p ${OUTPUTS_PATH}
mkdir -p ${METRICS_PATH}
mkdir -p ${LOGS_PATH}
mkdir -p ${WORKDIR_PATH}

for year in $(seq "$firstyear" "$lastyear");
do
    for var in tx tn pd ws pr6;
    do
        wget -nc -P ${INPUT_DATA_PATH}/ https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-EFAS/meteorological_forcings/EMO-1arcmin/${var}/EMO-1arcmin-${var}_${year}.nc
    done
    
    for month in {1..12};do
        echo ${year}-${month}
        poetry run python -m src.preprocess.main --year ${year} --month ${month} --which_precipitation "6h" --wind_limit 15.0 --domain_clip True --ll_lat 40.0 --ll_lon -1.5 --ur_lat 43.0 --ur_lon 3.5
        bash src/fwi_calc/run_geff.sh > ${LOGS_PATH}/geff_logs_${year}_$(printf "%02d" ${month}).txt
        poetry run python -m src.postprocess.postprocess --year ${year} --month ${month} --clear_inputs True --domain_clip True --ll_lat 40.0 --ll_lon -1.5 --ur_lat 43.0 --ur_lon 3.5
    done
    
    for var in tx tn pd ws pr6;
    do
        rm ${INPUT_DATA_PATH}/EMO-1arcmin-${var}_${year}.nc
    done
    
    rm ${WORKDIR_PATH}/output_restart.grib
done