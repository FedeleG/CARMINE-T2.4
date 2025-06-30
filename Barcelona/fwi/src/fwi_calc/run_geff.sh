#!/bin/bash
set -e

FILE_PATH=$(dirname "$(realpath $0)")
source .env

cd ${WORKDIR_PATH}

for filename in ./geff_*.namelist; do
    if [ -f ./output_restart.grib ]; then
        cp output_restart.grib input_restart.grib
    fi
    if [ -L ./geff.namelist ]; then
        rm ./geff.namelist
    fi

    ln -s $filename ./geff.namelist
    ${GEFF_BIN_PATH}/geff

    filename_split=(${filename//_/ })
    this_day=${filename_split[1]:0:2} 
    ${ECCODES_PATH}/grib_filter ${FILE_PATH}/rules_file fwi_output_${this_day}.grib
    rm fwi_output_${this_day}.grib
    rm $filename
done
