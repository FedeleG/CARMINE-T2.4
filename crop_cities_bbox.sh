#!/bin/bash
#
# ============================================================
# Script name : crop_cities_bbox.sh
# Purpose     : Crop NetCDF files over predefined city buffers
#               using CDO sellonlatbox
# Author      : <your name / institute>
# Requirements: CDO, NetCDF files with lon/lat coordinates
# ============================================================


# ============================================================
# INPUT / OUTPUT DIRECTORIES
# INDIR   : directory containing input NetCDF files
# OUTBASE : base directory for city-wise cropped outputs
# ============================================================

INDIR="/data/cmcc/.."
OUTBASE="/data/cmcc/.."

# Create output base directory if it does not exist
mkdir -p "$OUTBASE"


# ============================================================
# 1° BUFFER BOUNDING BOXES FOR CITIES
# Each entry contains:
#   lon_min lon_max lat_min lat_max
# Coordinates are in decimal degrees (EPSG:4326)
# ============================================================

declare -A CITY_BOX

CITY_BOX["Athens"]="22.73 24.73 36.98 38.98"
CITY_BOX["Barcelona"]="1.17 3.17 40.38 42.38"
CITY_BOX["Birmingham"]="-2.90 -0.90 51.48 53.48"
CITY_BOX["Bologna"]="10.34 12.34 43.49 45.49"
CITY_BOX["Brasov"]="24.60 26.60 44.66 46.66"
CITY_BOX["Funen-Odense"]="9.39 11.39 54.40 56.40"
CITY_BOX["Leipzig"]="11.37 13.37 50.34 52.34"
CITY_BOX["Prague"]="13.42 15.42 49.08 51.08"


# ============================================================
# LOOP OVER CITIES AND INPUT FILES
# For each city:
#   - create a dedicated output directory
#   - crop all NetCDF files using the city bounding box
# ============================================================

for CITY in "${!CITY_BOX[@]}"; do

    # Read bounding box values for the current city
    read LON_MIN LON_MAX LAT_MIN LAT_MAX <<< "${CITY_BOX[$CITY]}"

    # Log current processing status
    echo "=============================================="
    echo "Processing city: $CITY"
    echo "Bounding box: lon [$LON_MIN,$LON_MAX], lat [$LAT_MIN,$LAT_MAX]"
    echo "=============================================="

    # Create city-specific output directory
    CITY_OUT="${OUTBASE}/${CITY}"
    mkdir -p "$CITY_OUT"

    # Loop over all NetCDF files in input directory
    for FILE in "${INDIR}"/*.nc; do

        # Extract filename without path
        BASENAME=$(basename "$FILE")

        # Define output filename with city prefix
        OUT="${CITY_OUT}/${CITY}_${BASENAME}"

        # Crop dataset using CDO sellonlatbox
        # -O : overwrite output if it exists
        cdo -O sellonlatbox,$LON_MIN,$LON_MAX,$LAT_MIN,$LAT_MAX \
            "$FILE" "$OUT"

        # Log processed file
        echo "Processed: $OUT"
    done
done


# ============================================================
# END OF SCRIPT
# ============================================================

echo "CSA city cropping completed successfully"
