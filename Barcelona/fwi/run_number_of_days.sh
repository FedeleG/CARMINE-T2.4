#!/bin/bash
set -e
source .env

# Define your input parameters
THRES_LOW=$1
THRES_HIGH=$2
LEVEL=$3
YEAR_START=$4
YEAR_END=$5

mkdir -p ${OUTPUTS_PATH}/${LEVEL}_days

# Process each year
for year in $(seq $YEAR_START $YEAR_END); do
    # Find all monthly files for this year
    files=$(ls ${OUTPUTS_PATH}/FWI_${year}-*_init-test.nc)
    
    if [ -n "$files" ]; then
        cdo -L -mergetime $files "${OUTPUTS_PATH}/${LEVEL}_days/merged_${year}.nc"
        
        # 1. Create mask for values >= lower threshold
        cdo -L -gtc,$THRES_LOW "${OUTPUTS_PATH}/${LEVEL}_days/merged_${year}.nc" "${OUTPUTS_PATH}/${LEVEL}_days/mask_low_${year}.nc"
        
        # 2. Create mask for values <= upper threshold
        cdo -L -lec,$THRES_HIGH "${OUTPUTS_PATH}/${LEVEL}_days/merged_${year}.nc" "${OUTPUTS_PATH}/${LEVEL}_days/mask_high_${year}.nc"
        
        # 3. Multiply masks to get values between thresholds
        cdo -L -mul "${OUTPUTS_PATH}/${LEVEL}_days/mask_low_${year}.nc" "${OUTPUTS_PATH}/${LEVEL}_days/mask_high_${year}.nc" "${OUTPUTS_PATH}/${LEVEL}_days/mask_between_${year}.nc"
        
        # 4. Count days between thresholds
        cdo -L -yearsum "${OUTPUTS_PATH}/${LEVEL}_days/mask_between_${year}.nc" "${OUTPUTS_PATH}/${LEVEL}_days/FWI_${LEVEL}_days_${year}.nc"
        
        echo "Processed year $year"
        # Clean up temporary files for this year
        rm "${OUTPUTS_PATH}/${LEVEL}_days/merged_${year}.nc" "${OUTPUTS_PATH}/${LEVEL}_days/mask_low_${year}.nc" "${OUTPUTS_PATH}/${LEVEL}_days/mask_high_${year}.nc" "${OUTPUTS_PATH}/${LEVEL}_days/mask_between_${year}.nc"
        echo "Processed year $year"
    else
        echo "No files found for year $year"
    fi
done

# Create a list of all annual count files
annual_files=(${OUTPUTS_PATH}/${LEVEL}_days/FWI_${LEVEL}_days_*.nc)

if [ ${#annual_files[@]} -eq 0 ]; then
    echo "No annual files found for averaging"
    exit 1
fi

# Calculate the multi-year average of annual counts
echo "Calculating multi-year average..."
cdo -L -timmean -mergetime "${annual_files[@]}" ${OUTPUTS_PATH}/${LEVEL}_days/FWI_${LEVEL}_days_mean.nc
