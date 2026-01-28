This Python script generates maps of ensemble mean and standard deviation for a climate indicator (e.g., tn10prctile) over a selected Climate Smart Area (CSA). The spatial domain is defined using a shapefile of the Functional Urban Area (FUA), and the data are based on EURO-CORDEX projections.

Overview

The workflow is designed for pilot areas identified as FUAs. It reads precomputed ensemble mean and standard deviation from NetCDF files for multiple RCP scenarios, then visualizes the results as maps cropped to the FUA domain.

Key features include:

  Reading data: Loads ensemble mean and standard deviation for a given climate indicator from NetCDF files.
  
  Spatial masking: Restricts the maps to the FUA polygon with an optional buffer (default 1°).
  
  Scenario handling: Supports multiple RCP scenarios (e.g., RCP2.6, RCP4.5, RCP8.5).
  
  Dynamic color scaling: Colorbars are automatically set based on the min/max values within the FUA.
  
  Visualization: Produces a figure with 2 rows (ensemble mean and standard deviation) and 1 column per scenario.


Data Requirements

  NetCDF files must follow the naming convention:
  
  {CSA}_EU-CORDEX-11_YEAR_{indicator}_{baseline}_{future}_RCP{scenario}.nc
  
  
  indicator must match the NetCDF variable (var_in_nc).
  
  Files must include both {var_in_nc} (ensemble mean) and {var_in_nc}_STD (ensemble standard deviation).
  
  FUA shapefile containing a FUA_NAME field:
  
  FUA_Boundaries.shp


Example file structure:

  /data/.../CARMINE/CORDEX/CSA/Bologna/
      Bologna_EU-CORDEX-11_YEAR_tn10prctile_1981-2010_2036-2065_RCP26.nc

Configuration

  Set the following parameters at the top of the script:
  
  pilotarea — Name of the pilot area (e.g., "Bologna")
  
  indicator — Climate indicator (e.g., "tn10prctile")
  
  var_in_nc — Variable name in NetCDF ("TN10PRCTILE")
  
  unit — Units for plotting ("degreeC")
  
  scenarios — List of RCP scenarios (["26","45","85"])
  
  baseline / future — Reference and projection periods
  
  root_path — Path to CSA NetCDF files
  
  FUA_SHP — Path to the FUA shapefile
  
  The FUA name is automatically mapped via FUA_MAPPING.


Output

  A single figure per pilot area showing ensemble mean (row 1) and ensemble standard deviation (row 2) for all scenarios.
  
  The figure is cropped to the FUA polygon with a 1° buffer and saved as:
  
  UNCERTAINTY_{CSA}_{indicator}_all_RCP.png

Notes

  Ensemble mean and standard deviation must already be computed in the NetCDF files; the script does not recalculate them.
  
  The script automatically masks values outside the FUA polygon.
  
  Colorbars are dynamically scaled based on the masked FUA values.
  
  The current implementation focuses on one pilot area at a time.
