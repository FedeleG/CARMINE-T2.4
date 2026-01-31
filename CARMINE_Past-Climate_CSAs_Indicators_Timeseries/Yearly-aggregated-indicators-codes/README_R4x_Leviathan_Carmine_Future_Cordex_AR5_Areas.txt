Within this directory the R codes used to compute yearly aggregated climate indicator time series for EURO-CORDEX ensemble members are included.
The analyses refer to past and future periods (milestone 4 – MS4; Table 1).

The file R4x_Leviathan_Carmine_Future_Cordex_AR5_Areas.txt contain R-code content. Please copy and paste the code into R for usage.

Author – Jonathan Spinoni (CMCC)

README – How to Run the R Scripts
Requirements

  R (version 4.0 or newer recommended)
  
  Sufficient RAM (scripts can be memory intensive)

Required R packages:

  ncdf4
  
  SPEI
  
  cffdrs

Install missing packages with:

  install.packages(c("ncdf4", "SPEI", "cffdrs"))

Input Data

  All input data must be provided as NetCDF files
  
  Files must be stored in a local directory and named exactly as expected by the scripts
  
  The working directory must point to the folder containing the input NetCDF files

Before Running

  Before executing the scripts, check that:
  
  The selected datasets are available in the working directory
  
  Spatial and temporal coverage are consistent with the script configuration
  
  The machine has enough memory for the selected domain

How to Run

  Open R or RStudio
  
  Set the working directory to the folder containing the script and input data:
  
  setwd("path/to/your/data/")


Run the script:

  source("script_name.R")


Progress messages will be printed during execution.

Outputs
  
  The scripts generate NetCDF output files containing climate indicators and percentiles.
  Output files are written to the working directory.
  
Notes
  
  Execution time may be long (from minutes to hours)

The scripts are designed to run on laptops

Large spatial domains increase memory usage
