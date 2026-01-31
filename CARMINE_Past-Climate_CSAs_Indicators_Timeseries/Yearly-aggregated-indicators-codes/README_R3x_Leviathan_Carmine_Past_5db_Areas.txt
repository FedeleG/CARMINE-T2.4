Within this directory the code used to obtain yearly aggregated timeseries over the past (milestone 4 - MS4; Table 1) are included.
The code R3x_Leviathan_Carmine_Past_5db_Areas.txt is a R-code content. Please copy and paste on R for usage.
Author - Jonathan Spinoni (CMCC)
-----------------------------------------------------------------------------------------------------------------------------------
README – How to Run the R Script

Requirements
------------
- R (version 4.0 or newer recommended)
- Sufficient RAM (the script can be memory intensive)

Required R packages:
- ncdf4
- SPEI
- cffdrs

Install missing packages with:
install.packages(c("ncdf4", "SPEI", "cffdrs"))


Input Data
----------
All input data must be provided as NetCDF files.
Files must be stored in a local directory and named exactly as expected by the script.
The working directory must point to the folder containing the input NetCDF files.


Before Running
--------------
Before executing the script, check that:
- The selected dataset is available in the working directory
- The spatial and temporal coverage of the data is consistent
- The machine has enough memory for the selected domain


How to Run
----------
1. Open R or RStudio
2. Set the working directory to the folder containing the script and input data:
   setwd("path/to/your/data/")
3. Run the script:
   source("script_name.R")

Progress messages will be printed during execution.


Outputs
-------
The script generates NetCDF output files containing climate indicators and percentiles.
Output files are written to the working directory.


Notes
-----
- Execution time may be long (from minutes to hours)
- The script is designed to run on laptops
- Large spatial domains increase memory usage
