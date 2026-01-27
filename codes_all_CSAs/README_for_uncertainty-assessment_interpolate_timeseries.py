README for uncertainty-assessment_interpolate_timeseries.py
=================================================

Overview
--------
This script performs an **uncertainty and agreement assessment** of a selected climate indicator across multiple datasets for a given pilot area, handling **both differing spatial resolutions and temporal coverage**.  

Unlike `uncertainty-assessment_interpolate.py`, this version:  
- Regrids all datasets to a **shared common grid** (coarsest grid among datasets)  
- Processes **full time series**, computing temporal averages where appropriate  
- Applies a **FUA (Functional Urban Area) mask** to restrict analysis to the pilot area  
- Computes **pairwise metrics over space and/or time**, enabling detailed spatial and temporal agreement assessment  

Indicator
---------
- Indicator: PTOT  
- Variable name: PTOT  
- Meaning: Total precipitation  
- Units: Dataset-dependent, typically millimeters (mm) or kg m⁻²  

Datasets Supported
------------------
- CERRA       (1985–2020)  
- E-OBS       (1950–2024)  
- EMO1        (1990–2024)  
- ERA5        (1980–2022)  
- ERA5-2km    (1989–2018)  

Directory Structure
-------------------
The script assumes the following structure:  

/PATH/CARMINE/CARMINE-T2.4/<PilotArea>/<Dataset>/

NetCDF files should follow the naming convention:  

CARMINE_<Dataset>_<PilotArea>_<Indicator>_BSL_<StartBaseline>_<EndBaseline>_YY_<StartYear>_<EndYear>.nc

Example:  
CARMINE_CERRA_Barcelona_PTOT_BSL_1991_2020_YY_1985_2020.nc

Script Functionality
--------------------
1. File Discovery and Loading
   - Automatically builds dataset filenames  
   - Loads NetCDF files using `xarray`  
   - Identifies spatial coordinates (1D or 2D)  
   - Skips missing files or datasets missing the target variable  

2. Grid Detection and Regridding
   - Determines **coarsest grid** among datasets  
   - Interpolates all datasets to the coarsest grid using nearest-neighbor or linear methods  
   - Ensures consistent spatial coverage for pairwise comparison  

3. FUA Masking
   - Loads the FUA shapefile  
   - Applies polygon mask to retain only grid points inside the pilot area  

4. Temporal Aggregation
   - Computes temporal averages if time dimension exists  
   - Retains full time coverage for metrics computation  

5. Pairwise Statistics
   For each dataset pair, computes:  
   - **Bias** (mean difference)  
   - **RMSE** (root mean squared error)  
   - **Pearson correlation coefficient**  
   - **Spearman rank correlation coefficient**  

6. Plotting
   - Maps of **spatial distributions** for each dataset (FUA-masked)  
   - Heatmaps of pairwise metrics (bias, RMSE, Pearson r, Spearman rho)  

Dependencies
------------
- numpy  
- xarray  
- geopandas  
- matplotlib  
- seaborn  
- scipy  
- cartopy  
- shapely  

Intended Use
------------
- Assessing **spatial and temporal agreement** between multiple climate datasets  
- Evaluating **uncertainty** for climate indicators in pilot regions  
- Producing **regridded and masked pairwise comparisons** for datasets with differing resolutions and temporal spans  
- Quality control and exploratory analysis prior to further climate impact studies  

Notes
-----
- Input NetCDF files should contain pre-aggregated temporal data (daily, monthly, or yearly)  
- All metrics are computed **after regridding and FUA masking**  
- No original data are modified; outputs are **diagnostic plots only**
