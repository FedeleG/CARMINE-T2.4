README for uncertainty-assessment_interpolate.py
=================================================

Overview
--------
This script performs an **uncertainty and agreement assessment** of a selected climate indicator across multiple datasets for a given pilot area, with additional handling for **spatial grid differences**. 

Unlike the basic uncertainty-assessment, this version:
- Detects the **coarsest spatial grid** among datasets
- **Interpolates all datasets** to the coarsest grid
- Computes pairwise statistics only over the **common valid mask**
This allows meaningful calculation of metrics such as correlation, RMSE, and bias that would otherwise be inconsistent due to differing grid resolutions.

Indicator
---------
- Indicator: rr
- Variable name: RR
- Meaning: Total precipitation (rainfall)
- Units: Typically millimeters (mm) or kg m⁻² (dataset-dependent)

Datasets Supported
------------------
- era5-2km      (1989–2018)
- CERRA         (1981–2010)
- E-OBS         (1981–2010)
- EMO           (1991–2020)

Directory Structure
-------------------
The script assumes the following directory structure:

/work/cmcc/gf27821/CARMINE/CARMINE-T2.4/<PilotArea>/INDICATORS/

Files should follow the naming convention:
<pilot>_<dataset>_<indicator>_<period>.nc

Example:
barcelona_cerra_rr_eu_1981_2010.nc

Script Functionality
--------------------

1. File Discovery and Loading
   - Automatically builds dataset filenames
   - Loads NetCDF files using xarray
   - Identifies spatial coordinates (supports multiple conventions)
   - Skips missing files or datasets without the target variable

2. Grid Resolution Detection
   - Estimates approximate grid cell area for each dataset
   - Determines the **coarsest grid**, which serves as reference for interpolation

3. Interpolation to Reference Grid
   - All datasets are linearly interpolated onto the coarsest grid
   - Ensures **common spatial coordinates** for pairwise comparisons

4. Masking
   - Creates a **common valid mask** (finite values across all datasets)
   - Applies the mask to all interpolated datasets

5. Field Diagnostics
   - Prints standard deviations of each dataset after interpolation and masking
   - Helps detect loss of variability due to interpolation

6. Pairwise Statistics
   For each dataset pair, computes:
   - **Bias** (mean difference)
   - **RMSE** (root mean squared error)
   - **Pearson correlation coefficient**
   - **Spearman rank correlation coefficient**

7. Plotting
   - Heatmaps for all pairwise metrics:
     - Pearson correlation
     - Spearman correlation
     - RMSE
     - Bias
   - Annotated matrices for quick visual assessment

Dependencies
------------
- numpy
- xarray
- matplotlib
- seaborn
- scipy

Intended Use
------------
- Assessing **agreement and uncertainty** between climate datasets
- Producing **interpolated comparisons** for datasets with different spatial resolutions
- Quality control and exploratory analysis prior to spatial aggregation or impact assessment

Notes
-----
- Temporal aggregation should be pre-computed in the input files.
- All metrics are computed **after regridding** to ensure consistency.
- No data is modified; outputs are **diagnostic plots only**.
