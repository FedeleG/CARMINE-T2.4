README for uncertainty-assessment.py
====================================

Overview
--------
This script compares the spatial distribution and mean values of a selected climate indicator across multiple climate datasets for a given pilot area. 
It is designed to support inter-dataset consistency checks by:
- Comparing the spread of spatial values.
- Quantifying agreement between datasets using spatial means.

Indicator
---------
- Indicator: rr
- Variable name: RR
- Meaning: Total precipitation (rainfall)
- Units: Typically millimeters (mm) or kg m⁻² (dataset-dependent)

Datasets Supported
------------------
The script processes the following datasets if available:

| Dataset       | Period covered |
|---------------|----------------|
| era5-2km      | 1989–2018      |
| CERRA         | 1981–2010      |
| E-OBS         | 1981–2010      |
| EMO           | 1991–2020      |

Directory Structure
-------------------
The script assumes the following structure:

/work/CARMINE/
└── CARMINE-T2.4/
    └── <PilotArea>/
        └── INDICATORS/
            └── <pilot>_<dataset>_<indicator>_<period>.nc

Example:
barcelona_cerra_rr_eu_1981_2010.nc

Script Functionality
--------------------
1. File Discovery and Loading
   - Builds dataset-specific filenames automatically.
   - Loads NetCDF files using xarray.
   - Extracts the selected variable (RR) and flattens all spatial grid points into a 1D array.
   - Skips datasets if files or variables are missing.

2. Spatial Distribution Analysis
   - Produces a boxplot showing the distribution of all spatial values.
   - Each box corresponds to one dataset.
   - Useful for detecting systematic biases, spread differences, and outliers.

3. Spatial Mean Computation
   - Computes the mean precipitation value across all spatial grid points.
   - One mean value per dataset.

4. Pairwise Agreement Assessment
   - Computes absolute differences between spatial means for every dataset pair.
   - Displays results as a heatmap.
   - Lower values indicate better agreement between datasets.

Output Plots
------------
1. Boxplot
   - X-axis: Dataset
   - Y-axis: Indicator values (RR)
   - Shows spatial variability within each dataset.

2. Pairwise Agreement Heatmap
   - Matrix of absolute differences between spatial means.
   - Annotated values for direct comparison.
   - Useful for identifying datasets that diverge systematically.

Dependencies
------------
- numpy
- xarray
- matplotlib
- seaborn

Intended Use
------------
- Climate dataset intercomparison.
- Quality control and consistency checks.
- Supporting methodological sections in reports or deliverables.
- Exploratory analysis prior to spatial aggregation or impact assessment.

Notes
-----
- Temporal aggregation is assumed to be already performed in the input files.
- No spatial masking is applied: all grid points in the NetCDF files are used.
- The script does not modify or save data; it produces diagnostic plots only.
