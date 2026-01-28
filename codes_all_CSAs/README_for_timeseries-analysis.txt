# README – Timeseries Analysis for Climate Indicators in CSAs/FUAs

This Python script analyzes **annual timeseries of climate indicators (e.g., `RRx1d`) over Functional Urban Areas (FUAs) or Combined Statistical Areas (CSAs). It is designed to **assess the temporal coherence among multiple datasets**.

---

## Purpose

- Compute spatially-averaged annual timeseries** of climate indicators inside a FUA.  
- Generate boxplots and **annual timeseries plots**.  
- Compute Pearson and Spearman correlations** among datasets over a common period (e.g., 1991–2020).  
- Handle missing years by filling them with `NaN` to align datasets for correlation analysis.

> Unlike the other script (`uncertainty-assessment_interpolate_timeseries.py`), this script focuses on temporal coherence, while the other is oriented toward spatial patterns averaged over time**.

---

## Requirements

- Python 3.10+  
- Packages:

numpy
xarray
geopandas
pandas
matplotlib
seaborn
shapely
scipy

Install via pip:

pip install numpy xarray geopandas pandas matplotlib seaborn shapely scipy


---

## Configuration

Set parameters at the top of the script:

pilotarea = "Barcelona" # Target CSA/FUA
indicator = "RRx1d" # Climate indicator to analyze
var_in_nc = "RRx1d" # Variable name in NetCDF
unit = "mm" # Unit for plotting
start_year = 1991 # Start of analysis period
end_year = 2020 # End of analysis period


- `DATASETS`: dictionary with dataset names and their original year ranges.  
- `FUA_SHP`: path to FUA shapefile.

---

## Workflow

1. Load CSA/FUA shapefile and extract the FUA boundary.  
2. Load each dataset from NetCDF and select the target variable.  
3. Mask dataset to the FUA polygon.  
4. Compute **spatial mean** for each year.  
5. Align all datasets to the **common period 1991–2020**.  
6. Plot:
   - Boxplots of indicator values within the FUA  
   - Annual timeseries per dataset  
   - Pearson and Spearman correlation heatmaps  

---

## Input Data

- NetCDF files for each climate dataset (e.g., `CERRA`, `EOBS`, `EMO1`, `ERA5`, `ERA2km`) with `time`/`year` coordinates.  
- FUA shapefile defining urban boundaries.

---

## Output

- **Boxplots** of indicator values within the FUA.  
- Annual timeseries plots for all datasets.  
- Correlation heatmaps (Pearson and Spearman) for annual means.

---

## Usage

python timeseries-analysis.py


Plots will appear interactively. Missing years are handled automatically, and correlations are computed only on overlapping data.

---

## Notes

- This script is timeseries-focused, in contrast to the script `uncertainty-assessment_interpolate_timeseries.py` which is spatially-focused and examines FUA-masked maps and metrics.
