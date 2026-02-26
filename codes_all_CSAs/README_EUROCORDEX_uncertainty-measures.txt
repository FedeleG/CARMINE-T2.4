# EURO-CORDEX Indicator Sensitivity & Uncertainty Analysis

## CARMINE WP2.4 – Multi-CSA Trend & Uncertainty Diagnostics
- These analysis scripts used to assess structural uncertainty in climate indicators derived from the EURO-CORDEX ensemble for the CARMINE case study areas (CSAs).

## Requirements

carmine environment available in this repository "carmine_env_yml"

Install via conda:

conda env create -f environment.yml

---

## Associated scripts:

- EUROCORDEX_uncertainty_directional_agreement.py
- EUROCORDEX_uncertainty_robustness_heatmap.py
- EUROCORDEX_uncertainty_trend_magnitude.py

---

## Conceptual Framework

The analysis separates three dimensions of uncertainty:

- Trend magnitude → How strong is the projected change?
- Robustness → Is the signal stronger than model spread?
- Directional agreement → Do models agree on trend sign?

Together, these metrics quantify structured uncertainty across:

- Indicator type
- Scenario forcing
- Time horizon
- Spatial context (CSA)

Each script is self-contained and can be executed independently.

All scripts generate the master summary table:

analysis_outputs/tables/
    robustness_allCSAs_allIndicators.csv

This table contains:

- median_slope_decade
- IQR_slope_decade
- robustness_index
- spread_ratio
- frac_positive
- frac_negative
- n_models
- Units


The scripts work along the predefined internal CARMINE Windows and generate output for all of the following:

Label	Years
near	1981–2010
near2	1991–2020
mid     2021–2050
mid2	2036–2065
late	2071–2100
full	1971–2100

---

## Data Access

The EURO-CORDEX annual indicator dataset used in this analysis is publicly available via Zenodo:

https://zenodo.org/records/18454954

---

## Download Instructions

Download the full dataset archive from Zenodo.

Unzip the archive locally.

Ensure the extracted folder is named:

Cordex_Carmine

Place the folder in a location accessible from your local Python environment.

Required Folder Structure

After extraction, the directory structure should look like:

Cordex_Carmine/
│
├── Athens/
│   ├── CARMINE_Cordex_CMIP5_Athens_mod01_...
│   ├── ...
│
├── Barcelona/
│   ├── ...
│
├── ...
│
└── Ensemble_Outputs/

Important:

- The scripts operate on the CSA subfolders.
- The "Ensemble_Outputs" directory is not required for this analysis and is ignored.
- All NetCDF files must remain directly inside their respective CSA folders.

Script Configuration:

- In each script, set the local path to the unzipped data:

folderlocation = Path(r"set path to your location of Cordex_Carmine")
ROOT = folderlocation / "Cordex_Carmine"

Example (Windows):

folderlocation = Path(r"C:\Users\username\Data")
ROOT = folderlocation / "Cordex_Carmine"

#####################################################
################## Data Size ########################
#####################################################


The dataset contains:
7 CSAs
13 RCM realizations
~22 indicators
3 scenarios
Annual values for 1971–2100
Approximate size after extraction: ~1–2 GB.

