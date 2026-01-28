CARMINE – Project
This folder contains the scripts used to compute zone-based climate stress indicators for the CARMINE Climate- Resilient Development Pathways in
Metropolitan Regions of Europe. The scripts are written in R and rely on commonly used R packages for NetCDF handling, climate indices, and statistical analysis.

Package Requirements
ncdf4, lubridate, heatwaveR, SPEI, zoo, rslurm, snowfall


The workflow focuses on the construction of:

Heat magnitude day (HMD)

Standardized Precipitation Evapotranspiration Index (SPEI)

Combined Stress Index (CSI), representing compound heat–drought stress


The scripts are designed to work with gridded climate data provided in NetCDF format.


General workflow

The processing chain follows these steps:

1) Compute daily HMD from daily temperature data

2) Merge daily HMD blocks into a single daily dataset

3) Aggregate daily HMD into monthly HMD by summing daily values

4) Compute daily Potential Evapotranspiration (PET)

5) Compute monthly SPEI (scale = 1) from precipitation and PET

6) Compute the Combined Stress Index (CSI) using PCA between monthly HMD and SPEI

7) Identify CSI extreme months using a percentile threshold

8) Merge all temporary block outputs into final NetCDF files

Description of scripts


HMD_calc.R
Computes daily HMD for each grid cell using a block-based approach.

HMD_merge_blocks.R
Merges daily HMD blocks into a single daily HMD file per zone and ensemble.



hmd_monthly_sum_blocks.R
Aggregates daily HMD into monthly HMD, defined as the sum of daily HMD values within each month.

merge_hmd_monthly_sum_blocks.R
Merges monthly HMD blocks into a single monthly dataset.

hw_metrics_calc.R
Calculation of monthly heatwave metrics.

PET_calculation.R
Computes daily Potential Evapotranspiration (PET).

SPEI_calculation.R
Computes monthly SPEI (scale = 1) from the monthly water balance (precipitation minus PET).



csi_pca_blocks.R
Computes the Combined Stress Index (CSI) as the first principal component (PC1) of a PCA between:

monthly HMD

monthly SPEI

The PCA is performed per grid cell using standardized time series, so that CSI represents a single compound heat–drought stress index.
CSI extreme months are optionally defined as values exceeding the 90th percentile of the CSI time series.

merge_csi_blocks.R
Merges CSI block outputs into a final CSI dataset, including PCA coefficients.

Data availability

Due to data volume constraints, input and output datasets are not stored in this repository.
They are available via the associated Zenodo repository referenced in MS4.

This GitHub folder provides the scripts used for the computation, ensuring transparency and reproducibility.

Contributors

Isidoros Iakovidis (CMCC)

Niklas Luther (JLU)

Céline Müller (JLU)

Elena Xoplaki (CMCC)